"""Pull Perch 2.0's weights out of the SavedModel checkpoint into a plain .npz.

The published Perch 2.0 is a jax2tf export with native serialisation: the whole
computation lives inside two opaque XlaCallModule (StableHLO) nodes, and the
variables arrive as an anonymised flattened JAX pytree - `_tf_var_leaves/N`,
with N the position in the flattened tree and no layer names at all.

That opacity is what blocks every cheap route. tf2onnx cannot convert
XlaCallModule, the head cannot be truncated because its shape is baked into the
StableHLO, and the two unused outputs (spatial_embedding, spectrogram) cannot be
pruned. Recovering the raw tensors is the way out: with the weights in hand the
model can be rebuilt in a framework we control.

Leaf 496 is [14795, 1536, 4] - the prototype classifier, 4 prototypes per class,
90.90M of the model's 101.76M parameters. The backbone is the other ~10.9M,
which is EfficientNet-B3 sized.
"""

import re
from pathlib import Path

import numpy as np
import tensorflow as tf

SRC = Path("perch2_savedmodel/variables/variables")
OUT = Path("perch2_weights.npz")


def main() -> None:
    reader = tf.train.load_checkpoint(str(SRC))
    shapes = reader.get_variable_to_shape_map()

    leaves = {}
    for name in shapes:
        m = re.match(r"_tf_var_leaves/(\d+)/", name)
        if m:
            leaves[int(m.group(1))] = reader.get_tensor(name)

    order = sorted(leaves)
    print(f"leaves: {len(order)}  (indices {order[0]}..{order[-1]})")

    by_rank = {}
    total = 0
    for i in order:
        a = leaves[i]
        by_rank.setdefault(a.ndim, []).append(i)
        total += a.size
    print(f"total params: {total/1e6:.2f} M")
    for rank in sorted(by_rank):
        idxs = by_rank[rank]
        n = sum(leaves[i].size for i in idxs)
        print(f"  rank {rank}: {len(idxs):4d} tensors, {n/1e6:8.2f} M params  "
              f"(idx {min(idxs)}..{max(idxs)})")

    np.savez(OUT, **{str(i): leaves[i] for i in order})
    print(f"\nwrote {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
