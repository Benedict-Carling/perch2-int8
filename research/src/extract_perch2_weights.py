"""Pull Perch 2.0's weights out of the SavedModel checkpoint into a plain .npz.

The SavedModel is a jax2tf export, so tf2onnx can't convert it and its
variables have no layer names (`_tf_var_leaves/N`). Pulling the raw tensors
out lets the model be rebuilt in PyTorch.

Leaf 496 is the species head ([14795, 1536, 4], 90.9M parameters). The
backbone is the remaining ~10.9M.
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
