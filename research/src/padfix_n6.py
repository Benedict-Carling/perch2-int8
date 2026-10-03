import sys

import numpy as np
import onnx
from onnx import numpy_helper


def padfix(src: str, dst: str) -> int:
    m = onnx.load(src)
    g = m.graph
    n_fixed = 0
    for n in g.node:
        if n.op_type != "Pad":
            continue
        if len(n.input) >= 3 and n.input[2]:
            continue
        name = (n.name or f"pad{n_fixed}") + "_constant_value"
        g.initializer.append(numpy_helper.from_array(np.array(0.0, dtype=np.float32), name))
        if len(n.input) < 3:
            n.input.append(name)
        else:
            n.input[2] = name
        n_fixed += 1
    onnx.save(m, dst)
    return n_fixed


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    print(f"{src} -> {dst}: {padfix(src, dst)} Pad node(s) fixed")
