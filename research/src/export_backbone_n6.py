"""Export the Perch 2.0 backbone (log-mel -> 1536-d embedding) for the STM32N6.

The frontend runs on the CPU and the classifier is trained separately, so the
NPU graph is just the convolutional backbone and can be fully INT8.

Input is (1, 1, 500, 128): 500 frames of a 5 s window at 32 kHz, 128 mel bins.
"""
import numpy as np
import torch
import torch.nn as nn
import perch2_torch


class BackboneN6(nn.Module):
    """log-mel -> 1536-d embedding. Spatial mean folded in, as the probe needs a vector."""

    def __init__(self, net):
        super().__init__()
        self.backbone = net.backbone

    def forward(self, mel):
        return self.backbone(mel).mean(dim=(2, 3))


net = perch2_torch.load()
m = BackboneN6(net).eval()
x = torch.randn(1, 1, 500, 128)
with torch.no_grad():
    print("embedding:", tuple(m(x).shape))
# Fixed batch=1 so MBConv padding folds to constants the MCU compiler can schedule.
torch.onnx.export(
    m, x, "perch2_backbone_n6.onnx",
    input_names=["mel"], output_names=["embedding"],
    opset_version=17, dynamo=False, do_constant_folding=True,
)
import onnx, collections
g = onnx.load("perch2_backbone_n6.onnx").graph
print("ops:", dict(collections.Counter(n.op_type for n in g.node).most_common()))
print("params: %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
