"""Export the Perch 2.0 backbone alone, shaped for the STM32N6 / Neural-ART.

The cloud model is one graph: audio -> DFT -> mel -> backbone -> ProtoPNet head.
That is the wrong unit for an MCU. On an N6 the split is:

  M55 + CMSIS-DSP   audio -> log-mel. An FFT is cheap on the CPU and would be
                    wasteful on the NPU, and the log/clamp ops quantise badly -
                    mel energies span orders of magnitude before the log.
  Neural-ART NPU    mel -> 1536-d embedding. Dense convolutions, which is what
                    the accelerator exists for.
  M55               embedding -> UK species. A linear probe is a 1536xN matmul.

Exporting mel->embedding therefore has a second benefit: the fp32 islands that
had to be carved out of the cloud model (the DFT/mel MatMuls, the head's cosine
similarity) are simply not in this graph. Nothing needs excluding, so the whole
thing can be int8 with no CPU fallback - which the July 2026 N6 work identified
as the thing to protect.

Input is (1, 1, 500, 128): 500 frames of a 5 s window at 32 kHz, 128 mel bins.
Note this is smaller than the 500x160 assumed in the earlier N6 feasibility work,
so activation pressure is ~20% lower than that estimate.
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
# STATIC batch=1, no dynamic axes. The stride-2 padding in MBConv is computed
# from the tensor's own spatial size, which under a dynamic batch exports as live
# Shape/Gather/Mod/ConstantOfShape arithmetic. An MCU compiler cannot schedule
# that. Fixing the shape lets it all constant-fold to literal Pad amounts, which
# is also how it will actually run on device: one 5 s window at a time.
torch.onnx.export(
    m, x, "perch2_backbone_n6.onnx",
    input_names=["mel"], output_names=["embedding"],
    opset_version=17, dynamo=False, do_constant_folding=True,
)
import onnx, collections
g = onnx.load("perch2_backbone_n6.onnx").graph
print("ops:", dict(collections.Counter(n.op_type for n in g.node).most_common()))
print("params: %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
