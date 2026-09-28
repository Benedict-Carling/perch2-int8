import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 32_000
WIN_SAMPLES = 160_000
KERNEL, STRIDE, NFFT, N_MELS = 640, 320, 1024, 128
LOG_FLOOR, LOG_OFFSET, LOG_SCALAR = 1e-5, 0.0, 0.1

STEM_FEATURES, HEAD_FEATURES, REDUCTION_RATIO = 32, 1280, 4
STAGES = [
    (1, 16, 3, 1, 1),
    (2, 24, 3, 2, 6),
    (2, 40, 5, 2, 6),
    (3, 80, 3, 2, 6),
    (3, 112, 5, 1, 6),
    (4, 192, 5, 2, 6),
    (1, 320, 3, 1, 6),
]
WIDTH, DEPTH = 1.2, 1.4


def round_features(f: int, w: float = WIDTH, divisor: int = 8) -> int:
    f = f * w
    new = max(divisor, int(f + divisor / 2) // divisor * divisor)
    if new < 0.9 * f:
        new += divisor
    return int(new)


def round_blocks(n: int, d: float = DEPTH) -> int:
    return int(math.ceil(d * n))


def _pad_width(size: int, k: int) -> tuple[int, int]:
    return (k // 2) - (1 - size % 2), k // 2


class MelFrontend(nn.Module):

    def __init__(self, mel_matrix: np.ndarray):
        super().__init__()
        win = torch.from_numpy(np.hanning(KERNEL)).float()
        self.register_buffer("window", win)
        n = torch.arange(KERNEL, dtype=torch.float64)[:, None]
        k = torch.arange(NFFT // 2 + 1, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * k * n / NFFT
        scale = win.double().sum()
        w = (win.double()[:, None] / scale)
        self.register_buffer("dft_cos", (w * torch.cos(ang)).float())
        self.register_buffer("dft_sin", (-w * torch.sin(ang)).float())
        self.register_buffer("mel", torch.from_numpy(mel_matrix).float())

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        pad = STRIDE // 2
        x = F.pad(audio, (pad, pad))
        assert KERNEL == 2 * STRIDE, "framing shortcut assumes kernel == 2 * stride"
        blocks = x.reshape(x.shape[0], -1, STRIDE)
        frames = torch.cat([blocks[:, :-1], blocks[:, 1:]], dim=-1)
        re, im = frames @ self.dft_cos, frames @ self.dft_sin
        mag = torch.sqrt(re * re + im * im + 1e-20)
        mel = mag @ self.mel
        return LOG_SCALAR * torch.log(torch.clamp(mel, min=LOG_FLOOR) + LOG_OFFSET)


class SqueezeAndExcitation(nn.Module):
    def __init__(self, ch: int, reduced: int):
        super().__init__()
        self.Reduce = nn.Linear(ch, reduced)
        self.Expand = nn.Linear(reduced, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))
        s = F.silu(self.Reduce(s))
        s = torch.sigmoid(self.Expand(s))
        return x * s[:, :, None, None]


class MBConv(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, stride: int, expand: int):
        super().__init__()
        feats = cin * expand
        self.stride, self.k, self.expand = stride, k, expand
        if expand != 1:
            self.ExpandConv = nn.Conv2d(cin, feats, 1, bias=False)
            self.ExpandBatchNorm = nn.BatchNorm2d(feats)
        self.DepthwiseConv = nn.Conv2d(
            feats, feats, k, stride=stride,
            padding=0 if stride == 2 else k // 2,
            groups=feats, bias=False,
        )
        self.DepthwiseBatchNorm = nn.BatchNorm2d(feats)
        self.SqueezeAndExcitation_0 = SqueezeAndExcitation(
            feats, feats // (REDUCTION_RATIO * expand)
        )
        self.ProjectConv = nn.Conv2d(feats, cout, 1, bias=False)
        self.ProjectBatchNorm = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.expand != 1:
            x = F.silu(self.ExpandBatchNorm(self.ExpandConv(x)))
        if self.stride == 2:
            ht, hb = _pad_width(x.shape[2], self.k)
            wl, wr = _pad_width(x.shape[3], self.k)
            x = F.pad(x, (wl, wr, ht, hb))
        x = F.silu(self.DepthwiseBatchNorm(self.DepthwiseConv(x)))
        x = self.SqueezeAndExcitation_0(x)
        return self.ProjectBatchNorm(self.ProjectConv(x))


class Backbone(nn.Module):

    def __init__(self):
        super().__init__()
        stem_ch = round_features(STEM_FEATURES)
        self.stem_conv = nn.Conv2d(1, stem_ch, 3, stride=2, padding=0, bias=False)
        self.stem_bn = nn.BatchNorm2d(stem_ch)

        blocks, cin = [], stem_ch
        for n, feat, k, stride, expand in STAGES:
            cout = round_features(feat)
            for b in range(round_blocks(n)):
                blocks.append(MBConv(cin, cout, k, stride if b == 0 else 1, expand))
                cin = cout
        self.blocks = nn.ModuleList(blocks)

        head_ch = round_features(HEAD_FEATURES)
        self.head_conv = nn.Conv2d(cin, head_ch, 1, bias=False)
        self.head_bn = nn.BatchNorm2d(head_ch)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.stem_bn(self.stem_conv(mel)))
        i = 0
        for n, *_ in STAGES:
            for b in range(round_blocks(n)):
                y = self.blocks[i](x)
                x = y if b == 0 else y + x
                i += 1
        return F.silu(self.head_bn(self.head_conv(x)))


class ProtoPNetHead(nn.Module):

    def __init__(self, prototypes: np.ndarray, protop_kernel: np.ndarray, bias: np.ndarray):
        super().__init__()
        k = torch.from_numpy(prototypes).float()
        unit = k / (k.norm(dim=1, keepdim=True) + 1e-5)
        self.n_classes, self.n_proto = unit.shape[0], unit.shape[2]
        self.register_buffer("unit_kernel", unit.permute(1, 0, 2).reshape(unit.shape[1], -1).contiguous())
        self.register_buffer("pk", torch.from_numpy(protop_kernel).float().clamp(min=0))
        self.register_buffer("bias", torch.from_numpy(bias).float())

    def forward(self, spatial: torch.Tensor) -> torch.Tensor:
        b, d, h, w = spatial.shape
        x = spatial.permute(0, 2, 3, 1).reshape(b, h * w, d)
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-5)
        sims = x @ self.unit_kernel
        sims = sims.reshape(b, h * w, self.n_classes, self.n_proto)
        sims = sims.amax(dim=1)
        return (sims * self.pk).sum(-1) + self.bias


class Perch2(nn.Module):
    def __init__(self, mel_matrix, prototypes, protop_kernel, bias, want_embedding=False):
        super().__init__()
        self.frontend = MelFrontend(mel_matrix)
        self.backbone = Backbone()
        self.head = ProtoPNetHead(prototypes, protop_kernel, bias)
        self.want_embedding = want_embedding

    def forward(self, audio: torch.Tensor):
        mel = self.frontend(audio).unsqueeze(1)
        spatial = self.backbone(mel)
        logits = self.head(spatial)
        if self.want_embedding:
            return logits, spatial.mean(dim=(2, 3))
        return logits


def _t(k: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(k.transpose(3, 2, 0, 1).copy()).float()


def load(weights="perch2_weights.npz", mapping="perch2_leaf_mapping.json",
         mel="perch2_mel_matrix.npy", keep_classes=None, want_embedding=False) -> Perch2:
    z = np.load(weights)
    m = {v: z[k] for k, v in json.load(open(mapping)).items()}

    proto, pk, bias = m["head/prototypes"], m["head/protop_kernel"], m["head/bias"]
    if keep_classes is not None:
        idx = np.asarray(keep_classes)
        proto, pk, bias = proto[idx], pk[idx], bias[idx]

    net = Perch2(np.load(mel), proto, pk, bias, want_embedding=want_embedding)
    bb = net.backbone
    sd = {}

    sd["stem_conv.weight"] = _t(m["params/Stem_0/Conv_0/kernel"])
    sd["stem_bn.weight"] = torch.from_numpy(m["params/Stem_0/BatchNorm_0/scale"]).float()
    sd["stem_bn.bias"] = torch.from_numpy(m["params/Stem_0/BatchNorm_0/bias"]).float()
    sd["stem_bn.running_mean"] = torch.from_numpy(m["batch_stats/Stem_0/BatchNorm_0/mean"]).float()
    sd["stem_bn.running_var"] = torch.from_numpy(m["batch_stats/Stem_0/BatchNorm_0/var"]).float()

    sd["head_conv.weight"] = _t(m["params/Head_0/Conv_0/kernel"])
    sd["head_bn.weight"] = torch.from_numpy(m["params/Head_0/BatchNorm_0/scale"]).float()
    sd["head_bn.bias"] = torch.from_numpy(m["params/Head_0/BatchNorm_0/bias"]).float()
    sd["head_bn.running_mean"] = torch.from_numpy(m["batch_stats/Head_0/BatchNorm_0/mean"]).float()
    sd["head_bn.running_var"] = torch.from_numpy(m["batch_stats/Head_0/BatchNorm_0/var"]).float()

    for i, blk in enumerate(bb.blocks):
        p, bs = f"params/MBConv_{i}", f"batch_stats/MBConv_{i}"
        for tag, conv in (("Expand", "ExpandConv"), ("Depthwise", "DepthwiseConv"),
                          ("Project", "ProjectConv")):
            if tag == "Expand" and blk.expand == 1:
                continue
            sd[f"blocks.{i}.{conv}.weight"] = _t(m[f"{p}/{conv}/kernel"])
            bn = f"{tag}BatchNorm"
            sd[f"blocks.{i}.{bn}.weight"] = torch.from_numpy(m[f"{p}/{bn}/scale"]).float()
            sd[f"blocks.{i}.{bn}.bias"] = torch.from_numpy(m[f"{p}/{bn}/bias"]).float()
            sd[f"blocks.{i}.{bn}.running_mean"] = torch.from_numpy(m[f"{bs}/{bn}/mean"]).float()
            sd[f"blocks.{i}.{bn}.running_var"] = torch.from_numpy(m[f"{bs}/{bn}/var"]).float()
        for which in ("Reduce", "Expand"):
            se = f"{p}/SqueezeAndExcitation_0/{which}"
            sd[f"blocks.{i}.SqueezeAndExcitation_0.{which}.weight"] = torch.from_numpy(
                m[f"{se}/kernel"].T.copy()).float()
            sd[f"blocks.{i}.SqueezeAndExcitation_0.{which}.bias"] = torch.from_numpy(
                m[f"{se}/bias"]).float()

    missing, unexpected = bb.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "num_batches_tracked" not in k]
    if missing or unexpected:
        raise RuntimeError(f"weight load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    net.eval()
    return net
