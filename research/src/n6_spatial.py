from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx

ROOT = Path(__file__).resolve().parent

SPATIAL_FP32 = "/backbone/Mul_1_output_0"
SPATIAL_QDQ = "/backbone/Mul_1_output_0_DequantizeLinear_Output"


def expose_spatial(src: str, dst: str) -> str:
    m = onnx.load(str(ROOT / src))
    produced = {o for n in m.graph.node for o in n.output}
    name = SPATIAL_QDQ if SPATIAL_QDQ in produced else SPATIAL_FP32
    if name not in produced:
        raise SystemExit(f"{src}: no spatial tensor found "
                         f"({SPATIAL_FP32} / {SPATIAL_QDQ})")
    if name not in {o.name for o in m.graph.output}:
        vi = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None)
        m.graph.output.append(vi)
    onnx.save(m, str(ROOT / dst))
    return name


class SpatialHead:

    def __init__(self, backbone: str, frontend: str = "raw", threads: int = 8,
                 class_idx: list[int] | None = None):
        import onnxruntime as ort
        import torch

        import perch2_torch
        torch.set_num_threads(1)
        self.torch = torch
        net = perch2_torch.load()
        self.fe, self.head = net.frontend, net.head.eval()
        self.frontend = frontend

        self.cols = None
        if class_idx is not None:
            h = self.head
            P = h.n_proto
            keep = torch.tensor([c * P + p for c in class_idx for p in range(P)])
            h.unit_kernel = h.unit_kernel[:, keep].contiguous()
            h.pk = h.pk[class_idx].contiguous()
            h.bias = h.bias[class_idx].contiguous()
            h.n_classes = len(class_idx)
            self.cols = class_idx

        stem = Path(backbone).stem
        exposed = f"_spatial_{stem}.onnx"
        if not (ROOT / exposed).exists():
            self.tensor = expose_spatial(backbone, exposed)
        else:
            m = onnx.load(str(ROOT / exposed))
            outs = {o.name for o in m.graph.output}
            self.tensor = SPATIAL_QDQ if SPATIAL_QDQ in outs else SPATIAL_FP32
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(ROOT / exposed), so,
                                         providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.out_idx = [o.name for o in self.sess.get_outputs()].index(self.tensor)

    def mel(self, audio: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.fe(self.torch.from_numpy(audio[None])).numpy().astype(np.float32)

    def logits_from_mel(self, mel: np.ndarray) -> np.ndarray:
        from n6_embed import MEDIAN_RECENTRE
        med = mel - np.median(mel, axis=(1, 2), keepdims=True)
        src = {"mediansub": med,
               "mediansub_shift": med + MEDIAN_RECENTRE}.get(self.frontend, mel)
        spatial = self.sess.run(None, {self.inp: src[0][None, None]})[self.out_idx]
        with self.torch.no_grad():
            return self.head(self.torch.from_numpy(spatial)).numpy()[0]

    def spatial_from_mel(self, mel: np.ndarray) -> np.ndarray:
        from n6_embed import MEDIAN_RECENTRE
        med = mel - np.median(mel, axis=(1, 2), keepdims=True)
        src = {"mediansub": med,
               "mediansub_shift": med + MEDIAN_RECENTRE}.get(self.frontend, mel)
        return self.sess.run(None, {self.inp: src[0][None, None]})[self.out_idx]

    def embed_from_mel(self, mel: np.ndarray) -> np.ndarray:
        return self.spatial_from_mel(mel).mean(axis=(2, 3))[0]

    def logits(self, audio: np.ndarray) -> np.ndarray:
        return self.logits_from_mel(self.mel(audio))


def _selftest():
    import onnxruntime as ort
    d = np.load(ROOT / "perch2_tf_reference.npz")
    audio, ref = d["audio"], d["label"]
    sh = SpatialHead("perch2_backbone_n6_prep.onnx", "raw", threads=6)
    got = np.stack([sh.logits(audio[i]) for i in range(8)])
    r = ref[:8].astype(np.float64)
    g = got.astype(np.float64)
    cos = (r * g).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(g, axis=1))
    print(f"backbone+head vs OFFICIAL Perch 2.0 outputs:")
    print(f"  mean cosine {cos.mean():.8f}  min {cos.min():.8f}")
    print(f"  max abs err {np.abs(r - g).max():.3e}")
    print(f"  top-1 agreement {np.mean(r.argmax(1) == g.argmax(1))*100:.1f}%")
    ok = cos.min() > 0.9999
    print("[PASS] the split path is the full model" if ok else "[FAIL] split path diverges")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _selftest()
