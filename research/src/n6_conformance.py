from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
REF = ROOT / "perch2_tf_reference.npz"

GATE1_MIN_COSINE = 0.9999
GATE2_MAX_ABS_ERR = 1e-3


def per_clip_cosine(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="../../perch2_backbone_fp32.onnx")
    ap.add_argument("--reference", required=True, help="Local SavedModel reference NPZ; not redistributed")
    ap.add_argument("--golden", default="golden_frontend.npz")
    ap.add_argument("--n-golden", type=int, default=8)
    ap.add_argument("--out", default="n6_conformance.json")
    args = ap.parse_args()

    import onnxruntime as ort
    import torch
    import perch2_torch

    torch.set_num_threads(1)
    d = np.load(ROOT / args.reference, allow_pickle=False)
    audio, spec_ref, emb_ref = d["audio"], d["spectrogram"], d["embedding"]
    res = {}

    fe = perch2_torch.load().frontend
    with torch.no_grad():
        spec = fe(torch.from_numpy(audio)).numpy()
    err = np.abs(spec - spec_ref)
    res["gate2_frontend"] = {
        "max_abs_err": float(err.max()),
        "mean_abs_err": float(err.mean()),
        "threshold": GATE2_MAX_ABS_ERR,
        "pass": bool(err.max() < GATE2_MAX_ABS_ERR),
    }
    print(f"GATE-2 frontend    max|err| {err.max():.3e}  "
          f"(< {GATE2_MAX_ABS_ERR:.0e})  "
          f"{'PASS' if err.max() < GATE2_MAX_ABS_ERR else 'FAIL'}")

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    s = ort.InferenceSession(str(ROOT / args.backbone), so,
                             providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0].name
    got = np.stack([s.run(None, {inp: spec_ref[i][None, None]})[0][0]
                    for i in range(len(spec_ref))])
    cos = per_clip_cosine(emb_ref, got)
    res["gate1_backbone"] = {
        "model": args.backbone, "n": int(len(cos)),
        "min_per_clip_cosine": float(cos.min()),
        "mean_per_clip_cosine": float(cos.mean()),
        "threshold": GATE1_MIN_COSINE,
        "pass": bool(cos.min() >= GATE1_MIN_COSINE),
    }
    print(f"GATE-1 backbone    min per-clip cosine {cos.min():.8f}  "
          f"(>= {GATE1_MIN_COSINE})  "
          f"{'PASS' if cos.min() >= GATE1_MIN_COSINE else 'FAIL'}")

    k = args.n_golden
    med = spec_ref[:k] - np.median(spec_ref[:k], axis=(1, 2), keepdims=True)
    np.savez_compressed(
        ROOT / args.golden,
        audio=audio[:k].astype(np.float32),
        log_mel=spec_ref[:k].astype(np.float32),
        log_mel_median_subtracted=med.astype(np.float32),
        tolerance=np.array(GATE2_MAX_ABS_ERR),
        note=np.array(
            "audio: 32 kHz mono float32, 160000 samples. log_mel: the input the "
            "INT8 model expects. Ignore log_mel_median_subtracted; the model does "
            "not use it. Pass if max|err| < tolerance."),
    )
    print(f"\nwrote {args.golden}: {k} golden pairs "
          f"(audio -> log-mel)")

    res["golden"] = {"file": args.golden, "n": k,
                     "tolerance": GATE2_MAX_ABS_ERR}
    res["all_pass"] = bool(res["gate1_backbone"]["pass"]
                           and res["gate2_frontend"]["pass"])
    (ROOT / args.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {args.out}: all_pass={res['all_pass']}")
    if not res["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
