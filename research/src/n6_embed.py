from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SR, WIN = 32_000, 160_000

MEDIAN_RECENTRE = -0.9111

MODELS = {
    "fp32":            ("perch2_backbone_n6_prep.onnx", "raw"),
    "fp32_norm":       ("perch2_backbone_n6_prep.onnx", "mediansub"),
    "int8_incumbent":  ("n6_allops_norm_sym.onnx",      "mediansub"),
}


class Reader:

    def __init__(self):
        self._tars = {}

    def read_row(self, row):
        from audio_io import read_wav
        loc = row["location"]
        try:
            if loc.startswith("shard:"):
                name = loc.split(":", 1)[1]
                if name not in self._tars:
                    self._tars[name] = tarfile.open(ROOT / "shards" / name)
                m = self._tars[name].extractfile(f"{row['key']}.wav")
                if m is None:
                    return None
                return read_wav(io.BytesIO(m.read()))
            with open(ROOT / loc.split(":", 1)[1], "rb") as fh:
                return read_wav(fh)
        except Exception:
            return None

    def close(self):
        for t in self._tars.values():
            t.close()


def windows(x: np.ndarray, sr: int, k: int) -> np.ndarray:
    from math import gcd
    from scipy.signal import resample_poly
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    n = len(x) // WIN
    if n <= 1:
        return x[:WIN][None].astype(np.float32)
    idx = np.unique(np.linspace(0, n - 1, min(k, n)).round().astype(int))
    return np.stack([x[i * WIN:(i + 1) * WIN] for i in idx]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="n6_splits_inat.csv")
    ap.add_argument("--arm", default="eval")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--models", default="__all__",
                    help="comma-separated subset of MODELS, or '' for none")
    ap.add_argument("--extra", default="", help="name=path=transform, repeatable with ;")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="n6_emb_eval.npz")
    args = ap.parse_args()

    import onnxruntime as ort
    import torch
    import perch2_torch

    torch.set_num_threads(1)

    models = dict(MODELS)
    if args.models != "__all__":
        want = set(filter(None, args.models.split(",")))
        models = {k: v for k, v in models.items() if k in want}
    for spec in filter(None, args.extra.split(";")):
        name, path, tf = spec.split("=")
        models[name] = (path, tf)
    missing = [p for p, _ in models.values() if not (ROOT / p).exists()]
    if missing:
        raise SystemExit(f"missing model files: {missing}")

    rows = [r for r in csv.DictReader((ROOT / args.splits).open())
            if r["arm"] == args.arm]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{args.arm}: {len(rows):,} clips, {len(models)} models "
          f"({', '.join(models)})", flush=True)

    fe = perch2_torch.load().frontend
    sess = {}
    for name, (path, tf) in models.items():
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        s = ort.InferenceSession(str(ROOT / path), so,
                                 providers=["CPUExecutionProvider"])
        sess[name] = (s, s.get_inputs()[0].name, tf)

    rd = Reader()
    emb = {k: [] for k in models}
    meta, t0, skipped = [], time.time(), 0
    try:
        for i, r in enumerate(rows):
            got = rd.read_row(r)
            if not got:
                skipped += 1
                continue
            w = windows(got[0], got[1], args.windows)
            with torch.no_grad():
                mel = fe(torch.from_numpy(w)).numpy().astype(np.float32)
            med = mel - np.median(mel, axis=(1, 2), keepdims=True)
            for name, (s, inp, tf) in sess.items():
                src = {"mediansub": med,
                       "mediansub_shift": med + MEDIAN_RECENTRE}.get(tf, mel)
                v = np.stack([s.run(None, {inp: src[j][None, None]})[0][0]
                              for j in range(len(src))])
                emb[name].append(v.mean(axis=0))
            meta.append(r)
            if len(meta) % 500 == 0:
                el = time.time() - t0
                rate = len(meta) / el
                print(f"  {len(meta):,}/{len(rows):,}  {rate:.1f} clip/s  "
                      f"eta {(len(rows)-len(meta))/rate/60:.0f} min", flush=True)
    finally:
        rd.close()

    out = {f"emb_{k}": np.stack(v).astype(np.float32) for k, v in emb.items()
           if v}
    for col in ["key", "scientific_name", "group", "source", "licence_tier",
                "has_background"]:
        out[col] = np.array([m[col] for m in meta])
    out["models"] = np.array(json.dumps({k: v for k, v in models.items()}))
    np.savez_compressed(ROOT / args.out, **out)
    print(f"\nwrote {args.out}: {len(meta):,} clips, {skipped} unreadable, "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
