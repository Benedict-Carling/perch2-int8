from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="n6_splits_inat.csv")
    ap.add_argument("--arm", default="calib_small")
    ap.add_argument("--gain-jitter", type=float, default=0.0,
                    help="uniform +/- this many dB applied before the frontend")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--windows", type=int, default=1,
                    help="windows per clip. 1 for calibration, where clip "
                         "diversity is what matters.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import perch2_torch
    from n6_embed import Reader, windows

    rng = np.random.default_rng(args.seed)
    rows = [r for r in csv.DictReader((ROOT / args.splits).open())
            if r["arm"] == args.arm]
    fe = perch2_torch.load().frontend
    rd, out, t0 = Reader(), [], time.time()
    try:
        for r in rows:
            got = rd.read_row(r)
            if not got:
                continue
            x, sr = got
            if args.gain_jitter:
                db = rng.uniform(-args.gain_jitter, args.gain_jitter)
                x = x * (10.0 ** (db / 20.0))
            w = windows(x, sr, args.windows)
            with torch.no_grad():
                m = fe(torch.from_numpy(w)).numpy().astype(np.float32)
            out.extend(m)
    finally:
        rd.close()

    mel = np.stack(out)
    np.save(ROOT / args.out, mel)
    print(f"{args.out}: {mel.shape} from {len(rows)} clips "
          f"(jitter +/-{args.gain_jitter} dB) in {time.time()-t0:.0f}s")
    med = mel - np.median(mel, axis=(1, 2), keepdims=True)
    print(f"  raw range      [{mel.min():.3f}, {mel.max():.3f}]")
    print(f"  median-sub     [{med.min():.3f}, {med.max():.3f}]")
    print(f"  floor fraction {np.mean(mel <= -1.1512):.4f}")


if __name__ == "__main__":
    main()
