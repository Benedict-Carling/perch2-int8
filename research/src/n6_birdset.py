from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BIRDSET = ROOT / "birdset"

PAPER_DATASETS = ["PER", "NES", "UHH", "HSN", "NBP", "SSW", "SNE"]

PUBLISHED = {
    "roc_auc": {"PER": 0.786, "NES": 0.953, "UHH": 0.912, "HSN": 0.915,
                "NBP": 0.933, "SSW": 0.973, "SNE": 0.883, "Mean": 0.908},
    "cmap": {"PER": 0.232, "NES": 0.403, "UHH": 0.380, "HSN": 0.533,
             "NBP": 0.661, "SSW": 0.469, "SNE": 0.341, "Mean": 0.431},
    "top1": {"PER": 0.535, "NES": 0.562, "UHH": 0.595, "HSN": 0.659,
             "NBP": 0.724, "SSW": 0.789, "SNE": 0.791, "Mean": 0.665},
}


def species_list(ds: str) -> list[str]:
    """Return labels present in the pinned test metadata.

    The BirdSet snapshot pinned by this project does not contain the historical
    ``classes.py`` this evaluation used to fetch from the moving dataset head.
    The metadata is already required for evaluation and gives the exact labels
    used by these test examples.
    """
    labels, _ = load_split(ds)
    return sorted({str(label) for row in labels.values() for label in row})


def perch_classes() -> list[str]:
    import csv
    rows = list(csv.DictReader((ROOT / "perch_perch_v2_ebird_classes.csv").open()))
    return [r["ebird2021"] for r in rows]


def load_split(ds: str):
    import pandas as pd
    meta = pd.read_parquet(BIRDSET / ds / f"{ds}_metadata_test_5s.parquet")
    labels = {str(k): list(v) for k, v in meta["ebird_code_multilabel"].items()}
    shards = sorted((BIRDSET / ds).glob(f"{ds}_test5s_shard_*.tar.gz"))
    if not shards:
        raise SystemExit(f"no test_5s shards for {ds} under {BIRDSET/ds}")
    return labels, shards


def iter_audio(shards, labels, limit=0):
    import soundfile as sf
    n = 0
    for sp in shards:
        with tarfile.open(sp, "r:gz") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith(".ogg"):
                    continue
                key = Path(m.name).name
                if key not in labels:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                try:
                    x, sr = sf.read(io.BytesIO(f.read()), dtype="float32")
                except Exception:
                    continue
                if x.ndim > 1:
                    x = x.mean(axis=1)
                yield key, x.astype(np.float32), sr, labels[key]
                n += 1
                if limit and n >= limit:
                    return


def fit_5s(x: np.ndarray, sr: int, want=160_000) -> np.ndarray:
    if sr != 32_000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, 32_000)
        x = resample_poly(x, 32_000 // g, sr // g).astype(np.float32)
    if len(x) < want:
        x = np.pad(x, (0, want - len(x)))
    return x[:want].astype(np.float32)


def metrics(y_true: np.ndarray, scores: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    present = y_true.sum(axis=0) > 0
    aps, aucs = [], []
    for c in np.flatnonzero(present):
        aps.append(average_precision_score(y_true[:, c], scores[:, c]))
        aucs.append(roc_auc_score(y_true[:, c], scores[:, c]))

    labelled = y_true.sum(axis=1) > 0
    top1 = float(np.mean([y_true[i, scores[i].argmax()] > 0
                          for i in np.flatnonzero(labelled)]))
    return {"roc_auc": float(np.mean(aucs)), "cmap": float(np.mean(aps)),
            "top1": top1, "n_classes_scored": int(present.sum()),
            "n_segments": int(len(y_true)),
            "n_segments_labelled": int(labelled.sum())}


def run_pre(ds: str, model: str, limit: int, threads: int,
            backbone: str = "", frontend: str = "raw") -> dict:
    import onnxruntime as ort
    classes = species_list(ds)
    pidx = {c: i for i, c in enumerate(perch_classes())}
    missing = [c for c in classes if c not in pidx]
    cols = [pidx[c] for c in classes if c in pidx]
    kept = [c for c in classes if c in pidx]
    print(f"  {ds}: {len(classes)} classes, {len(kept)} present in Perch's "
          f"label space{', missing ' + ','.join(missing) if missing else ''}")

    labels, shards = load_split(ds)
    heads, sess, inp = {}, None, None
    if backbone:
        from n6_spatial import SpatialHead
        for spec in backbone.split(";"):
            parts = spec.split("=")
            tag, path, front = (parts if len(parts) == 3
                                else (Path(spec).stem, spec, frontend))
            heads[tag] = SpatialHead(path, front, threads, class_idx=cols)
            print(f"  backbone {tag}: {path} (frontend {front}) + Perch 2.0 head")
        cols = list(range(len(kept)))
    else:
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        sess = ort.InferenceSession(str(ROOT / model), so,
                                    providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
    idx = {c: i for i, c in enumerate(kept)}

    Y, t0 = [], time.time()
    S = {k: [] for k in heads} if heads else {"_": []}
    # The FP32 and INT8 graphs share the same mel features. Running their
    # independent ONNX sessions concurrently uses available CPU cores and
    # avoids making each paired example wait for both serial inference calls.
    pool = ThreadPoolExecutor(max_workers=len(heads)) if heads else None
    for name, x, sr, labs in iter_audio(shards, labels, limit):
        w = fit_5s(x, sr)[None]
        if heads:
            mel = next(iter(heads.values())).mel(w[0])
            futures = {tag: pool.submit(h.logits_from_mel, mel)
                       for tag, h in heads.items()}
            for tag, result in futures.items():
                S[tag].append(result.result()[cols])
        else:
            S["_"].append(sess.run(None, {inp: w})[0][0][cols])
        y = np.zeros(len(kept), np.int8)
        for l in labs:
            if l in idx:
                y[idx[l]] = 1
        Y.append(y)
        if len(Y) % 2000 == 0:
            r = len(Y) / (time.time() - t0)
            print(f"    {len(Y):,} segments  {r:.1f}/s", flush=True)
    if pool:
        pool.shutdown()
    Y = np.stack(Y)
    return {k: metrics(Y, np.stack(v)) for k, v in S.items()}


def run_emb(ds: str, models: dict, limit: int, threads: int, out: str):
    import onnxruntime as ort
    import torch

    import perch2_torch
    torch.set_num_threads(1)
    from n6_embed import MEDIAN_RECENTRE
    fe = perch2_torch.load().frontend

    classes = species_list(ds)
    idx = {c: i for i, c in enumerate(classes)}
    labels, shards = load_split(ds)
    sess = {}
    for nm, (path, front) in models.items():
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        sess[nm] = (ort.InferenceSession(str(ROOT / path), so,
                                         providers=["CPUExecutionProvider"]),
                    front)
    E = {k: [] for k in models}
    Y, t0 = [], time.time()
    for name, x, sr, labs in iter_audio(shards, labels, limit):
        w = fit_5s(x, sr)[None]
        with torch.no_grad():
            mel = fe(torch.from_numpy(w)).numpy().astype(np.float32)
        med = mel - np.median(mel, axis=(1, 2), keepdims=True)
        for nm, (s, front) in sess.items():
            src = {"mediansub": med,
                   "mediansub_shift": med + MEDIAN_RECENTRE}.get(front, mel)
            n_in = s.get_inputs()[0].name
            E[nm].append(s.run(None, {n_in: src[0][None, None]})[0][0])
        y = np.zeros(len(classes), np.int8)
        for l in labs:
            if l in idx:
                y[idx[l]] = 1
        Y.append(y)
        if len(Y) % 1000 == 0:
            r = len(Y) / (time.time() - t0)
            print(f"    {len(Y):,} segments  {r:.1f}/s", flush=True)
    d = {f"emb_{k}": np.stack(v).astype(np.float32) for k, v in E.items()}
    d["y"] = np.stack(Y)
    d["classes"] = np.array(classes)
    np.savez_compressed(ROOT / out, **d)
    print(f"  wrote {out}: {len(Y):,} segments, {len(models)} spaces")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="HSN")
    ap.add_argument("--protocol", default="pre", choices=["pre", "emb"])
    ap.add_argument("--model", default="perch2_rebuilt_full.onnx")
    ap.add_argument("--models", default="", help="emb protocol: name=path=frontend;...")
    ap.add_argument("--backbone", default="",
                    help="pre protocol: score this backbone with Perch's own head "
                         "instead of the monolithic fp32 model")
    ap.add_argument("--frontend", default="raw",
                    choices=["raw", "mediansub", "mediansub_shift"])
    ap.add_argument("--tag", default="", help="key to store results under")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="n6_birdset.json")
    args = ap.parse_args()
    if args.protocol == "pre" and args.models:
        ap.error("--models is only for --protocol emb; use --backbone and --tag for pretrained-head scoring")

    todo = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.protocol == "emb":
        models = {}
        for spec in args.models.split(";"):
            nm, path, front = spec.split("=")
            models[nm] = (path, front)
        for ds in todo:
            print(f"\n=== {ds} (embeddings) ===", flush=True)
            run_emb(ds, models, args.limit, args.threads, f"n6_birdset_emb_{ds}.npz")
        return

    res = {}
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        res = json.loads(p.read_text())
    for ds in todo:
        print(f"\n=== {ds} (pre-trained head) ===", flush=True)
        rr = run_pre(ds, args.model, args.limit, args.threads,
                     args.backbone, args.frontend)
        for tag, r in rr.items():
            for k in ("roc_auc", "cmap", "top1"):
                r[f"published_{k}"] = PUBLISHED[k].get(ds)
            key = (args.tag or args.model) if tag == "_" else tag
            res.setdefault(key, {})[ds] = r
            print(f"  {key:26s} ROC-AUC {r['roc_auc']:.3f} (pub {PUBLISHED['roc_auc'][ds]:.3f})"
                  f"  cmAP {r['cmap']:.3f} (pub {PUBLISHED['cmap'][ds]:.3f})"
                  f"  top-1 {r['top1']:.3f} (pub {PUBLISHED['top1'][ds]:.3f})", flush=True)
        p.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
