from __future__ import annotations

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BEANS = ROOT / "beans"

DATASETS = {
    "watkins": ("watkins.zip", "species", ["Weddell_Seal"],
                lambda root, r: root / r["path"]),
    "dogs": ("dog_barks.zip", "name", [],
             lambda root, r: root / "audio" / r["filename"]),
    "bats": ("egyptian_fruit_bats.zip", "Emitter", [],
             lambda root, r: root / "audio" / r["File Name"]),
    "humbugdb": (None, "species", [],
                 lambda root, r: root / "audio" / f"{r['id']}.wav"),
}


def _humbugdb_labels(df):
    df["species"] = df["species"].fillna("non-mosquito")
    counts = df["species"].value_counts()
    rare = counts[counts <= 100].index
    df["species"] = df["species"].replace(rare, "others")
    return df

PUBLISHED_LP = {"watkins": 0.885, "bats": 0.766, "dogs": 0.964,
                "humbugdb": 0.762, "esc50": 0.895, "cbi": 0.792, "speech": 0.801}


def extract(name: str) -> Path:
    out = BEANS / name
    zf = DATASETS[name][0]
    if zf is None:
        aud = out / "audio"
        if not aud.exists() or not any(aud.glob("*.wav")):
            aud.mkdir(parents=True, exist_ok=True)
            for z in sorted(out.glob("z*.zip")):
                with zipfile.ZipFile(z) as f:
                    f.extractall(aud)
        return out
    if out.exists() and any(out.rglob("*.csv")):
        return out
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BEANS / zf) as z:
        z.extractall(out)
    return out


def splits(name: str):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    _, label_col, drop, resolve = DATASETS[name]
    root = extract(name)
    if name == "humbugdb":
        ann = root / "metadata.csv"
        df = _humbugdb_labels(pd.read_csv(ann))
        base = root
    else:
        cands = list(root.rglob("annotations.csv"))
        if not cands:
            raise SystemExit(f"{name}: no annotations.csv under {root}")
        ann = cands[0]
        df = pd.read_csv(ann)
        base = ann.parent
    df["path"] = [str(resolve(base, r)) for _, r in df.iterrows()]
    df = df.rename(columns={label_col: "label"})
    for d in drop:
        df = df[df["label"] != d]
    exists = df["path"].map(lambda p: Path(p).exists())
    if not exists.all():
        print(f"  {name}: {(~exists).sum()} of {len(df)} annotated files missing "
              f"from the archive, dropped", flush=True)
        df = df[exists]
    tr, vt = train_test_split(df, test_size=0.4, random_state=42, shuffle=True,
                              stratify=df["label"])
    va, te = train_test_split(vt, test_size=0.5, random_state=42, shuffle=True,
                              stratify=vt["label"])
    return tr.sort_index(), va.sort_index(), te.sort_index()


def embed_all(rows, heads, sr_target=32_000, max_windows=6, resample=True):
    import soundfile as sf
    from n6_embed import windows as cut

    E = {k: [] for k in heads}
    Y, t0 = [], time.time()
    for i, (path, label) in enumerate(rows):
        try:
            x, sr = sf.read(path, dtype="float32")
        except Exception:
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)
        w = cut(x.astype(np.float32), sr_target if not resample else sr, max_windows)
        mels = [heads[next(iter(heads))].mel(w[j]) for j in range(len(w))]
        for k, h in heads.items():
            v = np.stack([h.embed_from_mel(m) for m in mels]).mean(axis=0)
            E[k].append(v)
        Y.append(label)
        if len(Y) % 500 == 0:
            print(f"    {len(Y):,}/{len(rows):,}  "
                  f"{len(Y)/(time.time()-t0):.1f}/s", flush=True)
    return {k: np.stack(v).astype(np.float32) for k, v in E.items()}, np.array(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="watkins,dogs")
    ap.add_argument("--models", required=True, help="name=path=frontend;...")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-resample", action="store_true",
                    help="treat samples as already 32 kHz (ultrasonic datasets)")
    ap.add_argument("--out", default="n6_beans.json")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from n6_spatial import SpatialHead

    heads = {}
    for spec in args.models.split(";"):
        nm, path, front = spec.split("=")
        heads[nm] = SpatialHead(path, front, args.threads, class_idx=[0])

    res = {}
    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        res = json.loads(p.read_text())
    for ds in args.datasets.split(","):
        print(f"\n=== {ds} ===", flush=True)
        tr, va, te = splits(ds)
        print(f"  {len(tr)} train / {len(va)} valid / {len(te)} test, "
              f"{tr['label'].nunique()} classes")
        rs = not args.no_resample
        Etr, ytr = embed_all(list(zip(tr["path"], tr["label"])), heads, resample=rs)
        Ete, yte = embed_all(list(zip(te["path"], te["label"])), heads, resample=rs)
        for nm in heads:
            sc = StandardScaler().fit(Etr[nm])
            clf = LogisticRegression(max_iter=3000, n_jobs=-1).fit(
                sc.transform(Etr[nm]), ytr)
            acc = float((clf.predict(sc.transform(Ete[nm])) == yte).mean())
            res.setdefault(nm, {})[ds] = {"accuracy": acc, "n_test": int(len(yte)),
                                          "n_classes": int(len(set(ytr))),
                                          "published_lp": PUBLISHED_LP.get(ds)}
            print(f"  {nm:26s} accuracy {acc:.3f} "
                  f"(published {PUBLISHED_LP.get(ds, float('nan')):.3f})", flush=True)
        p.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
