"""Download, verify, evaluate and optionally remove pinned Perch 2 BirdSet data.

Runs one BirdSet soundscape at a time so the complete evaluation fits on a
machine with less free disk than the sum of the seven public test subsets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/birdset_provenance.json"
DATA_DIR = ROOT / "research/src/birdset"
OUT = ROOT / "outputs/birdset_rerun.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default="PER,NES,UHH,HSN,NBP,SSW,SNE")
    ap.add_argument("--keep-data", action="store_true",
                    help="keep downloaded data under research/src/birdset")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug only: evaluate at most this many segments per subset")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out

    provenance = json.loads(MANIFEST.read_text())
    revision = provenance["commit"]
    requested = [x.strip().upper() for x in args.datasets.split(",") if x.strip()]
    for ds in requested:
        entries = {name: detail for name, detail in provenance["files"].items()
                   if name.startswith(f"{ds}/") and not name.endswith("_old.parquet")}
        if not entries or not any("metadata_test_5s.parquet" in n for n in entries):
            raise SystemExit(f"No pinned test files listed for {ds}")
        print(f"\n=== {ds}: fetch from {revision} and verify SHA-256 ===", flush=True)
        local_files: list[Path] = []
        for relative, detail in entries.items():
            local = DATA_DIR / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists() or local.stat().st_size != detail["size"]:
                fetched = Path(hf_hub_download(
                    repo_id=provenance["repo"], filename=relative,
                    repo_type="dataset", revision=revision,
                    local_dir=DATA_DIR))
                local = fetched
            got = sha256(local)
            if got != detail["sha"]:
                raise SystemExit(f"SHA-256 mismatch for {relative}: {got}")
            local_files.append(local)
            print(f"  verified {relative}", flush=True)

        cmd = [sys.executable, str(ROOT / "research/src/n6_birdset.py"),
               "--datasets", ds, "--protocol", "pre", "--threads", str(args.threads),
               "--out", str(out), "--backbone",
               f"fp32={ROOT / 'perch2_backbone_fp32.onnx'}=raw;"
               f"int8={ROOT / 'perch2_backbone_int8.onnx'}=raw"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        subprocess.run(cmd, cwd=ROOT / "research/src", check=True)
        if not args.keep_data:
            for local in local_files:
                local.unlink(missing_ok=True)
            for parent in sorted({p.parent for p in local_files}, key=lambda p: len(p.parts), reverse=True):
                try:
                    parent.rmdir()
                except OSError:
                    pass
    print(f"\nCompleted. Results: {out}")


if __name__ == "__main__":
    main()
