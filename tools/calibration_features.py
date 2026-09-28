"""Create raw calibration log-mel features from an explicit list of audio files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perch_int8 import load_audio, log_mel, WINDOW_SAMPLES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite {args.out}")
    matrix = np.load(ROOT / "perch2_mel_matrix.npy", allow_pickle=False)
    features, records = [], []
    for path in args.audio:
        wave = load_audio(path)
        # Exactly one window per file; no claim this recreates the historical set.
        window = wave[:WINDOW_SAMPLES]
        window = np.pad(window, (0, max(0, WINDOW_SAMPLES - len(window))))
        features.append(log_mel(window, matrix)[0, 0])
        records.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "window": "first five seconds after resampling; zero-pad if short"})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        np.save(f, np.stack(features), allow_pickle=False)
    args.out.with_suffix(".json").write_text(json.dumps({"frontend": "raw", "records": records}, indent=2) + "\n")
    print(f"Wrote {len(features)} calibration windows to {args.out}")


if __name__ == "__main__":
    main()
