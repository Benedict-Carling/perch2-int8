"""Explicitly regenerate regression fixtures; review changed hashes afterwards.

Requires research dependencies. Never invoked by tests or normal inference.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/src"))
from perch_int8 import PerchINT8
from perch2_torch import MelFrontend


def main():
    torch.set_num_threads(1)
    t = np.arange(160000, dtype=np.float64) / 32000
    rng = np.random.default_rng(20260928)
    audio = np.stack([
        np.zeros_like(t),
        0.15 * np.sin(2 * np.pi * (400 * t + 900 * t * t)),
        rng.normal(0, 0.02, len(t)),
        np.where(np.arange(len(t)) == 80000, 0.5, 0),
    ]).astype(np.float32)
    frontend = MelFrontend(np.load(ROOT / "perch2_mel_matrix.npy")).eval()
    with torch.no_grad():
        mels = frontend(torch.from_numpy(audio)).numpy()
    model = PerchINT8()
    direct = np.stack([model.session.run(["embedding"], {model.input_name: m[None, None]})[0][0] for m in mels])
    runtime = np.stack([model.embed_window(a) for a in audio])
    np.savez_compressed(ROOT / "synthetic_reference.npz", audio=audio, log_mel=mels,
                        embedding=runtime, embedding_direct_dft=direct,
                        note=np.array("Synthetic silence, chirp, seeded noise, impulse. log_mel: independent PyTorch direct DFT. embedding: released NumPy FFT frontend + ONNX CPU. embedding_direct_dft: exact reference log_mel + ONNX CPU. RAW log-mel; no median subtraction. Regression checks, not accuracy evidence."))
    sf.write(ROOT / "example.wav", audio[1], 32000, subtype="FLOAT")
    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ["synthetic_reference.npz", "example.wav"]:
        content = (ROOT / name).read_bytes()
        manifest["files"][name] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (ROOT / "SHA256SUMS").write_text("".join(f"{value['sha256']}  {name}\n" for name, value in manifest["files"].items()))
    print("Regenerated synthetic fixtures; inspect the diff before committing.")


if __name__ == "__main__":
    main()
