"""Perch 2.0 INT8 backbone: audio -> raw log-mel -> embedding.

No species classifier is included. Paths default to assets beside this module,
so both a GitHub checkout and a Hugging Face snapshot work offline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 32_000
WINDOW_SAMPLES = 160_000


def log_mel(audio: np.ndarray, mel_matrix: np.ndarray) -> np.ndarray:
    """One 5-second, 32 kHz mono float waveform -> [1,1,500,128].

    Symmetric Hann, sum-normalised magnitude STFT, 640-sample frames,
    320-sample hop, 1024-point FFT, 160 zero samples at each boundary.
    No median subtraction, gain normalisation, or L2 normalisation.
    """
    audio = np.asarray(audio)
    if audio.shape != (WINDOW_SAMPLES,):
        raise ValueError(f"Expected ({WINDOW_SAMPLES},) mono samples; got {audio.shape}")
    if not np.issubdtype(audio.dtype, np.floating):
        raise ValueError("Pass floating-point audio, not unscaled integer PCM")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinity")
    if mel_matrix.shape != (513, 128) or not np.isfinite(mel_matrix).all():
        raise ValueError("Expected a finite [513,128] mel matrix")
    padded = np.pad(audio.astype(np.float32), (160, 160))
    frames = np.lib.stride_tricks.sliding_window_view(padded, 640)[::320]
    window = np.hanning(640).astype(np.float32)
    spectrum = np.fft.rfft(frames * (window / window.sum()), n=1024, axis=-1)
    magnitude = np.abs(spectrum).astype(np.float32)
    mel = magnitude @ mel_matrix
    features = 0.1 * np.log(np.maximum(mel, 1e-5))
    return features.astype(np.float32)[None, None]


def load_audio(path: str | Path, *, time_expand_ultrasound: bool = False) -> np.ndarray:
    """Read, average channels, then resample ordinary audio to 32 kHz.

    Explicit time expansion instead reinterprets a higher-rate recording at
    32 kHz. It changes the duration and frequency axis; it is not resampling.
    """
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Audio must be non-empty and finite")
    audio = audio.mean(axis=1)
    if time_expand_ultrasound:
        if rate <= SAMPLE_RATE:
            raise ValueError("Time expansion requires a recording above 32 kHz")
    elif rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, rate // divisor)
    return audio.astype(np.float32)


class PerchINT8:
    def __init__(self, model: str | Path = ROOT / "perch2_backbone_int8.onnx",
                 mel_matrix: str | Path = ROOT / "perch2_mel_matrix.npy",
                 threads: int = 1):
        if threads < 1:
            raise ValueError("threads must be positive")
        self.mel_matrix = np.load(mel_matrix, allow_pickle=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model), options,
                                            providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 1, 500, 128]:
            raise ValueError("Expected the fixed-shape Perch embedding backbone")
        if "embedding" not in {x.name for x in outputs}:
            raise ValueError("Model has no embedding output")
        self.input_name = inputs[0].name

    def embed_window(self, audio_32k: np.ndarray) -> np.ndarray:
        mel = log_mel(audio_32k, self.mel_matrix)
        embedding = self.session.run(["embedding"], {self.input_name: mel})[0][0]
        if embedding.shape != (1536,) or not np.isfinite(embedding).all():
            raise RuntimeError("Model returned an invalid embedding")
        return embedding

    def embed_audio(self, audio_32k: np.ndarray) -> np.ndarray:
        """Non-overlapping windows; zero-pad the final partial window.

        Returns [number_of_windows,1536], retaining time order. No averaging.
        """
        audio_32k = np.asarray(audio_32k)
        if audio_32k.ndim != 1 or not len(audio_32k):
            raise ValueError("Expected non-empty mono audio at 32 kHz")
        if not np.issubdtype(audio_32k.dtype, np.floating) or not np.isfinite(audio_32k).all():
            raise ValueError("Expected finite floating-point audio")
        embeddings = []
        for start in range(0, len(audio_32k), WINDOW_SAMPLES):
            window = audio_32k[start:start + WINDOW_SAMPLES]
            if len(window) < WINDOW_SAMPLES:
                window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))
            embeddings.append(self.embed_window(window))
        return np.stack(embeddings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/embeddings.npy"))
    parser.add_argument("--model", type=Path, default=ROOT / "perch2_backbone_int8.onnx")
    parser.add_argument("--mel-matrix", type=Path, default=ROOT / "perch2_mel_matrix.npy")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time-expand-ultrasound", action="store_true",
                        help="reinterpret high-rate samples at 32 kHz instead of resampling")
    args = parser.parse_args()
    audio = load_audio(args.audio, time_expand_ultrasound=args.time_expand_ultrasound)
    model = PerchINT8(args.model, args.mel_matrix, args.threads)
    embeddings = model.embed_audio(audio)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as stream:
        np.save(stream, embeddings, allow_pickle=False)
    print(json.dumps({"output": str(args.out), "shape": list(embeddings.shape),
                      "dtype": str(embeddings.dtype), "finite": bool(np.isfinite(embeddings).all()),
                      "window_seconds": 5, "last_window_zero_padded": len(audio) % WINDOW_SAMPLES != 0,
                      "time_expanded": args.time_expand_ultrasound}))


if __name__ == "__main__":
    main()
