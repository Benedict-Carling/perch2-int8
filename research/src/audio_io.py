import wave
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

SR, WIN = 32_000, 160_000


def read_wav(fh) -> tuple[np.ndarray, int]:
    with wave.open(fh) as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        x = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def to_windows(x: np.ndarray, sr: int) -> np.ndarray:
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    if len(x) < WIN:
        x = np.pad(x, (0, WIN - len(x)))
    return x[: len(x) // WIN * WIN].reshape(-1, WIN).astype(np.float32)


def load_wav_bytes(fh) -> np.ndarray:
    x, sr = read_wav(fh)
    return to_windows(x, sr)


def load_wav(path: Path) -> np.ndarray:
    with open(path, "rb") as fh:
        return load_wav_bytes(fh)
