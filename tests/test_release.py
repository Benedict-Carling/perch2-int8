from pathlib import Path
import hashlib
import json
import platform
import sys

import numpy as np
import onnx
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from perch_int8 import PerchINT8, log_mel, load_audio, WINDOW_SAMPLES


@pytest.fixture(scope="module")
def model():
    return PerchINT8()


def test_artifact_integrity():
    for filename, expected in json.loads((ROOT / "manifest.json").read_text())["files"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected["sha256"]
    onnx.checker.check_model(str(ROOT / "perch2_backbone_int8.onnx"))


def test_frontend_and_embedding_regression(model):
    with np.load(ROOT / "synthetic_reference.npz", allow_pickle=False) as reference:
        for audio, expected_mel, expected_embedding, dft_embedding in zip(
                reference["audio"], reference["log_mel"], reference["embedding"],
                reference["embedding_direct_dft"]):
            np.testing.assert_allclose(log_mel(audio, model.mel_matrix)[0, 0],
                                       expected_mel, atol=1e-3, rtol=0)
            got = model.embed_window(audio)
            assert got.shape == (1536,)
            assert np.isfinite(got).all()
            if sys.platform == "darwin" and platform.machine() == "arm64":
                # This golden was produced by ORT on Apple Silicon. It is not
                # portable to Linux x86_64: QDQ CPU kernels differ measurably.
                cosine = np.dot(got, expected_embedding) / (
                    np.linalg.norm(got) * np.linalg.norm(expected_embedding))
                assert cosine > 0.9999
                np.testing.assert_allclose(got, expected_embedding, atol=0.02, rtol=0.01)
                # Check the model independently of FFT rounding at INT8 boundaries.
                direct = model.session.run(
                    ["embedding"], {model.input_name: expected_mel[None, None]})[0][0]
                np.testing.assert_allclose(direct, dft_embedding, atol=1e-4, rtol=1e-4)
            elif sys.platform == "linux" and platform.machine() == "x86_64":
                # Linux CI is an execution gate; its output is architecture- and
                # kernel-specific, so compare repeat runs instead of an ARM golden.
                np.testing.assert_array_equal(got, model.embed_window(audio))


def test_partial_window_is_padded(model):
    audio = load_audio(ROOT / "example.wav")
    got = model.embed_audio(np.concatenate([audio, audio[:123]]))
    assert got.shape == (2, 1536)
    np.testing.assert_allclose(got[1], model.embed_window(np.pad(audio[:123], (0, WINDOW_SAMPLES - 123))))


def test_resampling_and_explicit_time_expansion(tmp_path):
    path = tmp_path / "ultrasound.wav"
    sf.write(path, np.zeros(250_000, dtype=np.float32), 250_000, subtype="FLOAT")
    assert len(load_audio(path)) == 32_000
    assert len(load_audio(path, time_expand_ultrasound=True)) == 250_000


@pytest.mark.parametrize("audio", [np.zeros(7), np.zeros(WINDOW_SAMPLES, dtype=np.int16),
                                   np.full(WINDOW_SAMPLES, np.nan)])
def test_invalid_input_rejected(model, audio):
    with pytest.raises(ValueError):
        model.embed_window(audio)
