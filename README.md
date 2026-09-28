---
license: apache-2.0
library_name: onnx
tags:
  - audio
  - bioacoustics
  - feature-extraction
  - onnx
  - int8
  - stm32n6
  - perch
model-index:
  - name: Perch 2.0 INT8 backbone
    results: []
---

# Perch 2.0 INT8 backbone

**Private review candidate — not yet approved for public release.**

**Open review finding:** the Mac inference checks pass, but the strict embedding
parity check fails on Linux x86_64. Identical saved input features also differ
between hosts; this is not solely an audio-frontend difference. The failing
check is retained. See [CPU portability](docs/CPU_PORTABILITY.md) before relying
on cross-platform numerical agreement or the historical accuracy figures on a
new runtime/target.

A **12.3 MB statically quantised Perch 2.0 embedding backbone** for experiments
with INT8 accelerators. It converts raw log-mel features into 1,536-dimensional
embeddings. This repository also supplies a small NumPy audio frontend and an
ONNX Runtime example, so you can use the embeddings on a computer without
TensorFlow or PyTorch.

This is a community derivative by Benedict Carling / Alauda Audio, not an
official Google release. The original model is
[Google's Perch 2.0](https://www.kaggle.com/models/google/bird-vocalization-classifier/),
described in [The Bittern Lesson for Bioacoustics](https://arxiv.org/abs/2508.04665).

**No species classifier is included.** Train a small classifier on these
embeddings for your own task. This is a research preview, not a complete
STM32 firmware project. STM32N6 results below are compiler output, not board
measurements.

## Quickstart

Python **3.12** is the tested environment. From an authorised checkout of the
[private GitHub repository](https://github.com/Benedict-Carling/perch2-int8):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python perch_int8.py example.wav --out outputs/example.npy
```

The supplied WAV is a synthetic five-second chirp, not a wildlife recording.
The command writes a finite float32 array of shape **`(1, 1536)`**. Run the same
command with your own WAV or FLAC to obtain one embedding per five-second
window. Channels are averaged; ordinary audio is resampled to 32 kHz; the
last incomplete window is zero-padded. There is no automatic gain normalisation.

```python
from perch_int8 import PerchINT8, load_audio

model = PerchINT8()
embeddings = model.embed_audio(load_audio("example.wav"))
print(embeddings.shape)  # (1, 1536)
```

For ultrasonic recordings, ordinary resampling removes frequencies above
16 kHz. The explicit `--time-expand-ultrasound` option instead reinterprets
the original samples at 32 kHz, changing both duration and frequency. This
matches the time-expansion approach used in the historical bats evaluation;
it is not a promise that arbitrary bat/rodent recordings will classify well.

The [private Hugging Face model page](https://huggingface.co/BenedictCarling/perch2-int8)
contains the same quickstart files. The full research code and tests live in
the GitHub repository.

## Model interface and frontend

| Property | Value |
|---|---|
| ONNX input | `mel`: float32 `[1, 1, 500, 128]` |
| ONNX output | `embedding`: float32 `[1, 1536]` |
| Waveform frontend | 32 kHz mono float audio; 160,000 samples per window |
| STFT | 640-sample symmetric Hann, sum normalised; hop 320; FFT 1024 |
| Padding | 160 zero samples at each waveform boundary |
| Spectral quantity | Magnitude, not power |
| Mel projection | Supplied `[513,128]` filterbank, 60–16,000 Hz |
| Log transform | `0.1 * log(max(mel, 1e-5))` |
| Normalisation | **No median subtraction** and no output L2 normalisation |
| Quantisation | Static QDQ, signed asymmetric INT8 activations, per-channel INT8 weights |
| Calibration | Historical 192-window set; 99.99th-percentile ranges |

Float32 inputs/outputs and `Conv` node names are expected in a QDQ graph.
Execution placement and integer kernels depend on the runtime/compiler.
The ONNX file alone is not a claim that every target executes every operator
in INT8.

## Recorded evaluation

These are **historical experiment results**, packaged with their JSON reports;
the full datasets were not rerun during release preparation. The FP32 control
and INT8 arm use the same evaluation head and protocol. See
[evaluation details](docs/EVALUATION.md) and [reproduction status](research/README.md).

BirdSet: macro-average across seven soundscapes, 317,732 five-second segments.
The stock Perch FP32 ProtoPNet head is applied to spatial backbone features.
POW is excluded because it was used for Perch model selection.

| Backbone | ROC-AUC | cmAP | Top-1 |
|---|---:|---:|---:|
| Local FP32 control | 0.906 | 0.431 | 0.665 |
| INT8 | 0.894 | 0.408 | 0.635 |
| Change | −0.012 | −0.023 | −0.030 |

Four BEANS linear-probe tasks:

| Backbone | Watkins | Dogs | Bats | HumBugDB | Mean |
|---|---:|---:|---:|---:|---:|
| Local FP32 control | 0.900 | 0.978 | 0.781 | 0.810 | 0.867 |
| INT8 | 0.885 | 0.950 | 0.789 | 0.811 | 0.859 |

These results cover **4 of 12 BEANS tasks**. They do not establish improvement
over FP32, statistical equivalence, or general accuracy on all taxa. The
released Perch checkpoint's correspondence to the paper's variants remains
unresolved; the paired local FP32 control is the relevant comparison.

## STM32N6 compiler results

ST Edge AI Core **2.2.0-20266**, Neural-ART **1.1.1-14**, N6570-DK memory layout:

| Measurement from compiler | Result |
|---|---:|
| Accelerator epochs | 266 of 268 (99.25%) |
| Weights in external flash | 11.17 MiB |
| Total activations | 7.01 MiB |
| Activations in external PSRAM | 4.31 MiB |
| On-chip allocation | 2.70 MiB of 2.75 MiB |
| Cycle-derived estimate at 1 GHz | 156.9 ms per five-second input |

**External PSRAM is required. No hardware-in-the-loop validation has been done.**
The cycle figure is not measured latency and does not establish power, battery
life or whole-device throughput. Epoch coverage is not a percentage of runtime.
An MCU audio frontend, classifier, I/O and firmware integration remain separate
engineering work. See [compiler report](results/n6_compile_n6_divsmall_raw_asym.json).

## How this differs from other ONNX releases

[Justin Chu's ONNX port](https://huggingface.co/justinchuby/Perch-onnx) includes
a floating-point DFT-to-MatMul rewrite; that is a frontend implementation change.
[The community ARM INT8 model](https://huggingface.co/tphakala/Perch-v2) uses
dynamic MatMul quantisation while retaining FP32 convolutions. This release
instead targets static quantisation of the convolutional embedding backbone.

The **42.8 MB → 12.3 MB** comparison here is backbone versus backbone. The
community 131 MB artifact includes its frontend and large species classifier.
No matched accuracy, speed or energy comparison against that artifact has been
performed. The approaches serve different deployment needs.

## Verification and provenance

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Checks cover model hashes/ONNX validity, synthetic frontend and embedding
regression, window handling and resampling. Synthetic expected features come
from the independent PyTorch direct-DFT frontend used in the research code.
During release preparation the NumPy frontend was also checked against all
64 historical SavedModel reference clips: maximum absolute log-mel error
**4.66 × 10⁻⁵**, below the 10⁻³ tolerance. The recordings are not redistributed.

The supplied INT8 graph is byte-identical to the historical benchmark graph
after adding explicit zero-valued Pad inputs for compiler compatibility.
[manifest.json](manifest.json) and [validation report](results/release_validation.json)
record the hashes. Tests are packaging checks, not a repeat of benchmark accuracy.

## Licensing, attribution and review

Code, synthetic examples and derivative model assets are provided under
Apache 2.0, retaining the upstream Perch attribution; see [LICENSE](LICENSE)
and [NOTICE](NOTICE). Benchmark datasets keep their own licences and are not
included. No endorsement by Google or STMicroelectronics is implied.

See [REVIEW.md](REVIEW.md) for the remaining review items. Keep GitHub and
Hugging Face private until the owner explicitly approves public release.
