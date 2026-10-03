---
license: apache-2.0
library_name: onnx
pipeline_tag: feature-extraction
tags:
  - audio
  - bioacoustics
  - feature-extraction
  - onnx
  - int8
  - stm32n6
  - perch
---

# Perch 2.0 INT8 backbone

Google's [Perch 2.0](https://arxiv.org/abs/2508.04665) embedding model,
statically quantised to INT8 so it can run on small NPUs like the STM32N6.
The backbone drops from 42.8 MB to 12.3 MB and loses about 1 point of BirdSet
ROC-AUC.

It turns five seconds of audio into a 1,536-number embedding. There is no
species classifier included, so you train a small one on top for your task.

This is a personal research project by Benedict Carling, not an official
Google release.

## Quickstart

```bash
git clone https://github.com/Benedict-Carling/perch2-int8 && cd perch2-int8
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python perch_int8.py example.wav --out outputs/example.npy
```

You get one embedding per five-second window. From Python:

```python
from perch_int8 import PerchINT8, load_audio

model = PerchINT8()
embeddings = model.embed_audio(load_audio("example.wav"))  # (1, 1536)
```

The audio frontend is plain NumPy, so you don't need TensorFlow or PyTorch.
For bat recordings, `--time-expand-ultrasound` plays the samples back at
32 kHz instead of resampling.

## Train a classifier

Embed a few labelled clips per class, then fit a simple model
(`pip install scikit-learn`):

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from perch_int8 import PerchINT8, load_audio

model = PerchINT8()
clips = {"robin": ["robin1.wav", "robin2.wav"], "wren": ["wren1.wav", "wren2.wav"]}

X, y = [], []
for label, paths in clips.items():
    for path in paths:
        for emb in model.embed_audio(load_audio(path)):
            X.append(emb)
            y.append(label)

clf = LogisticRegression(max_iter=1000).fit(np.array(X), y)
print(clf.predict(model.embed_audio(load_audio("mystery.wav"))))
```

When you test it, hold out whole recordings or sites, not single windows.

## How it does

Compared against the same backbone in FP32, with the same head and data:

| | BirdSet ROC-AUC | BirdSet cmAP | BEANS (4 tasks) |
|---|---:|---:|---:|
| FP32 | 0.906 | 0.431 | 0.867 |
| INT8 | 0.894 | 0.408 | 0.859 |

On the STM32N6, ST's compiler puts 266 of 268 epochs on the NPU and estimates
about 157 ms per window at 1 GHz. It needs external PSRAM. I haven't measured
it on a real board yet.

More detail is in [docs/EVALUATION.md](docs/EVALUATION.md) and
[research/README.md](research/README.md).

## Known issues

- Embeddings from Mac (arm64) and Linux (x86_64) differ slightly for the same
  input. Both work, but don't expect identical numbers across machines. See
  [docs/CPU_PORTABILITY.md](docs/CPU_PORTABILITY.md).
- BEANS covers 4 of the 12 tasks.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Licence

Apache 2.0, with the upstream Perch attribution kept in [NOTICE](NOTICE).
