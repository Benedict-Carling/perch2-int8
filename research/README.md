# Research recipe and reproduction status

The runtime quickstart is self-contained. This directory holds the historical
research scripts; **full end-to-end reproduction is still under review**.
The original 192-window calibration corpus and third-party benchmark audio are
not bundled. Checkpoint bytes are pinned to the verified mirror revision in
`manifest.json`; correspondence to the paper variants remains unresolved.

## Quantise a new calibration set

The exact prepared FP32 backbone is supplied as `perch2_backbone_fp32.onnx`.
Install the research requirements from the repository root:

```bash
python -m pip install -r research/requirements.txt
python tools/calibration_features.py /path/to/clip1.wav /path/to/clip2.wav --out outputs/calib.npy
python research/src/n6_quantise.py --source perch2_backbone_fp32.onnx \
  --calib outputs/calib.npy --raw --asymmetric --ops all \
  --method percentile --percentile 99.99 --out outputs/requantised.onnx
python research/src/padfix_n6.py outputs/requantised.onnx outputs/requantised_padfix.onnx
```

Use representative, diverse recordings, not the synthetic example, for a useful
calibration. The command takes **one window per file**. This demonstrates the
recipe with your data; it does not reproduce the release weights bit-for-bit.
The historical calibration tensor's hash and shape are in the manifest. Its
audio/feature redistribution and download manifest require further review.

## Historical evaluation prerequisites

The BirdSet comparison uses the original FP32 species head on spatial features,
not a new classifier fitted to pooled embeddings. The large original head is
not included in this repository. Obtain the Perch 2.0 SavedModel and its class
CSV from [Google's release](https://www.kaggle.com/models/google/bird-vocalization-classifier/).
The verified download source is the
[`cgeorgiaw/Perch` mirror at revision c3c3c816976a6c1bb75140b8fbaef7397a6f708f](https://huggingface.co/cgeorgiaw/Perch/tree/c3c3c816976a6c1bb75140b8fbaef7397a6f708f).
`tools/prepare_research.py` checks checkpoint bytes and stops on mismatch.

```bash
hf download cgeorgiaw/Perch --revision c3c3c816976a6c1bb75140b8fbaef7397a6f708f \
  --include 'variables/*' 'assets/*' 'saved_model.pb' --local-dir outputs/perch2_savedmodel
```

After installing a compatible TensorFlow separately:

```bash
python tools/prepare_research.py --savedmodel outputs/perch2_savedmodel \
  --classes outputs/perch2_savedmodel/assets/perch_v2_ebird_classes.csv
```

Download BirdSet `test_5s` metadata/shards to `research/src/birdset/<dataset>/`.
The recorded data revision is `806ed2cda4ddcbe6efa194ccafff930aa0e557ce` in
[`DBD-research-group/BirdSet`](https://huggingface.co/datasets/DBD-research-group/BirdSet/tree/data).
Per-file checksums are in `results/birdset_provenance.json`.
BEANS archives belong in `research/src/beans/`; the script documents expected
archive names and annotation layouts. HumBugDB additionally needs metadata.
Dependencies and downloads for these full evaluations have not yet been
exercised in a fresh environment.

The corrected BirdSet commands are separate pretrained-head runs:

```bash
cd research/src
python n6_birdset.py --datasets PER,NES,UHH,HSN,NBP,SSW,SNE --protocol pre \
  --backbone ../../perch2_backbone_fp32.onnx --frontend raw --tag fp32 --out ../../outputs/birdset.json
python n6_birdset.py --datasets PER,NES,UHH,HSN,NBP,SSW,SNE --protocol pre \
  --backbone ../../perch2_backbone_int8.onnx --frontend raw --tag int8_divsmall_raw_asym --out ../../outputs/birdset.json
python n6_beans.py --datasets watkins,dogs,humbugdb \
  --models 'fp32=../../perch2_backbone_fp32.onnx=raw;int8_divsmall_raw_asym=../../perch2_backbone_int8.onnx=raw' \
  --out ../../outputs/beans.json
python n6_beans.py --datasets bats --no-resample \
  --models 'fp32=../../perch2_backbone_fp32.onnx=raw;int8_divsmall_raw_asym=../../perch2_backbone_int8.onnx=raw' \
  --out ../../outputs/beans.json
```

Do not pass `--models` to BirdSet's default `pre` protocol: that option belongs
to embedding extraction, a different evaluation mode. Outputs above are
separate from the immutable historical reports in `results/`.

`n6_conformance.py` requires a local SavedModel reference NPZ containing audio,
spectrograms and embeddings. This historical check is separate from the public
synthetic regression check (`python -m pytest -q`). No private reference audio
is redistributed. The legacy `n6_calib_mels.py` additionally needs its corpus
manifest and local audio; use the standalone calibration tool above instead.

## STM32N6 compilation

Install ST Edge AI Core separately from STMicroelectronics and follow its licence.
From the repository root:

```bash
python research/src/n6_compile.py --model ../../perch2_backbone_int8.onnx \
  --sdk /path/to/directory/containing/stedgeai
```

The historical report used Core 2.2.0-20266 and Neural-ART 1.1.1-14. The script
adds explicit zero Pad inputs. Compiler output is not hardware validation.
