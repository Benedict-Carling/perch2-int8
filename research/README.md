# Research recipe and reproduction status

The runtime quickstart is self-contained. This directory holds the research
scripts and a pinned end-to-end BirdSet reproduction command. Benchmark audio
is downloaded when requested and is not bundled. The original 192-window
calibration corpus is not bundled: it is needed to regenerate the exact INT8
weights, but not to evaluate the released FP32 and INT8 ONNX files.
Checkpoint bytes are pinned to the verified mirror revision in `manifest.json`;
correspondence to the paper variants remains unresolved.

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

The BirdSet comparison uses the same seven `test_5s` soundscape subsets as the
Perch 2.0 benchmark (PER, NES, UHH, HSN, NBP, SSW, SNE). The data snapshot is
pinned to commit `806ed2cda4ddcbe6efa194ccafff930aa0e557ce` in
[`DBD-research-group/BirdSet`](https://huggingface.co/datasets/DBD-research-group/BirdSet/tree/806ed2cda4ddcbe6efa194ccafff930aa0e557ce),
and every downloaded file is checked against `results/birdset_provenance.json`.
The metadata supplies the label names; the evaluation no longer requests the
missing, unpinned `classes.py` file. The runner downloads, verifies, evaluates,
and removes one subset at a time to limit disk use:

```bash
python tools/run_birdset_reproduction.py
```

It writes fresh paired FP32/INT8 results to `outputs/birdset_rerun.json`.
`--datasets NBP` runs one small subset; `--keep-data` retains verified files;
`--limit N` is for pipeline smoke tests only and must not be used for reported
benchmark scores. This evaluates the checked-in model artifacts with the
original Perch 2.0 head. It does not recalibrate the INT8 model. Compare a fresh
run with the checked-in local reports; the paper's rounded metrics are context,
not exact targets, because the precise checkpoint variant and scoring details
used for its tables are not fully resolved here.

In the validation run for this release, full-data NBP, PER, NES, UHH and HSN
scores matched their historical JSON reports to within `1e-7` on every metric.
SSW and SNE were not rerun to completion, so the seven-subset aggregate remains
the checked-in historical aggregate rather than a newly completed full rerun.

BEANS uses the same public task datasets, but the repository's four selected
tasks and local split/probe implementation are not the paper's complete
12-task, official-protocol evaluation. BEANS archives belong in
`research/src/beans/`; the script documents expected archive names and layouts.
The following commands reproduce the repository's local BEANS procedure, not
the paper's exact evaluation:

```bash
cd research/src
python n6_beans.py --datasets watkins,dogs,humbugdb \
  --models 'fp32=../../perch2_backbone_fp32.onnx=raw;int8_divsmall_raw_asym=../../perch2_backbone_int8.onnx=raw' \
  --out ../../outputs/beans.json
python n6_beans.py --datasets bats --no-resample \
  --models 'fp32=../../perch2_backbone_fp32.onnx=raw;int8_divsmall_raw_asym=../../perch2_backbone_int8.onnx=raw' \
  --out ../../outputs/beans.json
```

Outputs are separate from the immutable historical reports in `results/`.

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
