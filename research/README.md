# Research scripts

You don't need anything here to use the model. These are the scripts behind the
numbers in the main README. Install their extra dependencies first:

```bash
pip install -r research/requirements.txt
```

## Quantise with your own calibration audio

```bash
python tools/calibration_features.py clip1.wav clip2.wav --out outputs/calib.npy
python research/src/n6_quantise.py --source perch2_backbone_fp32.onnx \
  --calib outputs/calib.npy --raw --asymmetric --ops all \
  --method percentile --percentile 99.99 --out outputs/requantised.onnx
python research/src/padfix_n6.py outputs/requantised.onnx outputs/requantised_padfix.onnx
```

It takes one window per file, so use a varied set of real recordings. The
original calibration set isn't included, so your result won't match the
released weights bit for bit.

## Re-run BirdSet

BirdSet needs the original Perch 2.0 checkpoint for its species head, plus
TensorFlow:

```bash
hf download cgeorgiaw/Perch --revision c3c3c816976a6c1bb75140b8fbaef7397a6f708f \
  --include 'variables/*' 'assets/*' 'saved_model.pb' --local-dir outputs/perch2_savedmodel
python tools/prepare_research.py --savedmodel outputs/perch2_savedmodel \
  --classes outputs/perch2_savedmodel/assets/perch_v2_ebird_classes.csv
python tools/run_birdset_reproduction.py
```

The runner downloads each BirdSet subset at a pinned commit and checks its
hashes. It then scores FP32 and INT8, deletes the data and moves on to the next
subset. Use `--datasets NBP` for a quick run. Results go to
`outputs/birdset_rerun.json`.

When I re-ran NBP, PER, NES, UHH and HSN, they matched the saved reports to
within 1e-7. SSW and SNE haven't been re-run yet.

## Re-run BEANS

Put the BEANS archives in `research/src/beans/`, then:

```bash
cd research/src
MODELS='fp32=../../perch2_backbone_fp32.onnx=raw;int8_divsmall_raw_asym=../../perch2_backbone_int8.onnx=raw'
python n6_beans.py --datasets watkins,dogs,humbugdb --models "$MODELS" --out ../../outputs/beans.json
python n6_beans.py --datasets bats --no-resample --models "$MODELS" --out ../../outputs/beans.json
```

This runs my four-task setup, not the paper's full BEANS protocol.

## Compile for STM32N6

Install ST Edge AI Core from ST first, then run:

```bash
python research/src/n6_compile.py --model ../../perch2_backbone_int8.onnx \
  --sdk /path/to/directory/containing/stedgeai
```

The saved report used Core 2.2.0-20266 and Neural-ART 1.1.1-14.
