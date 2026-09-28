# Evaluation and evidence

The JSON reports are retained from the original experiment. Their model key
`int8_divsmall_raw_asym` identifies the released INT8 recipe. Other arms may
appear in the original reports; they are not release artifacts.

BirdSet uses the existing Perch ProtoPNet head in FP32 on spatial features,
with its class list restricted to each dataset. The head is not bundled in the
runtime model. Scores are macro-averaged across PER, NES, UHH, HSN, NBP, SSW and
SNE. Top-1 excludes unannotated segments; other metrics follow the source code.

BEANS uses frozen embeddings, StandardScaler fitted to training features, and
logistic regression. The included implementation makes stratified 60/20/20
splits with seed 42; the validation split is not used to fit the reported
classifier. This is the local evaluation procedure, not a guarantee of exact
equivalence to every upstream implementation. Bats audio is time-expanded.

The metrics use the existing Python/PyTorch frontend. The release adds a NumPy
frontend verified against the historical SavedModel reference spectrograms,
but does not rerun every benchmark with the new frontend. Changes near a
quantisation boundary can affect individual outputs.

The 64-clip release frontend validation is recorded separately from the
historical `n6_conformance.json`. It is not an independent labelled evaluation.
The reference audio is private and not part of the repository. Public regression
fixtures contain only synthetic signals; they catch implementation drift and
do not establish wildlife recognition accuracy.

For a custom classifier, extract INT8 embeddings for both training and test
audio, split by recording/site/recordist before fitting, train a small linear
model, and evaluate held-out performance and detection thresholds. Training on
FP32 embeddings and deploying on INT8 embeddings can introduce avoidable drift.

No head-to-head benchmark against the community dynamic MatMul INT8 artifact,
confidence-interval analysis, battery measurement or physical STM32N6 inference
run is included.
