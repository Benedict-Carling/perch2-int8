# Evaluation

The raw reports are in `results/`. The released model is the
`int8_divsmall_raw_asym` arm in those files.

**BirdSet.** Perch's own ProtoPNet head (FP32) runs on the backbone features.
Each dataset uses only its own class list. The scores are averaged over PER,
NES, UHH, HSN, NBP, SSW and SNE. POW is left out because Perch used it for
model selection.

**BEANS.** The embeddings are frozen and scaled, then fed to a logistic
regression. Splits are a stratified 60/20/20 with seed 42. It covers four tasks
(Watkins, Dogs, Bats, HumBugDB), not the full twelve. Bat audio is
time-expanded.

The scores came from the original PyTorch frontend. The NumPy frontend in this
repo matches it to within 5e-5 on 64 reference clips, but I didn't re-run every
benchmark with it.

**Training your own classifier.** Extract INT8 embeddings for both training and
test audio. Split by site or recording, not by window, then fit a small linear
model. If you train on FP32 embeddings and deploy on INT8, the results will
drift.

**Not done yet:** a head-to-head with other INT8 Perch ports, confidence
intervals, and inference on a physical STM32N6.
