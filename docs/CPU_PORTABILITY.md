# Open CPU portability finding

Release preparation uncovered numerical differences between ONNX Runtime
1.23.2 on macOS arm64 and GitHub-hosted Ubuntu x86_64. The INT8 model bytes are
identical. The strict embedding regression test fails on Linux and is retained
as a visible review gate. The model still executes and returns finite embeddings.

The probes compare four synthetic signals: silence, a chirp, seeded noise and
an impulse. They separately feed identical saved log-mel features and features
computed from audio. Differences persist with saved features, so the frontend
cannot be the sole explanation. Basic or disabled graph optimisation changes
some outputs but does not restore the required cross-host agreement.

Comparing each host's BASIC setting with the Mac BASIC reference yielded
cosines approximately 1.0000, 0.9992, 0.9985 and 0.9968 on one runner. Other
optimisation settings and runners differed. These four values are diagnostics,
not a justified production tolerance or a downstream accuracy assessment.
Exact responsible kernels/instructions have not yet been isolated.

Evidence: `results/cpu_portability.json`. Re-run the targeted probe with:

```bash
python research/debug/compare_cpu_outputs.py
```

No weights were changed to make the test pass. No claim of byte-identical
embeddings, equivalent accuracy across CPUs, or matching physical NPU output is
made. The recorded BirdSet/BEANS tables remain historical measurements from
their original runtime; rerun a representative labelled evaluation on each
intended deployment target before transferring those accuracy claims.

The runtime quickstart and all seven original release checks pass on the local
Mac, including a clean checkout. Linux CI remains a known failed parity gate,
not a completed portability certification.
