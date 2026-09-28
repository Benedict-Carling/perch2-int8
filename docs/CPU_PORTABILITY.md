# CPU portability finding and CI scope

Release preparation uncovered numerical differences between ONNX Runtime
1.23.2 on macOS arm64 and GitHub-hosted Ubuntu x86_64. The INT8 model bytes are
identical. The original strict embedding comparison used a macOS arm64 ORT
output as the golden on every platform. That comparison fails on Linux x86_64,
even though inference completes and returns finite embeddings.

The test now keeps the frontend check strict on every platform, keeps the
embedding golden comparison on its generating platform (macOS arm64), and
checks finite output plus repeatability on Linux x86_64. This makes Linux CI an
honest execution check without treating architecture-specific QDQ results as a
universal golden.

The probes compare four synthetic signals: silence, a chirp, seeded noise and
an impulse. They separately feed identical saved log-mel features and features
computed from audio. Differences persist with saved features, so the frontend
cannot be the sole explanation. Basic or disabled graph optimisation changes
some outputs but does not restore cross-host agreement. Changing optimisation
level also shifts output on the same Mac, confirming the previous golden was
tied to a particular runtime path.

Comparing each host's BASIC setting with the Mac BASIC reference yielded
cosines approximately 1.0000, 0.9992, 0.9985 and 0.9968 on one runner. Other
optimisation settings and runners differed. These four values are diagnostics,
not a justified production tolerance or a downstream accuracy assessment.

Evidence: `results/cpu_portability.json`. Re-run the targeted probe with:

```bash
python research/debug/compare_cpu_outputs.py
```

No weights or inference settings were changed to make the test pass. A green
Linux CI run means the pinned Linux runtime can execute the model, return valid
embeddings, and repeat deterministically for these fixtures. It does not
establish Linux-to-Mac embedding parity, equivalent task accuracy across CPUs,
or matching physical NPU output. The recorded BirdSet/BEANS tables remain
historical measurements from their original runtime; rerun a representative
labelled evaluation on each intended deployment target before transferring
those accuracy claims.

The runtime quickstart and release checks pass on the local Mac, including a
clean checkout. Linux CI is an execution and determinism check, not a completed
cross-platform numerical parity or deployment-accuracy certification.
