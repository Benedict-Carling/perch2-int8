# Private release review

Candidate: `0.1.0-rc.1`. Neither destination should be made public without
Benedict's explicit approval.

## Included for review

- Standalone NumPy/SciPy audio frontend and ONNX Runtime embedding example.
- INT8 release graph, exact prepared FP32 comparison backbone and mel filterbank.
- Synthetic WAV and independent direct-DFT feature/embedding regression fixture.
- Historical benchmark reports with a clear distinction from release smoke tests.
- Model hashes and verification that the historical benchmark graph becomes the
  shipped graph through the documented explicit-zero Pad fix only.
- Research code and corrected evaluation commands, with prerequisite gaps stated.

## Decisions/checks before public release

**Open numerical portability finding:** macOS arm64 and Linux x86_64 produce
measurably different embeddings from identical saved features under ONNX Runtime
1.23.2. The release checks now use the Mac golden only on its generating host;
Linux CI checks valid, repeatable inference. This resolves the misleading red
CI gate, but does not establish cross-platform parity or task accuracy on Linux.
Compare representative labelled evaluations on the intended deployment hosts
before making downstream accuracy claims. See `docs/CPU_PORTABILITY.md` and
`results/cpu_portability.json`.

1. Review the model card and confirm this is presented as an embedding research
   preview, with no species head or completed MCU firmware implied.
2. Review upstream attribution and checkpoint variant. The source bytes match
   the pinned `cgeorgiaw/Perch` mirror revision recorded in `manifest.json`.
   Its correspondence to Kaggle version numbers and paper variants is unresolved;
   do not invent those identifiers.
3. Review calibration provenance before distributing calibration audio or
   derived features. These are deliberately excluded from this release.
   Exact end-to-end calibration-set recreation is not yet packaged.
4. Exercise the full benchmark setup from upstream downloads before advertising
   one-command reproduction of the historical tables. The release tests are
   inference/packaging checks, not this full research rerun.
5. Review a simple downstream classifier example on an appropriately licensed
   labelled dataset before making task-specific performance claims.

Board validation, measured latency/power and a complete embedded implementation
would strengthen a later release. They are not prerequisites for publishing an
honestly scoped research preview, but the limitations must remain prominent.
