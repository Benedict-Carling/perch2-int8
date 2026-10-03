# Mac vs Linux output

The same INT8 model gives slightly different embeddings on macOS arm64 and
Linux x86_64 (ONNX Runtime 1.23.2), even when both get identical log-mel input.
So the difference comes from the runtime's INT8 kernels, not from the frontend.

On four synthetic test signals the cosine similarity between the two was
between 0.991 and 1.000. Changing ONNX Runtime's graph optimisation level
shifts the output a little too, even on one machine.

What this means in practice:

- Train and run your classifier on the same platform where you can.
- The accuracy numbers in the README were measured on the original setup. Check
  them again on your own target before relying on them.

The tests check the frontend exactly on every platform. They only compare
embeddings exactly on Mac. On Linux they check that the output is valid and
repeatable.

The raw numbers are in `results/cpu_portability.json`. To re-run:

```bash
python research/debug/compare_cpu_outputs.py
```
