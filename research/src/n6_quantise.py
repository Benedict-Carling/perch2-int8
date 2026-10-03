from __future__ import annotations

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.calibrate import HistogramCalibrater

ROOT = Path.cwd()

CHUNK = 8
_orig_collect = HistogramCalibrater.collect_data


class _Sub(CalibrationDataReader):
    def __init__(self, items):
        self.it = iter(items)

    def get_next(self):
        return next(self.it, None)


def _chunked_collect(self, data_reader):
    n, t0 = 0, time.time()
    while True:
        batch = []
        for _ in range(CHUNK):
            x = data_reader.get_next()
            if not x:
                break
            batch.append(x)
        if not batch:
            break
        _orig_collect(self, _Sub(batch))
        n += len(batch)
        print(f"    calibrated {n} samples ({time.time()-t0:.0f}s)", flush=True)


HistogramCalibrater.collect_data = _chunked_collect


class Reader(CalibrationDataReader):
    def __init__(self, mel):
        self.d = [{"mel": m[None, None].astype(np.float32)} for m in mel]
        self.it = iter(self.d)

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter(self.d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--source", required=True, help="Prepared FP32 backbone ONNX; paths relative to cwd")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--symmetric", action="store_true", default=True)
    ap.add_argument("--asymmetric", dest="symmetric", action="store_false")
    ap.add_argument("--raw", action="store_true",
                    help="calibrate on raw log-mel instead of median-subtracted. "
                         "The device frontend must then match: raw in, raw out.")
    ap.add_argument("--ops", default="all",
                    help="'all' (required for the NPU) or a comma-separated list "
                         "such as 'Conv,Gemm'. Restricting ops is DIAGNOSTIC "
                         "only: leaving Sigmoid/Mul/ReduceMean in float32 puts "
                         "74%% of compiled epochs back on the M55. It measures "
                         "how much accuracy all-ops quantisation is costing.")
    ap.add_argument("--recentre", action="store_true",
                    help="median-subtract then add MEDIAN_RECENTRE back, so the "
                         "tensor lands in Perch 2.0's training range")
    ap.add_argument("--percentile", type=float, default=99.99)
    ap.add_argument("--method", default="percentile",
                    choices=["percentile", "minmax", "entropy", "distribution"],
                    help="activation range estimator. Percentile clips the tail "
                         "at a chosen quantile; Entropy picks the clip point "
                         "that minimises KL to the float distribution, which is "
                         "the standard answer for networks whose activations "
                         "have long tails.")
    ap.add_argument("--extra-json", default="",
                    help='extra_options as JSON, e.g. \'{"MinimumRealRange":1e-4}\'')
    args = ap.parse_args()

    if (ROOT / args.out).exists():
        raise SystemExit(f"Refusing to overwrite {args.out}; choose a new output path")

    mel = np.load(ROOT / args.calib)
    if mel.ndim != 3 or mel.shape[1:] != (500, 128) or not len(mel) or not np.isfinite(mel).all():
        raise SystemExit("Calibration must be a non-empty finite [N,500,128] array")
    if args.limit:
        mel = mel[: args.limit]
    if not args.raw:
        mel = mel - np.median(mel, axis=(1, 2), keepdims=True)
        if args.recentre:
            from n6_embed import MEDIAN_RECENTRE
            mel = mel + MEDIAN_RECENTRE
    print(f"[{args.out}] calib={args.calib} n={len(mel)} "
          f"symmetric={args.symmetric} frontend={'raw' if args.raw else ('mediansub_shift' if args.recentre else 'mediansub')} "
          f"range=[{mel.min():.3f}, {mel.max():.3f}]", flush=True)

    t0 = time.time()
    kw = {}
    if args.ops != "all":
        kw["op_types_to_quantize"] = args.ops.split(",")
    method = {"percentile": CalibrationMethod.Percentile,
              "minmax": CalibrationMethod.MinMax,
              "entropy": CalibrationMethod.Entropy,
              "distribution": CalibrationMethod.Distribution}[args.method]
    extra = {"ActivationSymmetric": args.symmetric}
    if args.method == "percentile":
        extra["CalibPercentile"] = args.percentile
    if args.extra_json:
        extra.update(json.loads(args.extra_json))
    print(f"  method={args.method} extra={extra}", flush=True)
    quantize_static(
        str(ROOT / args.source), str(ROOT / args.out),
        calibration_data_reader=Reader(mel),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=method,
        extra_options=extra,
        **kw,
    )
    print(f"  wrote {args.out} "
          f"{(ROOT / args.out).stat().st_size/1e6:.1f} MB in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
