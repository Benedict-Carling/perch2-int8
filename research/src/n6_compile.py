from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GATE5_MIN_NPU_FRACTION = 0.95
ONCHIP_LIMIT_BYTES = 2.75 * 1024 * 1024


def parse_report(text: str) -> dict:
    out = {}
    m = re.search(r"Total number of epochs:\s*(\d+)\s*of which\s*(\d+)\s*implemented in software", text)
    if m:
        total, sw = int(m.group(1)), int(m.group(2))
        out.update(epochs_total=total, epochs_software=sw,
                   npu_fraction=round((total - sw) / total, 4))
    return out


def parse_c_info(path: Path) -> dict:
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    out = {}
    fp = d.get("memory_footprint") or {}
    for k in ("weights", "activations", "kernel_flash", "kernel_ram"):
        if k in fp:
            out[f"fp_{k}"] = fp[k]
    pools = d.get("memory_pools") or []
    contained = {i for p in pools for i in (p.get("subpools") or [])}
    top = [p for p in pools if p.get("id") not in contained and p.get("used_size_bytes")]
    used = {p["name"]: p["used_size_bytes"] for p in top}
    if used:
        out["pools_used"] = used
        out["onchip_bytes"] = sum(v for k, v in used.items()
                                  if "cpuRAM" in k or "npuRAM" in k)
        out["psram_bytes"] = used.get("hyperRAM", 0)
        out["flash_bytes"] = used.get("octoFlash", 0)
        out["onchip_within_limit"] = out["onchip_bytes"] <= ONCHIP_LIMIT_BYTES

    pe = d.get("power_estimates") or []
    cycles = sum(p.get("max_cycles", 0) for p in pe)
    if cycles:
        out["compiler_cycles"] = cycles
        out["compiler_cycle_ms_at_1ghz"] = round(cycles / 1e6, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sdk", required=True, help="dir containing the stedgeai binary")
    ap.add_argument("--profile", default="", help="e.g. profile-allmems--O3")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    src = ROOT / args.model
    stem = src.stem
    fixed = ROOT / f"{stem}_padfix.onnx"
    if not fixed.exists():
        from padfix_n6 import padfix
        n = padfix(str(src), str(fixed))
        print(f"padfix: {n} Pad node(s) fixed -> {fixed.name}")
    else:
        print(f"padfix: {fixed.name} exists")

    ws = ROOT / f"_stedgeai_ws_{stem}"
    outdir = ROOT / f"_stedgeai_out_{stem}"
    cmd = [str(Path(args.sdk) / "stedgeai"), "analyze",
           "--model", str(fixed), "--target", "stm32n6", "--st-neural-art"]
    if args.profile:
        cmd[-1] = f"--st-neural-art"
        cmd.append(args.profile)
    cmd += ["--workspace", str(ws), "--output", str(outdir), "--with-report"]
    print("  " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = p.stdout + p.stderr
    (ROOT / f"_stedgeai_{stem}.log").write_text(log)
    if p.returncode != 0:
        print(f"COMPILE FAILED rc={p.returncode}; see _stedgeai_{stem}.log")
        print(log[-3000:])
        raise SystemExit(1)

    rep = outdir / "network_analyze_report.txt"
    text = rep.read_text() if rep.exists() else log
    res = {"model": args.model, "returncode": p.returncode}
    res.update(parse_report(text))
    res.update(parse_c_info(outdir / "network_c_info.json"))
    if "npu_fraction" in res:
        res["gate5_npu_pass"] = bool(res["npu_fraction"] >= GATE5_MIN_NPU_FRACTION)
        print(f"  epochs {res['epochs_total']} total / {res['epochs_software']} software"
              f"  -> {res['npu_fraction']*100:.1f}% on NPU  "
              f"{'PASS' if res['gate5_npu_pass'] else 'FAIL'}")
    for k in ("fp_weights", "fp_activations", "cycles"):
        if k in res:
            print(f"  {k}: {res[k]:,}")

    dst = ROOT / (args.out or f"n6_compile_{stem}.json")
    dst.write_text(json.dumps(res, indent=2))
    if rep.exists():
        (ROOT / "stedgeai_reports").mkdir(exist_ok=True)
        (ROOT / "stedgeai_reports" / f"{stem}.analyze.txt").write_text(text)
    print(f"wrote {dst.name}")


if __name__ == "__main__":
    main()
