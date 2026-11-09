"""统一验证 CLI。

用法：
    python -m validation.run_all --backend cpu_ref            # 无 GPU 自检（参考实现互检）
    python -m validation.run_all --backend triton --device cuda:0
    python -m validation.run_all --backend triton,ascendc --device npu:0
    python -m validation.run_all --op rmsnorm --dtype fp16,bf16

输出：results/<时间戳>.json + 控制台汇总表。
cpu_ref 模式在无 GPU 的机器上验证"参考实现 + spec 阈值框架"链路正确性；
真机模式追加延迟/显存测量，并自动导出与 AISHPerf 报告字段对齐的 JSON。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from omnikernel.spec import load_spec, list_specs      # noqa: E402
from validation.harness import run_case                 # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="validation.run_all")
    ap.add_argument("--op", default="all")
    ap.add_argument("--backend", default="cpu_ref")
    ap.add_argument("--dtype", default="fp16,bf16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--quick", action="store_true", help="每算子仅跑第一个 workload")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    ops = list_specs() if args.op == "all" else [args.op]
    backends = [b.strip() for b in args.backend.split(",")]
    dtypes = [d.strip() for d in args.dtype.split(",")]

    results = []
    print(f"{'op':14s} {'backend':9s} {'dtype':6s} {'ok':4s} {'max_abs':>10s} "
          f"{'max_rel':>10s} {'lat_us':>10s} {'mem_mb':>8s}  workload")
    print("-" * 100)
    for op_id in ops:
        spec = load_spec(op_id)
        for backend in backends:
            workloads = spec.workloads[:1] if args.quick else spec.workloads
            for wl in workloads:
                for dtype in dtypes:
                    if dtype not in spec.dtype_support:
                        continue
                    tol = spec.tolerance.get(dtype, {"atol": 1e-2, "rtol": 1e-2})
                    try:
                        r = run_case(op_id, backend, wl, dtype, tol,
                                     args.device, ROOT, bench=not args.no_bench)
                    except FileNotFoundError as e:
                        print(f"{op_id:14s} {backend:9s}  SKIP  {e}")
                        continue
                    results.append(r.to_dict())
                    flag = "PASS" if r.ok else "FAIL"
                    print(f"{r.op:14s} {r.backend:9s} {r.dtype:6s} {flag:4s} "
                          f"{r.max_abs:10.2e} {r.max_rel:10.2e} "
                          f"{r.latency_us:10.1f} {r.peak_mem_mb:8.1f}  {wl}")
                    if r.note:
                        print(f"    note: {r.note}")

    outdir = ROOT / args.out
    outdir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    fp = outdir / f"report_{ts}.json"
    fp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    n_pass = sum(1 for r in results if r["ok"])
    print(f"\n合计 {n_pass}/{len(results)} PASS，报告: {fp}")
    return 0 if n_pass == len(results) and results else 1


if __name__ == "__main__":
    raise SystemExit(main())
