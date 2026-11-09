#!/usr/bin/env bash
# AISHPerf 评测执行入口（需在评测机运行，真机数据才有效）
# 前置：已 pip install -r requirements.txt 且（昇腾机）装好 CANN+torch_npu、
#       （GPU 机）装好 CUDA12+triton。
set -euo pipefail
cd "$(dirname "$0")/../.."        # 回到 OmniKernel 根

BACKEND="${1:-triton}"            # triton | ascendc
DEVICE="${2:-cuda:0}"             # cuda:0 | npu:0

echo "[1/3] 生成算子代码（模板模式，可复现）"
python -m omnikernel gen --op all --backend "$BACKEND"

echo "[2/3] 运行精度 + 性能验证（对齐 AISHPerf 三阶段）"
python -m validation.run_all --op all --backend "$BACKEND" \
    --dtype fp16,bf16 --device "$DEVICE" --out aishperf/run/results

echo "[3/3] 生成 AISHPerf 报告骨架"
python - <<'PY'
import json, glob, os
files = sorted(glob.glob("aishperf/run/results/report_*.json"))
if not files:
    print("no results found"); raise SystemExit(1)
rows = json.load(open(files[-1], encoding="utf-8"))
by_op = {}
for r in rows:
    by_op.setdefault(r["op"], []).append(r)
summary = []
for op, rs in by_op.items():
    ok = all(x["ok"] for x in rs)
    lats = [x["latency_us"] for x in rs if x["latency_us"] == x["latency_us"]]
    base = [x["baseline_us"] for x in rs if x["baseline_us"] == x["baseline_us"]]
    sp = (sum(b/l for b,l in zip(base,lats))/len(lats)) if lats and len(base)==len(lats) else None
    summary.append({
        "op": op,
        "precision_pass_rate": sum(x["ok"] for x in rs)/len(rs),
        "mean_latency_us": (sum(lats)/len(lats)) if lats else None,
        "mean_speedup": sp,
        "max_peak_mem_mb": max((x["peak_mem_mb"] for x in rs), default=None),
    })
json.dump({"suite": "omnikernel_operator_gen",
           "note": "数值来自评测机实测；未回填项为 None",
           "summary": summary},
          open("aishperf/run/summary.json","w",encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("summary ->", "aishperf/run/summary.json")
PY
echo "完成。请核对 aishperf/run/summary.json 后归档 results/。"
