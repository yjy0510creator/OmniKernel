#!/usr/bin/env bash
# OmniKernel 一键构建脚本：生成全部算子（双后端）+ 语法闸门校验。
# 用法:
#   bash build.sh                 # 模板模式（确定性、可复现，推荐）
#   bash build.sh --agent         # 追加 LLM Agent 优化（需配置 OMNIKERNEL_LLM_*）
set -euo pipefail
cd "$(dirname "$0")"

BACKENDS="triton"
# 仅当本机具备昇腾工具链时才尝试生成 ascendc（atvc 存在与否自动探测）
if command -v atvc >/dev/null 2>&1; then
  BACKENDS="triton,ascendc"
  echo "[build] 检测到 atvc，启用 ascendc 后端"
else
  echo "[build] 未检测到 atvc，跳过 ascendc 代码生成（模板仍在 templates/ascendc）"
fi

if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi

echo "[build] 生成算子代码 -> generated/"
$PY -m omnikernel gen --op all --backend "$BACKENDS" "$@"

echo "[build] 语法/导入自检（不依赖 GPU 的部分）"
$PY -m validation.run_all --op all --backend cpu_ref --dtype fp16 \
    --device cpu --quick --no-bench --out results

echo "[build] 完成。下一步在评测机执行: bash aishperf/run/run_eval.sh <backend> <device>"
