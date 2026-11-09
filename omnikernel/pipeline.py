"""OmniKernel 生成流水线 CLI。

用法：
    python -m omnikernel gen --op rmsnorm --backend triton            # 模板模式
    python -m omnikernel gen --op all     --backend triton,ascendc    # 全算子
    python -m omnikernel gen --op rmsnorm --backend triton --agent    # LLM Agent 增强
    python -m omnikernel gen --op all     --backend triton --agent --max-rounds 3

模板模式确定性输出（可复现）；--agent 模式在模板产物之上让 LLM 做优化改写，
并用 feedback 闭环（语法闸门→编译→数值）迭代修复，最多 max-rounds 轮，
任一轮失败即回滚到上一可用版本（模板产物永远保底）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omnikernel import engine, spec as spec_mod
from omnikernel.feedback import compile_gate, make_feedback_prompt, syntax_gate
from omnikernel.llm import LLMClient
from omnikernel.spec import OUT_DIR

AGENT_SYSTEM = (
    "你是资深异构计算算子工程师，精通 Triton/CUDA 与昇腾 AscendC。"
    "给定算子数学定义与参考实现骨架，输出保持接口签名不变、数值正确、"
    "针对目标硬件深度优化的完整源码。只输出一个代码块。"
)


def gen_one(op_id: str, backend: str, use_agent: bool, max_rounds: int,
            llm: LLMClient, quiet: bool = False) -> Path:
    s = spec_mod.load_spec(op_id)
    source = engine.render(s, backend)
    if backend == "ascendc":
        try:                        # C++ 后端只做括号/模板变量平衡检查
            engine.syntax_check(source, backend)
            leftover = [ln for ln in source.splitlines() if "${" in ln]
            if leftover:
                raise SyntaxError("存在未替换模板变量")
        except SyntaxError as e:
            raise RuntimeError(f"模板语法闸门失败: {e}")
    else:
        gate = syntax_gate(source)
        if not gate.ok:             # 模板本身有问题属于开发期 bug，直接报出
            raise RuntimeError(f"模板语法闸门失败: {gate.message}")

    out_name = {"triton": "kernel.py", "ascendc": "kernel.cpp", "cuda": "kernel.cu"}[backend]
    out_path = OUT_DIR / op_id / backend / out_name
    engine.emit(source, out_path)
    mode = "template"

    if use_agent and backend in ("triton", "cuda") and llm.available():
        brief = f"# {s.name}\n{s.description}\n数学定义:\n{s.math}\n接口: {s.raw.get('io')}"
        candidate = source
        for rnd in range(1, max_rounds + 1):
            reply = llm.chat(AGENT_SYSTEM,
                             f"请在保持接口不变的前提下优化这个 {backend} 算子：\n"
                             f"## 算子定义\n{brief}\n## 基线代码\n```python\n{candidate}\n```"
                             f"\n输出优化后的完整代码。")
            cand_code = LLMClient.extract_code(reply)
            g = syntax_gate(cand_code)
            if g.ok:
                g = compile_gate(cand_code, run_numeric=False)
            if g.ok:
                source, candidate, mode = cand_code, cand_code, f"agent(rnd={rnd})"
                engine.emit(source, out_path)
            else:
                prompt = make_feedback_prompt(cand_code, g, brief)
                try:
                    reply2 = LLMClient.extract_code(llm.chat(AGENT_SYSTEM, prompt))
                except Exception:
                    break
                if syntax_gate(reply2).ok:
                    source, candidate, mode = reply2, reply2, f"agent-fix(rnd={rnd})"
                    engine.emit(source, out_path)
                else:
                    break  # 回滚保底：保留模板版本
    elif use_agent and not llm.available():
        print("[warn] 未配置 OMNIKERNEL_LLM_*，--agent 自动退化为模板模式", file=sys.stderr)

    if not quiet:
        print(f"[gen] {op_id:16s} {backend:8s} -> {out_path.relative_to(OUT_DIR.parent)} ({mode}, {len(source.splitlines())} lines)")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="omnikernel")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="生成算子代码")
    g.add_argument("--op", default="all", help="算子 id 或 all")
    g.add_argument("--backend", default="triton,ascendc", help="逗号分隔：triton,ascendc,cuda")
    g.add_argument("--agent", action="store_true", help="启用 LLM Agent 优化改写")
    g.add_argument("--max-rounds", type=int, default=3)

    l = sub.add_parser("list", help="列出可用算子 spec")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        for i in spec_mod.list_specs():
            print(i)
        return 0

    ops = spec_mod.list_specs() if args.op == "all" else [args.op]
    backends = [b.strip() for b in args.backend.split(",")]
    llm = LLMClient()
    for op in ops:
        for b in backends:
            gen_one(op, b, args.agent, args.max_rounds, llm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
