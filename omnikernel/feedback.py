"""编译-测试反馈闭环：语法闸门 +（有硬件时）编译与数值测试，报错回喂修复。"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GateResult:
    ok: bool
    stage: str          # syntax | compile | numeric
    message: str = ""


def syntax_gate(source: str) -> GateResult:
    """不依赖 GPU 的静态闸门：语法 + 常见生成错误模式检查。"""
    try:
        compile(source, "<generated>", "exec")
    except SyntaxError as e:
        return GateResult(False, "syntax", f"{e.msg} (line {e.lineno})")
    problems = []
    for bad, why in [
        ("import triton", None),
        ("TODO", "包含未展开的 TODO 占位"),
        ("${", "存在未替换的模板变量"),
        ("undefined", "疑似残留 undefined 标识"),
    ]:
        if why and bad in source:
            problems.append(why)
    if problems:
        return GateResult(False, "syntax", "；".join(problems))
    return GateResult(True, "syntax")


def compile_gate(source: str, run_numeric: bool) -> GateResult:
    """在有 GPU/triton 的机器上 import 模块并可选跑一个最小数值检查。"""
    with tempfile.NamedTemporaryFile("w", suffix="_gen.py", delete=False,
                                     encoding="utf-8", dir=".") as f:
        f.write(source)
        tmp = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-c",
             f"import importlib.util,sys;"
             f"spec=importlib.util.spec_from_file_location('g','{tmp}');"
             f"m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
             + ("import torch;assert torch.cuda.is_available()" if run_numeric else "")],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout)[-1500:]
            return GateResult(False, "compile", err.strip())
        return GateResult(True, "compile")
    except subprocess.TimeoutExpired:
        return GateResult(False, "compile", "编译/导入超时(300s)")
    finally:
        Path(tmp).unlink(missing_ok=True)


def make_feedback_prompt(source: str, result: GateResult, spec_brief: str) -> str:
    """把失败上下文组装成回喂给 LLM 的修复提示。"""
    return (
        f"以下算子代码在 {result.stage} 阶段失败。\n"
        f"## 算子定义\n{spec_brief}\n"
        f"## 当前代码\n```python\n{source}\n```\n"
        f"## 错误信息\n{result.message}\n"
        "请只输出修复后的完整代码（一个 ```python 代码块），不要解释。"
    )
