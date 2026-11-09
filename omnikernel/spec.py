"""算子 Spec 解析与代码生成。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent          # OmniKernel 仓库根
SPECS_DIR = ROOT / "operator_specs"
TEMPLATES_DIR = ROOT / "templates"
OUT_DIR = ROOT / "generated"


@dataclass
class OperatorSpec:
    """operator_specs/*.yaml 的内存表示（单一事实源）。"""
    raw: dict[str, Any]
    path: Path

    @property
    def id(self) -> str:
        return self.raw["id"]

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def description(self) -> str:
        return str(self.raw.get("description", "")).strip()

    @property
    def math(self) -> str:
        return str(self.raw.get("math", "")).strip()

    @property
    def dtype_support(self) -> list[str]:
        return list(self.raw.get("dtype_support", ["fp16"]))

    @property
    def tolerance(self) -> dict:
        return self.raw.get("tolerance", {})

    @property
    def workloads(self) -> list[dict]:
        return list(self.raw.get("workloads", []))

    def backend(self, name: str) -> dict:
        cg = self.raw.get("codegen", {})
        return cg.get(name, {})

    def template_vars(self, backend: str) -> dict[str, Any]:
        """把 spec 的 codegen.<backend>.params 解析为模板渲染变量。"""
        cfg = self.backend(backend)
        params = dict(cfg.get("params", {}))
        shape = self._representative_shape()
        vars_: dict[str, Any] = {
            "OP_ID": self.id,
            "OP_NAME": self.name,
            "OP_DESC": self.description.replace("\n", " "),
            "MATH": self.math,
            "DTYPE_LIST": "/".join(self.dtype_support),
            "STRATEGY": cfg.get("strategy", ""),
        }
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("auto_pow2("):
                arg = v[len("auto_pow2("):-1]
                vars_[k] = _auto_pow2(_resolve_shape_arg(arg, shape))
            elif v == "auto":
                vars_[k] = _auto_warps(shape)
            else:
                vars_[k] = v
        return vars_

    def _representative_shape(self) -> dict[str, int]:
        """取最大 workload 作为特化参数依据（模板会运行期覆盖，仅影响默认值）。"""
        best: dict[str, int] = {}
        for w in self.workloads:
            if _size(w) >= _size(best):
                best = w
        return best


def _size(w: dict) -> int:
    n = 1
    for v in w.values():
        if isinstance(v, int):
            n *= v
    return n


def _resolve_shape_arg(arg: str, shape: dict[str, int]) -> int:
    if "/" in arg:
        a, b = arg.split("/")
        return shape.get(a.strip(), 1024) // int(b)
    return shape.get(arg, 1024)


def _auto_pow2(n: int) -> int:
    p = 16
    while p < n:
        p *= 2
    return min(p, 32768)


def _auto_warps(shape: dict[str, int]) -> int:
    hidden = max(shape.values())
    if hidden <= 1024:
        return 4
    if hidden <= 4096:
        return 8
    return 16


def load_spec(op_id: str) -> OperatorSpec:
    path = SPECS_DIR / f"{op_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"operator spec not found: {path}")
    return OperatorSpec(yaml.safe_load(path.read_text(encoding="utf-8")), path)


def list_specs() -> list[str]:
    return sorted(p.stem for p in SPECS_DIR.glob("*.yaml"))
