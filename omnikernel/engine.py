"""模板/特化层：把 spec 参数渲染进后端模板，并用 py_compile 做语法闸门。"""
from __future__ import annotations

import py_compile
import tempfile
from pathlib import Path
from string import Template

from omnikernel.spec import OperatorSpec, TEMPLATES_DIR, ROOT

BACKEND_LANG = {"triton": "py", "cuda": "py", "ascendc": "cpp"}


def render(spec: OperatorSpec, backend: str) -> str:
    """渲染 spec 指定的后端模板，返回源码字符串。"""
    cfg = spec.backend(backend)
    if not cfg:
        raise KeyError(f"spec {spec.id} 未定义 codegen.{backend}")
    # spec 内 template 路径相对仓库根（如 templates/triton/xxx.py.tmpl）
    tpath = ROOT / cfg["template"]
    if not tpath.exists():
        tpath = TEMPLATES_DIR / backend / f"{spec.id}.{'py' if backend!='ascendc' else 'cpp'}.tmpl"
    tmpl = Template(tpath.read_text(encoding="utf-8"))
    return tmpl.safe_substitute(spec.template_vars(backend))


def syntax_check(source: str, backend: str) -> None:
    """语法闸门：Python 后端走 py_compile；C++ 后端仅做括号/引号平衡粗检。"""
    if BACKEND_LANG.get(backend) == "py":
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(source)
            tmp = f.name
        try:
            py_compile.compile(tmp, doraise=True)
        finally:
            Path(tmp).unlink(missing_ok=True)
    else:
        if source.count("{") != source.count("}"):
            raise SyntaxError("大括号不平衡")


def emit(source: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(source, encoding="utf-8")
    return out_path
