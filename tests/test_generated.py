"""generated/ 产物静态校验（无 GPU 依赖）：语法、残留占位符、接口约定。"""
from __future__ import annotations

import py_compile
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "generated"
FAILS = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def py_files():
    return sorted(GEN.glob("*/*/kernel.py"))


def cpp_files():
    return sorted(GEN.glob("*/*/kernel.cpp"))


if __name__ == "__main__":
    pys = py_files()
    cps = cpp_files()
    check("生成产物存在", bool(pys and cps), f"(py={len(pys)} cpp={len(cps)})")
    for p in pys:
        src = p.read_text(encoding="utf-8")
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                             encoding="utf-8") as f:
                f.write(src)
                tmp = f.name
            py_compile.compile(tmp, doraise=True)
            Path(tmp).unlink(missing_ok=True)
            ok, err = True, ""
        except Exception as e:
            ok, err = False, str(e)
            Path(tmp).unlink(missing_ok=True)
        check(f"py_compile {p.parent.parent.name}/{p.parent.name}", ok, err)
        if "${" in src:
            check(f"占位符残留 {p}", False)
        if "def run(" not in src:
            check(f"接口约定 run() {p}", False)
    for p in cps:
        src = p.read_text(encoding="utf-8")
        check(f"cpp 括号平衡 {p.parent.parent.name}", src.count("{") == src.count("}"))
        check(f"cpp 无残留变量 {p.parent.parent.name}", "${" not in src)
    print("-" * 50)
    print("全部通过" if not FAILS else f"失败: {FAILS}")
    raise SystemExit(1 if FAILS else 0)
