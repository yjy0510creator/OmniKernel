"""精度 + 延迟 + 显存 三合一测试 harness。

后端接口约定：所有生成代码暴露 `run(**inputs) -> Tensor`
（AscendC 后端通过 torch_npu 扩展接入，接口同名）。
"""
from __future__ import annotations

import importlib.util
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch

from validation import reference as ref


@dataclass
class CaseResult:
    op: str
    backend: str
    dtype: str
    workload: dict
    ok: bool
    max_abs: float = 0.0
    max_rel: float = 0.0
    latency_us: float = float("nan")
    peak_mem_mb: float = float("nan")
    baseline_us: float = float("nan")
    note: str = ""

    def to_dict(self):
        return asdict(self)


def load_backend_run(op_id: str, backend: str, root: Path):
    """动态加载 generated/<op>/<backend>/kernel.py 的 run。"""
    if backend in ("cpu_ref",):
        return lambda **kw: ref.call_ref(op_id, kw)
    path = root / "generated" / op_id / backend / "kernel.py"
    if not path.exists():
        raise FileNotFoundError(f"未生成的后端: {path}")
    spec = importlib.util.spec_from_file_location(f"gen_{op_id}_{backend}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run


def accuracy(op_id: str, run_fn, wl: dict, dtype: str, device: str,
             atol: float, rtol: float, seed: int = 2026) -> tuple[bool, float, float]:
    # 参考实现内部按算子语义自行升/降精度（int8 需保持整型），故输入原样传入。
    inp = ref.make_inputs(op_id, wl, dtype, device=device, seed=seed)
    y_ref = ref.call_ref(op_id, inp)
    y = run_fn(**inp)
    yf, rf = y.float(), y_ref.float()
    max_abs = (yf - rf).abs().max().item()
    denom = rf.abs().clamp_min(1e-6)
    max_rel = ((yf - rf).abs() / denom).max().item()
    ok = torch.allclose(yf, rf, atol=atol, rtol=rtol)
    return ok, max_abs, max_rel


def benchmark(fn, device: str, warmup: int = 10, iters: int = 50) -> tuple[float, float]:
    """返回 (中位延迟 us, 峰值显存 MB)。"""
    dev = torch.device(device)
    for _ in range(warmup):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters * 1e6
    mem = torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda" else float("nan")
    return dt, mem


def run_case(op_id: str, backend: str, wl: dict, dtype: str, tol: dict,
             device: str, root: Path, bench: bool = True) -> CaseResult:
    run_fn = load_backend_run(op_id, backend, root)
    atol, rtol = tol["atol"], tol["rtol"]
    note = ""
    try:
        ok, ma, mr = accuracy(op_id, run_fn, wl, dtype, device, atol, rtol)
    except Exception as e:
        return CaseResult(op_id, backend, dtype, wl, False, note=f"执行异常: {e}")
    res = CaseResult(op_id, backend, dtype, wl, ok, ma, mr)
    if bench and ok and device.startswith("cuda"):
        inp = ref.make_inputs(op_id, wl, dtype, device=device)
        try:
            lat, mem = benchmark(lambda: run_fn(**inp), device)
            base = ref.call_ref
            lat_b, _ = benchmark(
                lambda: base(op_id, inp), device, warmup=3, iters=10)
            res.latency_us, res.peak_mem_mb, res.baseline_us = lat, mem, lat_b
        except Exception as e:
            note = f"benchmark 异常: {e}"
    res.note = note
    return res
