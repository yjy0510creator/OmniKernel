"""OmniKernel 验证框架：为每个算子提供 PyTorch eager 参考实现。

参考实现全部使用 fp32 计算再降回目标 dtype，作为精度比对的金标准。
与 PyTorch 官方 F.* 的等价性在 run_all.py 自检中核验（--self-check 模式）。
"""
from __future__ import annotations

import math
import torch


def _dtype(spec_dtype: str, device: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16,
            "fp32": torch.float32}[spec_dtype]


# ---------------- RMSNorm ----------------
def rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    xf = x.float()
    rms = torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf / rms * w.float()).to(x.dtype)


# ---------------- RoPE (NeoX 半拼接) ----------------
def rope_cos_sin(seq: int, head_dim: int, theta: float = 10000.0,
                 device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32,
                                        device=device) / head_dim))
    pos = torch.arange(seq, dtype=torch.float32, device=device)
    ang = torch.outer(pos, inv)                      # [seq, half]
    emb = torch.cat([ang, ang], dim=-1)              # 半拼接: [seq, head_dim]
    return emb.cos(), emb.sin()


def rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    half = x.shape[-1] // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    c1, c2 = cos[..., :half], cos[..., half:]
    s1, s2 = sin[..., :half], sin[..., half:]
    o1 = x1 * c1 - x2 * s1
    o2 = x2 * c2 + x1 * s2
    return torch.cat([o1, o2], dim=-1).to(x.dtype)


# ---------------- Softmax ----------------
def softmax_ref(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    xf = x.float() * scale
    return torch.softmax(xf, dim=-1).to(x.dtype)


# ---------------- SiLU-Mul ----------------
def silu_mul_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() * torch.sigmoid(a.float()) * b.float()).to(a.dtype)


# ---------------- FlashAttention 前向参考 ----------------
# 首选 F.sdpa（fp32 计算，等价数学定义）；不可用时回退到"行分块 × 全列"手工实现，
# 两种路径都保证 causal / 非 causal 的正确性；分块仅限制 S 物化峰值显存。
def flash_attn_ref(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   causal: bool = True, chunk: int = 128) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    qf, kf, vf = q.float(), k.float(), v.float()
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        o = torch.nn.functional.scaled_dot_product_attention(
            qf, kf, vf, is_causal=causal)
        return o.to(q.dtype)
    b, h, s, d = qf.shape
    o = torch.empty_like(qf)
    col = torch.arange(s, device=qf.device)
    for i in range(0, s, chunk):
        qs = qf[:, :, i:i + chunk, :]
        att = (qs @ kf.transpose(-1, -2)) * scale      # [b,h,rows,s]
        if causal:
            rows = torch.arange(i, i + qs.shape[2], device=qf.device)
            att = att.masked_fill(col[None, None, None, :] >
                                  rows[None, None, :, None], float("-inf"))
        o[:, :, i:i + chunk, :] = torch.softmax(att, dim=-1) @ vf
    return o.to(q.dtype)


# ---------------- Quant GEMM W8A8（参考 = 同量化输入 fp32 反量化 matmul） ----------------
def quant_gemm_ref(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor,
                   sb: torch.Tensor) -> torch.Tensor:
    af = a.float() * sa[:, None]
    bf = b.float() * sb[None, :]
    return (af @ bf).half()


# ---------------- 统一输入生成 ----------------
def make_inputs(op_id: str, wl: dict, dtype: str, device: str = "cpu",
                seed: int = 2026) -> dict:
    g = torch.Generator(device="cpu").manual_seed(seed)
    dt = _dtype(dtype, device)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(dt).to(device)

    if op_id == "rmsnorm":
        b, s, h = wl["batch"], wl["seq"], wl["hidden"]
        return dict(x=rnd(b, s, h), w=rnd(h), eps=1e-5)
    if op_id == "rope":
        b, h, s, d = wl["batch"], wl["heads"], wl["seq"], wl["head_dim"]
        cos, sin = rope_cos_sin(s, d, device=device)
        return dict(x=rnd(b, h, s, d), cos=cos, sin=sin)
    if op_id == "softmax":
        return dict(x=rnd(wl["rows"], wl["cols"], scale=3.0), scale=1.0)
    if op_id == "silu_mul":
        r, it = wl["rows"], wl["inter"]
        return dict(a=rnd(r, it), b=rnd(r, it))
    if op_id == "flash_attn":
        b, h, s, d = wl["batch"], wl["heads"], wl["seq"], wl["dim"]
        causal = bool(wl.get("causal", True))
        return dict(q=rnd(b, h, s, d, scale=0.3), k=rnd(b, h, s, d, scale=0.3),
                    v=rnd(b, h, s, d, scale=0.3), causal=causal)
    if op_id == "quant_gemm_w8a8":
        m, n, kk = wl["M"], wl["N"], wl["K"]
        a = torch.randint(-127, 128, (m, kk), generator=g).to(torch.int8).to(device)
        b = torch.randint(-127, 128, (kk, n), generator=g).to(torch.int8).to(device)
        sa = (torch.rand(m, generator=g) * 0.02 + 0.005).to(device)
        sb = (torch.rand(n, generator=g) * 0.02 + 0.005).to(device)
        return dict(a=a, b=b, sa=sa, sb=sb)
    raise KeyError(f"未知算子 {op_id}")


def call_ref(op_id: str, inp: dict) -> torch.Tensor:
    if op_id == "rmsnorm":
        return rmsnorm_ref(inp["x"], inp["w"], inp["eps"])
    if op_id == "rope":
        return rope_ref(inp["x"], inp["cos"], inp["sin"])
    if op_id == "softmax":
        return softmax_ref(inp["x"], inp["scale"])
    if op_id == "silu_mul":
        return silu_mul_ref(inp["a"], inp["b"])
    if op_id == "flash_attn":
        return flash_attn_ref(inp["q"], inp["k"], inp["v"], inp["causal"])
    if op_id == "quant_gemm_w8a8":
        return quant_gemm_ref(inp["a"], inp["b"], inp["sa"], inp["sb"])
    raise KeyError(op_id)
