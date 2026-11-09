"""参考实现交叉核验（CPU 可跑，无需 GPU）。

目的：证明 validation/reference.py 的金标准本身正确——
与 PyTorch 官方算子 / 精确整数运算逐项交叉比对。
这是"算子生成质量-验证过程"链条的第一环：
  官方实现 ≈ OmniKernel 参考实现 ≈（真机上）≈ 生成 kernel。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch                                            # noqa: E402
from validation import reference as ref                 # noqa: E402

G = torch.Generator().manual_seed(2026)
FAILS = []


def check(name: str, cond: bool, detail: str = ""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name} {detail}")
    if not cond:
        FAILS.append(name)


def t_rmsnorm():
    x = torch.randn(4, 7, 64, generator=G)
    w = torch.randn(64, generator=G)
    y1 = ref.rmsnorm_ref(x, w)
    if hasattr(torch.nn.functional, "rms_norm"):
        y2 = torch.nn.functional.rms_norm(x.float(), (64,), weight=w, eps=1e-5)
        check("rmsnorm vs F.rms_norm", torch.allclose(y1, y2, atol=1e-5))
    else:  # 手工 fp64 金标准
        xf = x.double()
        y2 = (xf / (xf.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
              * w.double()).float()
        check("rmsnorm vs fp64-manual", torch.allclose(y1, y2, atol=1e-4))


def t_softmax():
    x = torch.randn(128, 256, generator=G) * 3
    check("softmax vs F.softmax",
          torch.allclose(ref.softmax_ref(x), torch.softmax(x.float(), -1),
                         atol=1e-6))


def t_silu_mul():
    a = torch.randn(64, 128, generator=G)
    b = torch.randn(64, 128, generator=G)
    y2 = (torch.nn.functional.silu(a.float()) * b.float()).to(a.dtype)
    check("silu_mul vs F.silu*Mul", torch.allclose(ref.silu_mul_ref(a, b), y2,
                                                   atol=1e-6))


def t_rope():
    b, h, s, d = 2, 3, 16, 32
    x = torch.randn(b, h, s, d, generator=G)
    cos, sin = ref.rope_cos_sin(s, d)
    y1 = ref.rope_ref(x, cos, sin)
    # 复数旋转金标准：把 (x1,x2) 视作复数对 z=x1+i*x2，乘 e^{iθ_p}（逐通道对）
    z = torch.complex(x[..., :d // 2].float(), x[..., d // 2:].float())
    rot = torch.complex(cos[..., :d // 2], sin[..., :d // 2])   # [s, half]
    z2 = (z * rot[None, None, :, :])
    y2 = torch.cat([z2.real, z2.imag], dim=-1)
    check("rope vs 复数旋转金标准", torch.allclose(y1.float(), y2, atol=1e-5))


def t_flash_attn():
    b, h, s, d = 1, 2, 48, 16
    q, k, v = (torch.randn(b, h, s, d, generator=G, dtype=torch.float64)
               for _ in range(3))
    # 全量 fp64 金标准（不做分块）
    for causal in (True, False):
        att = (q @ k.transpose(-1, -2)) / (d ** 0.5)
        if causal:
            mask = torch.ones(s, s, dtype=torch.bool).tril()
            att = att.masked_fill(~mask, float("-inf"))
        gold = torch.softmax(att, dim=-1) @ v
        got = ref.flash_attn_ref(q.half(), k.half(), v.half(),
                                 causal=causal, chunk=16)
        check(f"flash_attn(分块) vs fp64全量 causal={causal}",
              (got.double() - gold).abs().max().item() < 3e-3)


def t_quant_gemm():
    M, K, N = 16, 64, 32
    a = torch.randint(-127, 128, (M, K), generator=G, dtype=torch.int32)
    b = torch.randint(-127, 128, (K, N), generator=G, dtype=torch.int32)
    sa = torch.rand(M, generator=G) * 0.02 + 0.005
    sb = torch.rand(N, generator=G) * 0.02 + 0.005
    # 整数 matmul 精确金标准（int64 无舍入），再乘 scale
    gold = (a.double() @ b.double()) * sa[:, None] * sb[None, :]
    got = ref.quant_gemm_ref(a.to(torch.int8), b.to(torch.int8), sa, sb)
    rel = ((got.double() - gold).abs() / gold.abs().clamp_min(1)).max().item()
    check("quant_gemm vs int64精确matmul", rel < 5e-2, f"(max_rel={rel:.1e})")


if __name__ == "__main__":
    t_rmsnorm(); t_softmax(); t_silu_mul(); t_rope()
    t_flash_attn(); t_quant_gemm()
    print("-" * 60)
    print("全部通过" if not FAILS else f"失败项: {FAILS}")
    raise SystemExit(1 if FAILS else 0)
