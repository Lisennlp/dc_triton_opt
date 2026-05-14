"""Full DC residual benchmark vs torch and FlashAttention-2.

Run:
    CUDA_VISIBLE_DEVICES=2 python bench_atg_vs_opt.py
"""

import math
import time

import torch
import torch._dynamo
from flash_attn import flash_attn_func

from dc_attention_torch import dc_attention_window_chunked_residual
from triton_dc_residual import TritonDCResidual
from triton_dc_fused import TritonDCResidualFused


torch._dynamo.config.cache_size_limit = 64

device = "cuda"
dtype = torch.float16
W = 256
CHUNK = W
warmup, repeat = 10, 30
TORCH_TOKEN_LIMIT = 4096


def make(B, T, N=32, D=128):
    q = torch.randn(B, T, N, D, device=device, dtype=dtype)
    k = torch.randn(B, T, N, D, device=device, dtype=dtype)
    v = torch.randn(B, T, N, D, device=device, dtype=dtype)
    pre_w1 = torch.randn(B, T, N, device=device, dtype=dtype)
    pre_w2 = torch.randn(B, T, N, device=device, dtype=dtype)
    pre_dd = torch.randn(B, T, N, device=device, dtype=dtype)
    post_w1 = torch.randn(B, T, N, device=device, dtype=dtype)
    post_w2 = torch.randn(B, T, N, device=device, dtype=dtype)
    post_dd = torch.randn(B, T, N, device=device, dtype=dtype)
    return q, k, v, (pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd)


def bench(fn, warmup_iters=warmup, repeat_iters=repeat):
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat_iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat_iters * 1e6


def fmt_us(us):
    if us is None:
        return "   skip"
    if us >= 1e18:
        return "    OOM"
    return f"{us:7.0f}u"


def fmt_ratio(us, ref):
    if us is None:
        return "   skip"
    if us >= 1e18:
        return "    OOM"
    return f"{us / ref:6.2f}x"


N, D = 32, 128
sc = 1.0 / math.sqrt(D)

configs = [
    (1, 256),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (16, 4096),
    (32, 2048),
    (32, 4096),
    (64, 2048),
    (64, 4096),
]


print("=" * 60)
print("Correctness check (fused vs torch)")
print("=" * 60)
for B_c, T_c in [(1, 256), (1, 512), (2, 1024)]:
    q_c, k_c, v_c, dc_c = make(B_c, T_c, N, D)
    sl_c = torch.full((B_c,), T_c, device=device, dtype=torch.int32)
    ref = dc_attention_window_chunked_residual(q_c, k_c, v_c, dc_c, sc, W, sl_c, CHUNK)
    # ref is [B,N,T,D] (from torch.cat on dim=2)
    fused_out = TritonDCResidualFused.forward(q_c, k_c, v_c, dc_c, sc, W, sl_c)
    # fused_out is [B,T,N,D], transpose to [B,N,T,D] for comparison
    fused_bntd = fused_out.permute(0, 2, 1, 3).contiguous()
    diff = (fused_bntd.float() - ref.float()).abs()
    rdiff = diff / (ref.float().abs() + 1e-6)
    print(f"  B={B_c} T={T_c}: max_abs={diff.max().item():.4e}  max_rel={rdiff.max().item():.4e}  "
          f"mean_abs={diff.mean().item():.4e}")
    del q_c, k_c, v_c, dc_c, sl_c, ref, fused_bntd, fused_out
    torch.cuda.empty_cache()
print()

hdr = (
    f"{'B':>3s} {'T':>5s} {'B*T':>7s} | "
    f"{'Fused':>8s} {'Opt':>8s} {'Torch':>8s} | "
    f"{'FA2-w':>8s} {'FA2-f':>8s} | "
    f"{'Fsd/FA2w':>9s} {'Opt/FA2w':>9s} {'Fsd/FA2f':>9s}"
)
print(hdr)
print("-" * len(hdr))

for B, T in configs:
    tokens = B * T
    q, k, v, dc_weights = make(B, T, N, D)
    sl = torch.full((B,), T, device=device, dtype=torch.int32)

    try:
        fused_out = TritonDCResidualFused.forward(q, k, v, dc_weights, sc, W, sl)
        us_fused = bench(
            lambda: TritonDCResidualFused.forward(q, k, v, dc_weights, sc, W, sl)
        )
    except Exception as exc:
        print(f"[Fused failed] B={B} T={T}: {type(exc).__name__}: {exc}")
        us_fused = float("inf")
        torch.cuda.empty_cache()

    try:
        opt_buffers = TritonDCResidual.alloc_buffers(q, W)
        TritonDCResidual.forward(q, k, v, dc_weights, sc, W, sl, buffers=opt_buffers)
        us_opt = bench(
            lambda: TritonDCResidual.forward(
                q, k, v, dc_weights, sc, W, sl, buffers=opt_buffers
            )
        )
    except Exception as exc:
        print(f"[Opt failed] B={B} T={T}: {type(exc).__name__}: {exc}")
        us_opt = float("inf")
        torch.cuda.empty_cache()

    if tokens <= TORCH_TOKEN_LIMIT:
        try:
            dc_attention_window_chunked_residual(q, k, v, dc_weights, sc, W, sl, CHUNK)
            us_torch = bench(
                lambda: dc_attention_window_chunked_residual(
                    q, k, v, dc_weights, sc, W, sl, CHUNK
                ),
                warmup_iters=5,
                repeat_iters=10,
            )
        except Exception:
            us_torch = float("inf")
            torch.cuda.empty_cache()
    else:
        us_torch = None

    us_faw = bench(
        lambda: flash_attn_func(
            q, k, v, softmax_scale=sc, causal=True, window_size=(W - 1, 0)
        )
    )
    us_faf = bench(lambda: flash_attn_func(q, k, v, softmax_scale=sc, causal=True))

    print(
        f"{B:3d} {T:5d} {tokens:7d} | "
        f"{fmt_us(us_fused)} {fmt_us(us_opt)} {fmt_us(us_torch)} | "
        f"{fmt_us(us_faw)} {fmt_us(us_faf)} | "
        f"{fmt_ratio(us_fused, us_faw)} {fmt_ratio(us_opt, us_faw)} {fmt_ratio(us_fused, us_faf)}"
    )

    del q, k, v, dc_weights, sl
    torch.cuda.empty_cache()

print()
print("Fused = TritonDCResidualFused single-kernel fused DC")
print("Opt   = TritonDCResidual multi-kernel DC (previous best)")
print(f"Torch = dc_attention_window_chunked_residual, skipped above {TORCH_TOKEN_LIMIT} tokens")
print(f"FA2-w = FlashAttention-2 causal sliding window, window_size={W}")
print("FA2-f = FlashAttention-2 full causal")
