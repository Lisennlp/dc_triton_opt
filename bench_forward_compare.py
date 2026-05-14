"""Benchmark: TritonDCResidual.forward vs Torch DC residual vs torch.compile vs FA2.

Run on GPU 2:
    CUDA_VISIBLE_DEVICES=2 python bench_forward_compare.py
"""

import math
import time

import torch
import torch._dynamo
from flash_attn import flash_attn_func

from dc_attention_torch import dc_attention_window_chunked_residual
from triton_dc_residual import TritonDCResidual

torch._dynamo.config.cache_size_limit = 64

device = "cuda"
dtype = torch.float16
W = 256
CHUNK = W
warmup, repeat = 10, 30
TORCH_TOKEN_LIMIT = 8192*50

N, D = 32, 128
sc = 1.0 / math.sqrt(D)

configs = [
    # (1, 256),
    # (1, 512),
    # (1, 1024),
    # (1, 2048),
    (1, 4096),
    (4, 4096),
    (16, 4096),
    # (32, 2048),
    (32, 4096),
    # (64, 2048),
    (64, 4096),
    (1, 16384),
    (1, 16384*2),
    (1, 16384*4),
    (1, 16384*6),
    (1, 16384*8),
    (1, 16384*16),

]


def make(B, T, N=N, D=D):
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
        return "    skip"
    if us >= 1e18:
        return "     OOM"
    return f"{us:8.0f}"


def fmt_ratio(us, ref):
    if us is None or ref is None:
        return "   skip"
    if us >= 1e18 or ref >= 1e18:
        return "    OOM"
    return f"{us / ref:6.2f}x"


dc_compiled = torch.compile(dc_attention_window_chunked_residual)

print("=" * 100)
print("Forward-only benchmark: TritonDCResidual vs Torch vs torch.compile vs FA2")
print(f"N={N}, D={D}, window={W}, chunk={CHUNK}, dtype={dtype}")
print("=" * 100)
print()

hdr = (
    f"{'B':>3s} {'T':>5s} {'B*T':>7s} | "
    f"{'Triton':>8s} {'Torch':>8s} {'Compile':>8s} | "
    f"{'FA2-w':>8s} {'FA2-f':>8s} | "
    f"{'Tri/FA2w':>9s} {'Tor/FA2w':>9s} {'Cmp/FA2w':>9s} {'Tri/FA2f':>9s}"
)
print(hdr)
print("-" * len(hdr))

for B, T in configs:
    tokens = B * T
    q, k, v, dc_weights = make(B, T, N, D)
    sl = torch.full((B,), T, device=device, dtype=torch.int32)

    # --- TritonDCResidual ---
    try:
        bufs = TritonDCResidual.alloc_buffers(q, W)
        TritonDCResidual.forward(q, k, v, dc_weights, sc, W, sl, buffers=bufs)
        us_triton = bench(
            lambda: TritonDCResidual.forward(q, k, v, dc_weights, sc, W, sl, buffers=bufs)
        )
    except Exception as exc:
        print(f"  [Triton failed] B={B} T={T}: {type(exc).__name__}: {exc}")
        us_triton = float("inf")
        torch.cuda.empty_cache()

    # --- Torch DC residual ---
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

    # --- torch.compile DC residual ---
    if tokens <= TORCH_TOKEN_LIMIT:
        try:
            dc_compiled(q, k, v, dc_weights, sc, W, sl, CHUNK)
            us_compile = bench(
                lambda: dc_compiled(q, k, v, dc_weights, sc, W, sl, CHUNK),
                warmup_iters=5,
                repeat_iters=10,
            )
        except Exception:
            us_compile = float("inf")
            torch.cuda.empty_cache()
    else:
        us_compile = None

    # --- FA2 sliding window ---
    us_faw = bench(
        lambda: flash_attn_func(
            q, k, v, softmax_scale=sc, causal=True, window_size=(W - 1, 0)
        )
    )

    # --- FA2 full causal ---
    us_faf = bench(lambda: flash_attn_func(q, k, v, softmax_scale=sc, causal=True))

    print(
        f"{B:3d} {T:5d} {tokens:7d} | "
        f"{fmt_us(us_triton)} {fmt_us(us_torch)} {fmt_us(us_compile)} | "
        f"{fmt_us(us_faw)} {fmt_us(us_faf)} | "
        f"{fmt_ratio(us_triton, us_faw)} {fmt_ratio(us_torch, us_faw)} "
        f"{fmt_ratio(us_compile, us_faw)} {fmt_ratio(us_triton, us_faf)}"
    )

    del q, k, v, dc_weights, sl
    if 'bufs' in dir():
        del bufs
    torch.cuda.empty_cache()

print()
print("Legend:")
print("  Triton  = TritonDCResidual.forward (multi-kernel triton)")
print("  Torch   = dc_attention_window_chunked_residual (pure torch)")
print("  Compile = torch.compile(dc_attention_window_chunked_residual)")
print(f"  FA2-w   = FlashAttention-2 causal + sliding window (window={W})")
print("  FA2-f   = FlashAttention-2 full causal")
print(f"  Torch/Compile skipped when B*T > {TORCH_TOKEN_LIMIT}")
print("  All times in microseconds (us)")
