"""Benchmark: v0-v4 single-kernel DC.

Run: CUDA_VISIBLE_DEVICES=2 python bench_onekernel.py
"""

import math
import time
import torch
from flash_attn import flash_attn_func
from triton_dc_onekernel_v0 import TritonDCOneKernel as V0
from triton_dc_onekernel_v1 import TritonDCOneKernel as V1
from triton_dc_onekernel_v2 import TritonDCOneKernel as V2
from triton_dc_onekernel_v3 import TritonDCOneKernel as V3
from triton_dc_onekernel_v4 import TritonDCOneKernel as V4
from triton_dc_residual import TritonDCResidual4K as FK
# from fa2w_head_serial import fa2w_head_serial
from triton_attn_head_serial import AttnHeadSerial as sATTN

device = "cuda"
dtype = torch.float16
N, D = 32, 128
T = 4096
sc = 1.0 / math.sqrt(D)
warmup, repeat = 10, 30


def make(B, T):
    q = torch.randn(B, T, N, D, device=device, dtype=dtype)
    k = torch.randn(B, T, N, D, device=device, dtype=dtype)
    v = torch.randn(B, T, N, D, device=device, dtype=dtype)
    ws = tuple(torch.randn(B, T, N, device=device, dtype=dtype) for _ in range(6))
    sl = torch.full((B,), T, device=device, dtype=torch.int32)
    return q, k, v, ws, sl


def bench(fn):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1e6


def fmt(v):
    return f"{v:7.0f}u" if v < 1e9 else "    FAIL"


def rat(a, b):
    return f"{a / b:6.2f}x" if b > 0 and a < 1e9 else "   N/A"


Bs = [8, 16, 32, 64]
configs = [
    # (BM, W) — all need BM+W=power_of_2; V2 auto num_warps for large KL
    (16, 112),   # BM+W=128
    # (32, 96),    # BM+W=128
    (16, 240),   # BM+W=256
]
Gs = [1, 2, 4, 8]

hdr = (
    f"{'B':>3} {'BM':>3} {'W':>4} {'G':>3} {'HPG':>4} | "
    f"{'V0':>8} {'V1':>8} {'V2':>8} {'V3':>8} {'V4':>8} {'4k':>8} {'sATN':>8} {'FA2w':>8} | "
    f"{'V3/fw':>6} {'V4/fw':>6} {'V4/V3':>6} {'4k/fw':>6} {'sA/fw':>6}"
)
print(hdr)
print("-" * len(hdr))

for B in Bs:
    for BM, W in configs:
        q, k, v, ws, sl = make(B, T)

        us_faw = bench(lambda: flash_attn_func(
            q, k, v, softmax_scale=sc, causal=True, window_size=(W - 1, 0)))

        for G in Gs:
            if N % G != 0:
                continue
            HPG = N // G

            # V0/V1/V2/V3 (need HPG >= 2 and even)
            us_v0 = us_v1 = us_v2 = us_v3 = us_v4 = float("inf")
            if HPG >= 2 and HPG % 2 == 0:
                try:
                    V0.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM)
                    us_v0 = bench(lambda: V0.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM))
                except Exception:
                    pass
                try:
                    V1.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM)
                    us_v1 = bench(lambda: V1.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM))
                except Exception:
                    pass
                try:
                    us_v2 = 0.0
                    # V2.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM)
                    # us_v2 = bench(lambda: V2.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM))
                except Exception:
                    pass

            # V3 (needs HPG >= 2 and even)
            if HPG >= 2 and HPG % 2 == 0:
                try:
                    V3.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM)
                    us_v3 = bench(lambda: V3.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM))
                except Exception:
                    us_v3 = float("inf")
                try:
                    V4.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM)
                    us_v4 = bench(lambda: V4.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM))
                except Exception:
                    us_v4 = float("inf")
            else:
                us_v3 = float("inf")
                us_v4 = float("inf")

            # 4K (4-kernel grouped via reshape, any HPG >= 1)
            try:
                FK.forward(q, k, v, ws, sc, W, sl, G=G)
                us_4k = bench(lambda: FK.forward(q, k, v, ws, sc, W, sl, G=G))
            except Exception:
                us_4k = float("inf")

            # sATTN (head-serial attn, same structure as V3 but no DC mixing)
            try:
                sATTN.forward(q, k, v, sc, W, sl, G=G, chunk_size=BM)
                us_sa = bench(lambda: sATTN.forward(q, k, v, sc, W, sl, G=G, chunk_size=BM))
            except Exception:
                us_sa = float("inf")

            print(
                f"{B:3d} {BM:3d} {W:4d} {G:3d} {HPG:4d} | "
                f"{fmt(us_v0)} {fmt(us_v1)} {fmt(us_v2)} {fmt(us_v3)} "
                f"{fmt(us_v4)} {fmt(us_4k)} {fmt(us_sa)} {fmt(us_faw)} | "
                f"{rat(us_v3, us_faw)} {rat(us_v4, us_faw)} {rat(us_v4, us_v3)} "
                f"{rat(us_4k, us_faw)} {rat(us_sa, us_faw)}"
            )

        del q, k, v, ws, sl
        torch.cuda.empty_cache()

print()
print("V0   = 3-sweep, register s_acc/a_acc")
print("V1   = fused sweep1+2, last pair cached")
print("V2   = V1 + auto num_warps for large KL")
print("V3   = V1 + exp2 softmax + autotune")
print("V4   = optimized Triton cache-four-QK specialization; otherwise V3 fallback")
print("4k   = TritonDCResidual4K (4-kernel, reshape B*G batches)")
print("sATN = head-serial attn (same structure as V3, no DC mixing, with autotune)")
print("FA2w = FlashAttention-2 causal sliding window (all heads parallel)")
