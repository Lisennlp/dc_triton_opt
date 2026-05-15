"""Benchmark: single-kernel DC variants.

Run: CUDA_VISIBLE_DEVICES=1 python bench_onekernel_h100.py
"""

import math
import time
import torch
# from flash_attn import flash_attn_func
from triton_dc_onekernel_v3 import TritonDCOneKernel as V3
from triton_dc_onekernel_v4 import TritonDCOneKernel as V4
from triton_dc_onekernel_v4_h100 import TritonDCOneKernel as V4H
from triton_dc_onekernel_v4_h100 import TritonDCOneKernelCombined as V4HC
from triton_dc_onekernel_v4_h100 import TritonDCOneKernelCombinedProbs as V4HCP
from triton_dc_onekernel_v4_h100 import TritonDCOneKernelMixedProbs as V4HCM
from triton_dc_onekernel_v4_h100 import TritonDCOneKernelMixedProbs256 as V4HCM256

device = "cuda"
dtype = torch.float16
N, D = 32, 128
T = 4096
sc = 1.0 / math.sqrt(D)
warmup, repeat = 10, 30
_fa3_interface = None
_fa3_cu_seqlens = {}
_fa3_error = None
_fa2_error = None
_variant_errors = {}


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
    return f"{a / b:6.2f}x" if b > 0 and a < 1e9 and b < 1e9 else "   N/A"


def fa2_window_attention(q, k, v, softmax_scale, window):
    return flash_attn_func(
        q, k, v, softmax_scale=softmax_scale, causal=True, window_size=(window - 1, 0)
    )


def fa3_supported(device):
    global _fa3_error
    device_idx = device.index
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_idx)
    if major < 9:
        _fa3_error = (
            f"requires sm90+ Hopper/Blackwell; current device is sm{major}{minor} "
            f"({torch.cuda.get_device_name(device_idx)})"
        )
        return False
    return True

# h100这里必须这么写，不要改
def get_fa3_interface():
    global _fa3_interface, _fa3_error
    if _fa3_interface is not None:
        return _fa3_interface
    if _fa3_error is not None:
        raise RuntimeError(_fa3_error)
    try:
        from kernels import get_kernel

        mod = get_kernel("kernels-community/flash-attn3")
        _fa3_interface = getattr(mod, "flash_attn_interface", mod)
        return _fa3_interface
    except Exception as exc:
        _fa3_error = f"{type(exc).__name__}: {exc}"
        raise RuntimeError(_fa3_error) from exc


def fa3_cu_seqlens(B, T, device):
    device_idx = device.index
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    key = (device_idx, B, T)
    cu = _fa3_cu_seqlens.get(key)
    if cu is None:
        cu = torch.arange(0, (B + 1) * T, T, device=device, dtype=torch.int32)
        _fa3_cu_seqlens[key] = cu
    return cu


def fa3_window_attention(q, k, v, softmax_scale, window):
    B, T, H, D = q.shape
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise RuntimeError("FA3 hub kernel only supports BF16 in this environment")

    cu = fa3_cu_seqlens(B, T, q.device)
    out = get_fa3_interface().flash_attn_varlen_func(
        q.reshape(B * T, H, D),
        k.reshape(B * T, H, D),
        v.reshape(B * T, H, D),
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=T,
        max_seqlen_k=T,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=(window - 1, 0),
    )
    if isinstance(out, tuple):
        out = out[0]
    return out.view(B, T, H, D)


Bs = [8, 16, 32, 64]
configs = [
    # (BM, W) — small-window H100 trial, both use KL=128.
    (16, 112),   # preferred setting: smaller register tile.
    (32, 96),    # larger M, more pressure; included to check H100 behavior.
    # Wider-window H100 trial, both use KL=256.
    (16, 240),   # preferred large-W setting: BM*KL matches (32, 96).
    (32, 224),   # higher register pressure; included as an upper-pressure probe.
]
Gs = [8]  # fixed target for the HPG=4 H100 branches

hdr = (
    f"{'B':>3} {'BM':>3} {'W':>4} {'G':>3} {'HPG':>4} | "
    f"{'V3':>8} {'V4':>8} {'V4H':>8} {'V4HC':>8} {'V4HCP':>8} {'V4HCM':>8} {'V4HCM256':>9} {'FA2w':>8} {'FA3w':>8} | "
    f"{'V4/fa3':>7} {'V4H/fa3':>8} {'V4HC/fa3':>9} {'V4HCP/fa3':>10} {'V4HCM/fa3':>10} {'V4HCM256/fa3':>13} "
    f"{'V4H/V4':>7} {'V4HC/V4':>8} {'V4HCP/V4':>9} {'V4HCM/V4':>9} {'V4HCM256/V4':>12}"
)
print(hdr)
print("-" * len(hdr))

for B in Bs:
    for BM, W in configs:
        q, k, v, ws, sl = make(B, T)

        # us_fa2w = bench(lambda: fa2_window_attention(q, k, v, sc, W))
        us_fa2w = float("inf")
        if us_fa2w < 10.0:
            _fa2_error = f"measured {us_fa2w:.2f} us for B={B}, W={W}; ignored as implausible"
            us_fa2w = float("inf")
        q3 = k3 = v3_fa3 = None
        try:
            if _fa3_error is None and fa3_supported(q.device):
                q3 = q.to(torch.bfloat16)
                k3 = k.to(torch.bfloat16)
                v3_fa3 = v.to(torch.bfloat16)
                us_fa3w = bench(lambda: fa3_window_attention(q3, k3, v3_fa3, sc, W))
            else:
                us_fa3w = float("inf")
        except Exception as exc:
            us_fa3w = float("inf")
            if _fa3_error is None:
                _fa3_error = f"{type(exc).__name__}: {exc}"
        finally:
            del q3, k3, v3_fa3

        for G in Gs:
            if N % G != 0:
                continue
            HPG = N // G

            # V3/V4/V4H family (needs HPG >= 2 and even)
            use_128_special = BM + W == 128
            use_256_special = BM + W == 256
            us_v3 = us_v4 = us_v4h = us_v4hc = us_v4hcp = us_v4hcm = us_v4hcm256 = float("inf")
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
                try:
                    V4H.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM)
                    us_v4h = bench(lambda: V4H.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM))
                except Exception:
                    us_v4h = float("inf")
                if use_128_special:
                    try:
                        V4HC.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM)
                        us_v4hc = bench(lambda: V4HC.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM))
                    except Exception:
                        us_v4hc = float("inf")
                    try:
                        V4HCP.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM)
                        us_v4hcp = bench(lambda: V4HCP.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM))
                    except Exception:
                        us_v4hcp = float("inf")
                    try:
                        V4HCM.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM)
                        us_v4hcm = bench(lambda: V4HCM.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM))
                    except Exception:
                        us_v4hcm = float("inf")
                if use_256_special:
                    try:
                        V4HCM256.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM)
                        us_v4hcm256 = bench(lambda: V4HCM256.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM))
                    except Exception:
                        us_v4hcm256 = float("inf")
            else:
                us_v3 = float("inf")
                us_v4 = float("inf")
                us_v4h = float("inf")
                us_v4hc = float("inf")
                us_v4hcp = float("inf")
                us_v4hcm = float("inf")
                us_v4hcm256 = float("inf")

            print(
                f"{B:3d} {BM:3d} {W:4d} {G:3d} {HPG:4d} | "
                f"{fmt(us_v3)} {fmt(us_v4)} {fmt(us_v4h)} {fmt(us_v4hc)} {fmt(us_v4hcp)} {fmt(us_v4hcm)} {fmt(us_v4hcm256)} "
                f"{fmt(us_fa2w)} {fmt(us_fa3w)} | "
                f"{rat(us_v4, us_fa3w)} {rat(us_v4h, us_fa3w)} {rat(us_v4hc, us_fa3w)} {rat(us_v4hcp, us_fa3w)} {rat(us_v4hcm, us_fa3w)} {rat(us_v4hcm256, us_fa3w)} "
                f"{rat(us_v4h, us_v4)} {rat(us_v4hc, us_v4)} {rat(us_v4hcp, us_v4)} {rat(us_v4hcm, us_v4)} {rat(us_v4hcm256, us_v4)}"
            )

        del q, k, v, ws, sl
        torch.cuda.empty_cache()

print()
print("V3   = exp2 softmax + autotune baseline")
print("V4   = optimized Triton cache-four-QK specialization; otherwise V3 fallback")
print("V4H  = H100-oriented V4 experiment; HPG=4/W+BM=128 has a fixed small-window branch")
print("V4HC = V4H HPG=4/W+BM=128 combined-output experiment: build a_acc first, then PV+AV with one OUT store")
print("V4HCP= V4HC + cached fp16 probs: avoids the second softmax pass at extra register pressure")
print("V4HCM= V4HCP + mixed attention weights: ((post_dd+1)*probs + post_w2*a_acc) @ V, one final V dot")
print("V4HCM256= V4HCM idea specialized for BM+W=256 / KL=256; benchmarked only on the wide-window rows")
print("FA2w = FlashAttention-2 causal sliding window, FP16 inputs")
print("FA3w = FlashAttention-3 varlen causal sliding window, BF16 inputs; BF16 casts are outside timing")
if _fa2_error is not None:
    print(f"FA2w unavailable/error: {_fa2_error}")
if _fa3_error is not None:
    print(f"FA3w unavailable/error: {_fa3_error}")
for name, err in _variant_errors.items():
    print(f"{name} unavailable/error: {err}")
