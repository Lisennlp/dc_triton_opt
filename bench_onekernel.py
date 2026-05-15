"""A800 / FlashAttention-2 benchmark for one-kernel DC variants.

Run:
  CUDA_VISIBLE_DEVICES=2 python bench_onekernel.py

Environment knobs:
  B_LIST=8,16,32,64
  CONFIGS=16:112,32:96,16:240,32:224
  G_LIST=8
  WARMUP=10
  REPEAT=30
  RUN_COMPONENTS=0   # set 1 to also benchmark Pre0/Post0/Post1
"""

from __future__ import annotations

import math
import os
import time

import torch
from flash_attn import flash_attn_func

from triton_dc_onekernel_v3 import TritonDCOneKernel as V3
from triton_dc_onekernel_v4 import TritonDCOneKernel as V4
from triton_dc_onekernel_v5 import TritonDCOneKernel as V5
from triton_dc_onekernel_v4_h100 import TritonDCOneKernel as V4H
from triton_dc_onekernel_v4_h100 import TritonDCOneKernelCombined as V4HC
from triton_dc_onekernel_v4_h100 import TritonDCOneKernelCombinedProbs as V4HCP
from triton_dc_onekernel_v4_h100 import TritonDCOneKernelMixedProbs as V4HCM
from triton_dc_onekernel_v4_h100 import TritonDCOneKernelMixedProbs256 as V4HCM256
from triton_dc_onekernel_v4_h100 import TritonDCOneKernelMixedProbs256Narrow as V4HCM256N
from triton_dc_onekernel_Prev0 import TritonDCOneKernel as PreV0
from triton_dc_onekernel_Postv0 import TritonDCOneKernel as PostV0
from triton_dc_onekernel_Postv1 import TritonDCOneKernel as PostV1


device = "cuda"
dtype = torch.float16
N, D = 32, 128
T = int(os.environ.get("T", "4096"))
sc = 1.0 / math.sqrt(D)
warmup = int(os.environ.get("WARMUP", "10"))
repeat = int(os.environ.get("REPEAT", "30"))
run_components = os.environ.get("RUN_COMPONENTS", "0") == "1"

_variant_errors: dict[str, str] = {}
_fa2_error: str | None = None


def parse_int_list(name: str, default: str) -> list[int]:
    raw = os.environ.get(name, default)
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError(f"{name} produced an empty list")
    return vals


def parse_configs(default: str) -> list[tuple[int, int]]:
    raw = os.environ.get("CONFIGS", default)
    configs: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        bm_s, w_s = item.replace("x", ":").split(":")
        configs.append((int(bm_s), int(w_s)))
    if not configs:
        raise ValueError("CONFIGS produced an empty list")
    return configs


def next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def make(B: int):
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


def fmt_us(v: float) -> str:
    return f"{v:8.0f}u" if v < 1e9 else "    FAIL"


def fmt_ratio(a: float, b: float) -> str:
    return f"{a / b:6.2f}x" if a < 1e9 and b < 1e9 and b > 0 else "   N/A"


def fa2_window_attention(q, k, v, window: int):
    return flash_attn_func(
        q,
        k,
        v,
        softmax_scale=sc,
        causal=True,
        window_size=(window - 1, 0),
    )


def supports_v4(G: int, HPG: int, KL: int) -> bool:
    return G <= 8 and 4 <= HPG <= 8 and KL <= 128


def supports_v5(G: int, HPG: int, KL: int) -> bool:
    return G <= 8 and 4 <= HPG <= 8 and KL <= 128


def supports_v4h(G: int, HPG: int, KL: int) -> bool:
    return G <= 8 and 4 <= HPG <= 8 and KL <= 256


def supports_hpg4_128(BM: int, W: int, G: int, HPG: int, KL: int) -> bool:
    return G <= 8 and HPG == 4 and KL == 128 and BM + W == 128


def supports_hpg4_256(BM: int, W: int, G: int, HPG: int, KL: int) -> bool:
    return G <= 8 and HPG == 4 and KL == 256 and BM + W == 256


def run_variant(name: str, fn, enabled: bool) -> float:
    if not enabled:
        return float("inf")
    try:
        fn()
        return bench(fn)
    except Exception as exc:
        _variant_errors.setdefault(name, f"{type(exc).__name__}: {exc}")
        return float("inf")


def full_dc_variants(q, k, v, ws, sl, BM: int, W: int, G: int, HPG: int, KL: int):
    return [
        (
            "V3",
            True,
            lambda: V3.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM),
        ),
        (
            "V4",
            supports_v4(G, HPG, KL),
            lambda: V4.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM),
        ),
        (
            "V5",
            supports_v5(G, HPG, KL),
            lambda: V5.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM),
        ),
        (
            "V4H",
            supports_v4h(G, HPG, KL),
            lambda: V4H.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM),
        ),
        (
            "V4HC",
            supports_hpg4_128(BM, W, G, HPG, KL),
            lambda: V4HC.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM),
        ),
        (
            "V4HCP",
            supports_hpg4_128(BM, W, G, HPG, KL),
            lambda: V4HCP.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM),
        ),
        (
            "V4HCM",
            supports_hpg4_128(BM, W, G, HPG, KL),
            lambda: V4HCM.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM),
        ),
        (
            "V4HCM256",
            supports_hpg4_256(BM, W, G, HPG, KL),
            lambda: V4HCM256.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM),
        ),
        (
            "V4HCM256N",
            supports_hpg4_256(BM, W, G, HPG, KL),
            lambda: V4HCM256N.forward(q, k, v, ws, sc, W, None, G=G, chunk_size=BM),
        ),
    ]


def component_variants(q, k, v, ws, sl, BM: int, W: int, G: int):
    pre_ws = ws[:3]
    post_ws = ws[3:]
    return [
        (
            "Pre0",
            True,
            lambda: PreV0.forward(q, k, v, pre_ws, sc, W, sl, G=G, chunk_size=BM),
        ),
        (
            "Post0",
            True,
            lambda: PostV0.forward(q, k, v, post_ws, sc, W, sl, G=G, chunk_size=BM),
        ),
        (
            "Post1",
            True,
            lambda: PostV1.forward(q, k, v, post_ws, sc, W, sl, G=G, chunk_size=BM),
        ),
    ]


Bs = parse_int_list("B_LIST", "8,16,32,64")
configs = parse_configs("16:112,32:96,16:240,32:224")
Gs = parse_int_list("G_LIST", "8")

full_variant_names = ["V3", "V4", "V5", "V4H", "V4HC", "V4HCP", "V4HCM", "V4HCM256", "V4HCM256N"]
variant_names = list(full_variant_names)
ratio_names = ["V3", "V4", "V5", "V4H", "V4HCM", "V4HCM256", "V4HCM256N"]
if run_components:
    variant_names += ["Pre0", "Post0", "Post1"]
    ratio_names += ["Pre0", "Post1"]

hdr = (
    f"{'B':>3} {'BM':>3} {'W':>4} {'G':>3} {'HPG':>4} {'KL':>4} | "
    f"{'FA2w':>8} "
    + " ".join(f"{name:>9}" for name in variant_names)
    + " | "
    + " ".join(f"{name + '/fw':>10}" for name in ratio_names)
    + f" {'Best':>9} {'Best/fw':>8}"
)
print(hdr)
print("-" * len(hdr))

for B in Bs:
    for BM, W in configs:
        q, k, v, ws, sl = make(B)
        try:
            us_fa2w = bench(lambda: fa2_window_attention(q, k, v, W))
            if us_fa2w < 10.0:
                _fa2_error = f"measured {us_fa2w:.2f} us for B={B}, BM={BM}, W={W}"
                us_fa2w = float("inf")
        except Exception as exc:
            _fa2_error = f"{type(exc).__name__}: {exc}"
            us_fa2w = float("inf")

        for G in Gs:
            if N % G != 0:
                continue
            HPG = N // G
            if HPG < 2 or HPG % 2 != 0:
                continue

            KL = next_power_of_2(BM + W - 1)
            timings: dict[str, float] = {}

            for name, enabled, fn in full_dc_variants(q, k, v, ws, sl, BM, W, G, HPG, KL):
                timings[name] = run_variant(name, fn, enabled)

            if run_components:
                for name, enabled, fn in component_variants(q, k, v, ws, sl, BM, W, G):
                    timings[name] = run_variant(name, fn, enabled)

            valid_full = [(name, timings[name]) for name in full_variant_names if timings.get(name, float("inf")) < 1e9]
            if valid_full:
                best_name, best_us = min(valid_full, key=lambda item: item[1])
            else:
                best_name, best_us = "N/A", float("inf")

            print(
                f"{B:3d} {BM:3d} {W:4d} {G:3d} {HPG:4d} {KL:4d} | "
                f"{fmt_us(us_fa2w)} "
                + " ".join(fmt_us(timings.get(name, float("inf"))) for name in variant_names)
                + " | "
                + " ".join(fmt_ratio(timings.get(name, float("inf")), us_fa2w) for name in ratio_names)
                + f" {best_name:>9} {fmt_ratio(best_us, us_fa2w)}"
            )

        del q, k, v, ws, sl
        torch.cuda.empty_cache()

print()
print("FA2w     = FlashAttention-2 causal sliding window, FP16 inputs")
print("V3       = exp2 softmax + autotune baseline")
print("V4       = cache-four-QK specialization, only shown when it does not fall back to V3")
print("V5       = A800-oriented V5 cache4/cache8 path, only shown when it does not fall back")
print("V4H      = H100-oriented V4 experiment; measured on A800 for portability comparison")
print("V4HC     = BM+W=128 combined-output path, one OUT store")
print("V4HCP    = V4HC + cached fp16 probabilities")
print("V4HCM    = V4HCP + mixed final attention weights, one final V dot")
print("V4HCM256 = V4HCM idea for BM+W=256 / KL=256")
print("V4HCM256N= V4HCM256 with fp16 s_acc/a_acc to reduce KL=256 register pressure")
if run_components:
    print("Pre0/Post0/Post1 are component experiments, not full DC replacements.")
if _fa2_error is not None:
    print(f"FA2w unavailable/error: {_fa2_error}")
for name, err in _variant_errors.items():
    print(f"{name} unavailable/error: {err}")
