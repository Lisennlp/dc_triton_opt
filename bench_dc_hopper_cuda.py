"""Microbenchmark for experimental dc_hopper_cuda kernels.

Build first:
  cd dc_hopper_cuda
  /home/lishengping/miniconda3/bin/python setup.py build_ext --inplace

Run on H100:
  CUDA_VISIBLE_DEVICES=1 /home/lishengping/miniconda3/bin/python bench_dc_hopper_cuda.py

This is intentionally separate from bench_onekernel_h100.py until correctness
and speed are both acceptable.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch

from triton_dc_onekernel_v4_h100 import TritonDCOneKernelMixedProbs256 as V4HCM256


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "dc_hopper_cuda"))

import dc_hopper_cuda  # noqa: E402


device = "cuda"
dtype = torch.float16
N, D = 32, 128
T = 4096
BM, W = 16, 240
scaling = 1.0 / math.sqrt(D)
warmup, repeat = 5, 20


def make_case(B: int):
    q = torch.randn(B, T, N, D, device=device, dtype=dtype)
    k = torch.randn(B, T, N, D, device=device, dtype=dtype)
    v = torch.randn(B, T, N, D, device=device, dtype=dtype)
    ws = tuple(torch.randn(B, T, N, device=device, dtype=dtype) for _ in range(6))
    return q, k, v, ws


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
    return f"{v:8.0f}u" if v < 1e20 else "    FAIL"


def ratio(a, b):
    return f"{a / b:7.2f}x" if a < 1e20 and b < 1e20 and b > 0 else "    N/A"


def cuda_opt(q, k, v, ws):
    pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = ws
    return dc_hopper_cuda.forward_hpg4_wide_opt(
        q,
        k,
        v,
        pre_w1,
        pre_w2,
        pre_dd,
        post_w1,
        post_w2,
        post_dd,
        scaling,
        W,
        BM,
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(f"this benchmark needs sm90+; current device is sm{major}{minor}")

    hdr = f"{'B':>3} {'BM':>3} {'W':>4} | {'V4HCM256':>10} {'CUDAopt':>10} | {'opt/V4':>8}"
    print(hdr)
    print("-" * len(hdr))
    for B in (8, 16, 32, 64):
        q, k, v, ws = make_case(B)
        try:
            V4HCM256.forward(q, k, v, ws, scaling, W, None, G=8, chunk_size=BM)
            us_v4 = bench(lambda: V4HCM256.forward(q, k, v, ws, scaling, W, None, G=8, chunk_size=BM))
        except Exception:
            us_v4 = float("inf")
        try:
            cuda_opt(q, k, v, ws)
            us_opt = bench(lambda: cuda_opt(q, k, v, ws))
        except Exception:
            us_opt = float("inf")
        print(f"{B:3d} {BM:3d} {W:4d} | {fmt(us_v4)} {fmt(us_opt)} | {ratio(us_opt, us_v4)}")
        del q, k, v, ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
