"""Correctness and latency check for the V5 HPG=4 experiment.

Run:
  CUDA_VISIBLE_DEVICES=0 python bench_v5_hpg4.py

Environment knobs:
  B=16 T=4096 BM=16 W=112 G=8 WARMUP=10 REPEAT=30 SKIP_FA2=0
"""

from __future__ import annotations

import math
import os
import time

import torch
from flash_attn import flash_attn_func

from dc_attention_torch import dc_attention_window_chunked_residual_grouped
from triton_dc_onekernel_v4 import TritonDCOneKernel as V4
from triton_dc_onekernel_v5_hpg4 import TritonDCOneKernel as V5


device = "cuda"
dtype = torch.float16
N, D = 32, 128
scaling = 1.0 / math.sqrt(D)


def make(B, T):
    q = torch.randn(B, T, N, D, device=device, dtype=dtype)
    k = torch.randn(B, T, N, D, device=device, dtype=dtype)
    v = torch.randn(B, T, N, D, device=device, dtype=dtype)
    weights = tuple(torch.randn(B, T, N, device=device, dtype=dtype) for _ in range(6))
    seq_lens = torch.full((B,), T, device=device, dtype=torch.int32)
    return q, k, v, weights, seq_lens


def bench(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat * 1e6


def diff_stats(out, ref_bntd):
    diff = (out.permute(0, 2, 1, 3).float() - ref_bntd.float()).abs()
    return diff.max().item(), diff.mean().item()


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available; skipping V5 HPG=4 benchmark.")
        return

    torch.manual_seed(0)
    B = int(os.environ.get("B", "16"))
    T = int(os.environ.get("T", "4096"))
    BM = int(os.environ.get("BM", "16"))
    W = int(os.environ.get("W", "112"))
    G = int(os.environ.get("G", "8"))
    warmup = int(os.environ.get("WARMUP", "10"))
    repeat = int(os.environ.get("REPEAT", "30"))
    skip_fa2 = os.environ.get("SKIP_FA2", "0") == "1"
    HPG = N // G
    assert HPG == 4, f"bench_v5_hpg4 expects HPG=4, got G={G}, HPG={HPG}"

    print(f"V5 HPG=4 check: B={B} T={T} N={N} D={D} BM={BM} W={W} G={G}")
    print(f"warmup={warmup} repeat={repeat}")

    qc, kc, vc, wc, slc = make(1, min(T, 512))
    ref = dc_attention_window_chunked_residual_grouped(
        qc, kc, vc, wc, scaling, W, slc, BM, G=G
    )
    out_v4 = V4.forward(qc, kc, vc, wc, scaling, W, slc, G=G, chunk_size=BM)
    variants = [
        ("V5 qh+ah", dict(cache_qk_half=True, a_acc_half=True)),
        ("V5 qh+af", dict(cache_qk_half=True, a_acc_half=False)),
        ("V5 qf+af", dict(cache_qk_half=False, a_acc_half=False)),
    ]
    print("\nCorrectness vs torch grouped ref:")
    m, a = diff_stats(out_v4, ref)
    print(f"  V4       max={m:.4e} mean={a:.4e}")
    for name, opts in variants:
        out = V5.forward(qc, kc, vc, wc, scaling, W, slc, G=G, chunk_size=BM, **opts)
        m, a = diff_stats(out, ref)
        mv4 = (out.float() - out_v4.float()).abs().max().item()
        print(f"  {name:8s} max={m:.4e} mean={a:.4e} max_vs_v4={mv4:.4e}")
    del qc, kc, vc, wc, slc, ref, out_v4
    torch.cuda.empty_cache()

    q, k, v, weights, seq_lens = make(B, T)
    V4.forward(q, k, v, weights, scaling, W, seq_lens, G=G, chunk_size=BM)
    for _, opts in variants:
        V5.forward(q, k, v, weights, scaling, W, seq_lens, G=G, chunk_size=BM, **opts)
    torch.cuda.synchronize()

    us_fa2w = None
    if not skip_fa2:
        us_fa2w = bench(
            lambda: flash_attn_func(q, k, v, softmax_scale=scaling, causal=True, window_size=(W - 1, 0)),
            warmup,
            repeat,
        )
    us_v4 = bench(
        lambda: V4.forward(q, k, v, weights, scaling, W, seq_lens, G=G, chunk_size=BM),
        warmup,
        repeat,
    )
    print("\nLatency:")
    if us_fa2w is None:
        print("  FA2-w     skipped")
        print(f"  V4       {us_v4:8.0f} us")
    else:
        print(f"  FA2-w    {us_fa2w:8.0f} us")
        print(f"  V4       {us_v4:8.0f} us  {us_v4 / us_fa2w:5.2f}x FA2-w")
    for name, opts in variants:
        us = bench(
            lambda opts=opts: V5.forward(q, k, v, weights, scaling, W, seq_lens, G=G, chunk_size=BM, **opts),
            warmup,
            repeat,
        )
        if us_fa2w is None:
            print(f"  {name:8s} {us:8.0f} us  {us / us_v4:5.2f}x V4")
        else:
            print(f"  {name:8s} {us:8.0f} us  {us / us_fa2w:5.2f}x FA2-w  {us / us_v4:5.2f}x V4")


if __name__ == "__main__":
    main()
