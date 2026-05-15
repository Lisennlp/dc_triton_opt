"""Correctness gate for the experimental Hopper CUDA DC extension.

Build first on an H100 machine:
  cd dc_hopper_cuda
  /home/lishengping/miniconda3/bin/python setup.py build_ext --inplace

Run:
  CUDA_VISIBLE_DEVICES=1 /home/lishengping/miniconda3/bin/python test_dc_hopper_cuda.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

from triton_dc_onekernel_v4_h100 import (
    TritonDCOneKernelMixedProbs,
    TritonDCOneKernelMixedProbs256,
)


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "dc_hopper_cuda"))


def make_case(B: int, T: int, dtype: torch.dtype):
    q = torch.randn(B, T, 32, 128, device="cuda", dtype=dtype)
    k = torch.randn(B, T, 32, 128, device="cuda", dtype=dtype)
    v = torch.randn(B, T, 32, 128, device="cuda", dtype=dtype)
    ws = tuple(torch.randn(B, T, 32, device="cuda", dtype=dtype) for _ in range(6))
    return q, k, v, ws


def check(B: int, T: int, W: int):
    import dc_hopper_cuda

    dtype = torch.float16
    scaling = 1.0 / math.sqrt(128)
    q, k, v, ws = make_case(B, T, dtype)
    pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = ws

    if W == 96:
        ref = TritonDCOneKernelMixedProbs.forward(
            q, k, v, ws, scaling, W, None, G=8, chunk_size=32
        )
    elif W == 224:
        ref = TritonDCOneKernelMixedProbs256.forward(
            q, k, v, ws, scaling, W, None, G=8, chunk_size=32
        )
    else:
        raise ValueError(f"unsupported W={W}")

    got = dc_hopper_cuda.forward_hpg4_bm32_ref(
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
    )
    torch.cuda.synchronize()

    diff = (got.float() - ref.float()).abs()
    print(
        f"B={B:2d} T={T:4d} W={W:3d}: "
        f"max={diff.max().item():.4e} mean={diff.mean().item():.4e}"
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(f"this test needs sm90+; current device is sm{major}{minor}")

    for B in (1, 2):
        for T, W in ((128, 96), (256, 224), (512, 96), (512, 224)):
            check(B, T, W)


if __name__ == "__main__":
    main()
