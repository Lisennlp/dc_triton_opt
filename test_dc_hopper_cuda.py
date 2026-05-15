"""Correctness gate for the experimental Hopper CUDA DC extension.

Build first on an H100 machine:
  cd dc_hopper_cuda
  /home/lishengping/miniconda3/bin/python setup.py build_ext --inplace

Run:
  CUDA_VISIBLE_DEVICES=1 /home/lishengping/miniconda3/bin/python test_dc_hopper_cuda.py

Run the first tensor-core experiment:
  CUDA_VISIBLE_DEVICES=1 /home/lishengping/miniconda3/bin/python test_dc_hopper_cuda.py --opt

Run the cluster/DSM diagnostic forward:
  CUDA_VISIBLE_DEVICES=1 /home/lishengping/miniconda3/bin/python test_dc_hopper_cuda.py --cluster

This test intentionally focuses on KL=256 wide-window targets:
  BM=32,W=224 and BM=16,W=240.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

from triton_dc_onekernel_v4_h100 import (
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


def check(B: int, T: int, BM: int, W: int, use_opt: bool = False, use_cluster: bool = False):
    import dc_hopper_cuda

    dtype = torch.float16
    scaling = 1.0 / math.sqrt(128)
    q, k, v, ws = make_case(B, T, dtype)
    pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = ws

    if BM + W != 256:
        raise ValueError(f"unsupported BM={BM}, W={W}")
    ref = TritonDCOneKernelMixedProbs256.forward(
        q, k, v, ws, scaling, W, None, G=8, chunk_size=BM
    )
    torch.cuda.synchronize()

    if use_cluster:
        try:
            got = dc_hopper_cuda.forward_hpg4_wide_cluster(
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
        except Exception as exc:
            print(f"CLUSTER B={B:2d} T={T:4d} BM={BM:2d} W={W:3d}: ERROR {type(exc).__name__}: {exc}")
            return False
        max_limit = 1.2e-1
        mean_limit = 1.0e-3
        label = "CLUSTER"
    elif use_opt:
        if not (BM == 16 and W == 240):
            print(f"B={B:2d} T={T:4d} BM={BM:2d} W={W:3d}: SKIP opt")
            return True
        try:
            got = dc_hopper_cuda.forward_hpg4_wide_opt(
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
        except Exception as exc:
            print(f"OPT B={B:2d} T={T:4d} BM={BM:2d} W={W:3d}: ERROR {type(exc).__name__}: {exc}")
            return False
        max_limit = 2.0e-1
        mean_limit = 2.0e-3
        label = "OPT"
    else:
        got = dc_hopper_cuda.forward_hpg4_wide_ref(
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
        max_limit = 1.2e-1
        mean_limit = 1.0e-3
        label = "REF"
    try:
        torch.cuda.synchronize()
    except Exception as exc:
        print(
            f"{label} B={B:2d} T={T:4d} BM={BM:2d} W={W:3d}: "
            f"SYNC ERROR {type(exc).__name__}: {exc}",
            flush=True,
        )
        raise

    diff = (got.float() - ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    passed = mean_diff <= mean_limit and max_diff <= max_limit
    print(
        f"{label} B={B:2d} T={T:4d} BM={BM:2d} W={W:3d}: "
        f"max={max_diff:.4e} mean={mean_diff:.4e} "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt",
        action="store_true",
        help="test the experimental tensor-core opt path; currently BM=16,W=240 only",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="test the Hopper cluster/DSM diagnostic forward",
    )
    args = parser.parse_args()
    if args.opt and args.cluster:
        raise ValueError("--opt and --cluster are mutually exclusive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(f"this test needs sm90+; current device is sm{major}{minor}")

    all_passed = True
    for B in (1, 2):
        for T, BM, W in ((256, 32, 224), (512, 32, 224), (256, 16, 240), (512, 16, 240)):
            mode = "CLUSTER" if args.cluster else ("OPT" if args.opt else "REF")
            print(f"Running {mode} B={B} T={T} BM={BM} W={W}", flush=True)
            all_passed = check(B, T, BM, W, use_opt=args.opt, use_cluster=args.cluster) and all_passed
    if not all_passed:
        raise RuntimeError("dc_hopper_cuda correctness gate failed")


if __name__ == "__main__":
    main()
