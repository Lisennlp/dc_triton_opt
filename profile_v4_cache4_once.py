"""Run one Triton V4 cache4 call for Nsight Compute profiling.

Usage:
  sudo env CUDA_VISIBLE_DEVICES=2 HOME=/tmp B=16 T=4096 N=32 D=128 BM=16 W=112 G=4 \
    /usr/local/cuda-12.2/nsight-compute-2023.2.2/ncu \
      --target-processes all \
      --profile-from-start off \
      --kernel-name regex:dc_onekernel_cache4 \
      --launch-count 1 \
      --metrics l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_local_op_st.sum,smsp__sass_sectors_mem_local.sum \
      /home/lishengping/miniconda3/bin/python profile_v4_cache4_once.py
"""

import math
import os

import torch

from triton_dc_onekernel_v4 import TritonDCOneKernel as V4


def main():
    torch.manual_seed(0)
    B = int(os.environ.get("B", "16"))
    T = int(os.environ.get("T", "4096"))
    N = int(os.environ.get("N", "32"))
    D = int(os.environ.get("D", "128"))
    BM = int(os.environ.get("BM", "16"))
    W = int(os.environ.get("W", "112"))
    G = int(os.environ.get("G", "4"))
    scaling = 1.0 / math.sqrt(D)

    q = torch.randn(B, T, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, T, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, T, N, D, device="cuda", dtype=torch.float16)
    weights = tuple(
        torch.randn(B, T, N, device="cuda", dtype=torch.float16)
        for _ in range(6)
    )
    seq_lens = torch.full((B,), T, device="cuda", dtype=torch.int32)

    # Warm up first so NCU does not profile Triton JIT/autotune launches.
    V4.forward(q, k, v, weights, scaling, W, seq_lens, G=G, chunk_size=BM)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    V4.forward(q, k, v, weights, scaling, W, seq_lens, G=G, chunk_size=BM)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
