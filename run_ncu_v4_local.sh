#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
BM="${BM:-16}"
W="${W:-112}"
G="${G:-4}"
B="${B:-16}"
T="${T:-4096}"
N="${N:-32}"
D="${D:-128}"
HOME_DIR="${HOME_DIR:-/tmp}"
PYTHON_BIN="${PYTHON_BIN:-/home/lishengping/miniconda3/bin/python}"
NCU_BIN="${NCU_BIN:-/usr/local/cuda-12.2/nsight-compute-2023.2.2/ncu}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton-v4-cache}"

METRICS="${METRICS:-l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum}"

exec sudo env \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  HOME="${HOME_DIR}" \
  TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
  B="${B}" \
  T="${T}" \
  N="${N}" \
  D="${D}" \
  BM="${BM}" \
  W="${W}" \
  G="${G}" \
  "${NCU_BIN}" \
    --target-processes all \
    --profile-from-start off \
    --kernel-name regex:dc_onekernel_cache4 \
    --launch-count 1 \
    --metrics "${METRICS}" \
    "${PYTHON_BIN}" profile_v4_cache4_once.py
