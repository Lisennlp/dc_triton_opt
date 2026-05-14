/*
 * Persistent-kernel fused DC residual attention.
 *
 * Grid: (B, 1, 1)   — one block per batch element.
 * Block: N warps × 32 threads = N × 32 threads (each warp owns one head).
 *
 * For each chunk [q_start, q_end):
 *   Phase 1 — pre-agg:
 *     Each warp computes its head's QK tile and accumulates pw1*QK.
 *     Inter-warp reduction via shared memory → s_buf in smem.
 *     __syncthreads()
 *
 *   Phase 2 — per-head softmax + PV:
 *     Each warp: recompute QK, build score from s_buf, online softmax + PV.
 *     Also accumulate post_w1 * probs into shared a_buf.
 *     __syncthreads()
 *
 *   Phase 3 — final:
 *     Each warp: a_buf @ V for its head, combine with direct PV, write output.
 *
 * All cross-head communication is done via shared memory + __syncthreads(),
 * no atomic_add, no HBM intermediate buffers for m/l/s/a.
 */

// This file provides the design spec.
// Actual implementation requires a PyTorch CUDA extension with:
//   - Custom __global__ kernel
//   - Shared memory allocation for s_buf[BM][BK] + a_buf[BM][BK]
//   - Warp-level matrix multiply via wmma or mma.sync for QK and PV
//   - pybind11 binding

// Placeholder: the implementation will follow.
// For now, see triton_dc_fused.py for the best Triton version.
