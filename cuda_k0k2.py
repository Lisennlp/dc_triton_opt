"""CUDA kernels for K0 (q_agg) and K2 (output mix) — replaces Triton versions.

Key advantages over Triton:
  - Higher occupancy: grid (T, B) with D threads/block vs Triton's (T/BM, B) with 4 warps
  - Lower launch overhead: ~5us vs ~15us per Triton kernel
  - Shared memory broadcast for pw2/pw1 scalars
"""

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ═══════════════════════════════════════════════════════════
// K0: q_agg[B, T, D] = Σ_n pw2_pre[B, T, n] * Q[B, T, n, D]
//
// Grid: (T, B), Block: (D).  Each thread computes one (t, d) element.
// pw2 is broadcast via shared memory.
// ═══════════════════════════════════════════════════════════

__global__ void k0_qagg_kernel(
    const __half* __restrict__ Q,         // [B, T, N, D]
    const __half* __restrict__ pw2_pre,   // [B, T, N]
    __half* __restrict__ q_agg,           // [B, T, D]
    int T, int N, int D)
{
    const int b = blockIdx.y;
    const int t = blockIdx.x;
    const int d = threadIdx.x;
    if (t >= T || d >= D) return;

    extern __shared__ float s_pw2[];      // [N]
    if (d < N) {
        s_pw2[d] = __half2float(pw2_pre[b * T * N + t * N + d]);
    }
    __syncthreads();

    float acc = 0.0f;
    const int base = b * T * N * D + t * N * D;
    #pragma unroll 8
    for (int n = 0; n < N; n++) {
        acc += s_pw2[n] * __half2float(Q[base + n * D + d]);
    }
    q_agg[b * T * D + t * D + d] = __float2half(acc);
}


// ═══════════════════════════════════════════════════════════
// K2: output-space rank-1 post-mix (fused reduce + broadcast)
//
//   out_agg[b,t,d] = Σ_n pw2_post[n] * attn_out[b,t,n,d]
//   out[b,t,n,d]   = pw1_post[n] * out_agg[b,t,d]
//
// Grid: (T, B), Block: (D).
// ═══════════════════════════════════════════════════════════

__global__ void k2_outmix_kernel(
    const __half* __restrict__ attn_out,  // [B, T, N, D]
    const __half* __restrict__ pw2_post,  // [B, T, N]
    const __half* __restrict__ pw1_post,  // [B, T, N]
    __half* __restrict__ out,             // [B, T, N, D]
    int T, int N, int D)
{
    const int b = blockIdx.y;
    const int t = blockIdx.x;
    const int d = threadIdx.x;
    if (t >= T || d >= D) return;

    extern __shared__ float s_pw[];       // [2*N]: pw2 then pw1
    if (d < N) {
        const int idx = b * T * N + t * N + d;
        s_pw[d]     = __half2float(pw2_post[idx]);
        s_pw[N + d] = __half2float(pw1_post[idx]);
    }
    __syncthreads();

    // Reduction: out_agg = Σ_n pw2[n] * attn_out[n]
    float agg = 0.0f;
    const int base_in = b * T * N * D + t * N * D;
    #pragma unroll 8
    for (int n = 0; n < N; n++) {
        agg += s_pw[n] * __half2float(attn_out[base_in + n * D + d]);
    }

    // Broadcast: out[n] = pw1[n] * agg
    const int base_out = b * T * N * D + t * N * D;
    #pragma unroll 8
    for (int n = 0; n < N; n++) {
        out[base_out + n * D + d] = __float2half(s_pw[N + n] * agg);
    }
}


// ═══════════════════════════════════════════════════════════
// C++ wrappers
// ═══════════════════════════════════════════════════════════

torch::Tensor cuda_k0_qagg(torch::Tensor Q, torch::Tensor pw2_pre) {
    const int B = Q.size(0), T = Q.size(1), N = Q.size(2), D = Q.size(3);
    auto q_agg = torch::empty({B, T, D}, Q.options());

    dim3 grid(T, B);
    dim3 block(D);
    int smem = N * sizeof(float);

    k0_qagg_kernel<<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(pw2_pre.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(q_agg.data_ptr<at::Half>()),
        T, N, D);
    return q_agg;
}

void cuda_k0_qagg_inplace(torch::Tensor Q, torch::Tensor pw2_pre, torch::Tensor q_agg) {
    const int B = Q.size(0), T = Q.size(1), N = Q.size(2), D = Q.size(3);
    dim3 grid(T, B);
    dim3 block(D);
    int smem = N * sizeof(float);
    k0_qagg_kernel<<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(Q.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(pw2_pre.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(q_agg.data_ptr<at::Half>()),
        T, N, D);
}

torch::Tensor cuda_k2_outmix(
    torch::Tensor attn_out, torch::Tensor pw2_post, torch::Tensor pw1_post)
{
    const int B = attn_out.size(0), T = attn_out.size(1);
    const int N = attn_out.size(2), D = attn_out.size(3);
    auto out = torch::empty_like(attn_out);

    dim3 grid(T, B);
    dim3 block(D);
    int smem = 2 * N * sizeof(float);

    k2_outmix_kernel<<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(pw2_post.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(pw1_post.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        T, N, D);
    return out;
}

void cuda_k2_outmix_inplace(
    torch::Tensor attn_out, torch::Tensor pw2_post, torch::Tensor pw1_post,
    torch::Tensor out)
{
    const int B = attn_out.size(0), T = attn_out.size(1);
    const int N = attn_out.size(2), D = attn_out.size(3);
    dim3 grid(T, B);
    dim3 block(D);
    int smem = 2 * N * sizeof(float);
    k2_outmix_kernel<<<grid, block, smem>>>(
        reinterpret_cast<const __half*>(attn_out.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(pw2_post.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(pw1_post.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        T, N, D);
}
"""

CPP_SRC = r"""
torch::Tensor cuda_k0_qagg(torch::Tensor Q, torch::Tensor pw2_pre);
void cuda_k0_qagg_inplace(torch::Tensor Q, torch::Tensor pw2_pre, torch::Tensor q_agg);
torch::Tensor cuda_k2_outmix(
    torch::Tensor attn_out, torch::Tensor pw2_post, torch::Tensor pw1_post);
void cuda_k2_outmix_inplace(
    torch::Tensor attn_out, torch::Tensor pw2_post, torch::Tensor pw1_post,
    torch::Tensor out);
"""

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="dc_cuda_k0k2",
            cpp_sources=[CPP_SRC],
            cuda_sources=[CUDA_SRC],
            functions=[
                "cuda_k0_qagg", "cuda_k0_qagg_inplace",
                "cuda_k2_outmix", "cuda_k2_outmix_inplace",
            ],
            verbose=False,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    return _module


def cuda_k0_qagg(Q: torch.Tensor, pw2_pre: torch.Tensor) -> torch.Tensor:
    """q_agg[B,T,D] = Σ_n pw2_pre[n] * Q[B,T,n,D].  CUDA kernel."""
    return _get_module().cuda_k0_qagg(Q.contiguous(), pw2_pre.contiguous())


def cuda_k0_qagg_inplace(
    Q: torch.Tensor, pw2_pre: torch.Tensor, q_agg: torch.Tensor
) -> None:
    """In-place version — writes into pre-allocated q_agg."""
    _get_module().cuda_k0_qagg_inplace(Q.contiguous(), pw2_pre.contiguous(), q_agg)


def cuda_k2_outmix(
    attn_out: torch.Tensor,
    pw2_post: torch.Tensor,
    pw1_post: torch.Tensor,
) -> torch.Tensor:
    """Fused reduce+broadcast: out_agg = Σ pw2*attn; out = pw1*out_agg.  CUDA kernel."""
    return _get_module().cuda_k2_outmix(
        attn_out.contiguous(), pw2_post.contiguous(), pw1_post.contiguous())


def cuda_k2_outmix_inplace(
    attn_out: torch.Tensor,
    pw2_post: torch.Tensor,
    pw1_post: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """In-place version — writes into pre-allocated out."""
    _get_module().cuda_k2_outmix_inplace(
        attn_out.contiguous(), pw2_post.contiguous(), pw1_post.contiguous(), out)
