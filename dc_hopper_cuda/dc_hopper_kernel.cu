#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>

namespace {

using namespace nvcuda;
namespace cg = cooperative_groups;

constexpr int kNHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kG = 8;
constexpr int kHPG = 4;
constexpr int kWarpSize = 32;
constexpr int kOptBM16 = 16;
constexpr int kOptKL256 = 256;
constexpr int kOptWarps = 8;
constexpr int kOptThreads = kOptWarps * kWarpSize;
constexpr int kClusterDimX = 4;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_HALF(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Half, #x " must be fp16")

__device__ __forceinline__ bool valid_qk(
    int qpos,
    int kpos,
    int T,
    int window) {
    return qpos >= 0 && qpos < T && kpos >= 0 && kpos < T && kpos <= qpos &&
           qpos - kpos < window;
}

__device__ __forceinline__ int qkv_idx(int b, int t, int h, int d, int T) {
    return (((b * T + t) * kNHeads + h) * kHeadDim + d);
}

__device__ __forceinline__ int w_idx(int b, int t, int h, int T) {
    return ((b * T + t) * kNHeads + h);
}

__global__ void cluster_dsm_smoke_kernel(float* __restrict__ out, int num_clusters) {
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = static_cast<int>(cluster.block_rank());
    const int cluster_id = static_cast<int>(blockIdx.x) / kClusterDimX;

    extern __shared__ unsigned char smem_raw[];
    float* smem = reinterpret_cast<float*>(smem_raw);
    if (threadIdx.x == 0) {
        smem[0] = static_cast<float>(rank + 1);
    }
    cluster.sync();

    if (threadIdx.x == 0 && cluster_id < num_clusters) {
        float sum = 0.0f;
        #pragma unroll
        for (int r = 0; r < kClusterDimX; ++r) {
            float* remote = cluster.map_shared_rank(smem, r);
            sum += remote[0];
        }
        out[cluster_id * kClusterDimX + rank] = sum;
    }
}

__global__ void dc_hpg4_wide_cluster_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ pre_w1,
    const half* __restrict__ pre_w2,
    const half* __restrict__ pre_dd,
    const half* __restrict__ post_w1,
    const half* __restrict__ post_w2,
    const half* __restrict__ post_dd,
    half* __restrict__ out,
    int T,
    int num_chunks,
    float scaling,
    int window,
    int BM,
    int KL) {
    cg::cluster_group cluster = cg::this_cluster();
    const int hloc = static_cast<int>(cluster.block_rank());
    const int cluster_id = static_cast<int>(blockIdx.x) / kClusterDimX;
    const int group = cluster_id % kG;
    const int tmp = cluster_id / kG;
    const int m_block = tmp % num_chunks;
    const int b = tmp / num_chunks;
    const int tid = threadIdx.x;
    const int h = group * kHPG + hloc;
    const int q_start = m_block * BM;
    const int k_start = q_start - window + 1;

    extern __shared__ unsigned char smem_raw[];
    half* qk_tile = reinterpret_cast<half*>(smem_raw);     // [BM, KL]
    half* probs = qk_tile + BM * KL;                       // [BM, KL]

    for (int idx = tid; idx < BM * KL; idx += blockDim.x) {
        const int m = idx / KL;
        const int j = idx - m * KL;
        const int qpos = q_start + m;
        const int kpos = k_start + j;
        float acc = 0.0f;
        if (valid_qk(qpos, kpos, T, window)) {
            #pragma unroll
            for (int d = 0; d < kHeadDim; ++d) {
                const float qv = __half2float(q[qkv_idx(b, qpos, h, d, T)]);
                const float kv = __half2float(k[qkv_idx(b, kpos, h, d, T)]);
                acc += qv * kv;
            }
            acc *= scaling;
        }
        qk_tile[idx] = __float2half_rn(acc);
        probs[idx] = __float2half_rn(0.0f);
    }
    cluster.sync();

    for (int m = tid; m < BM; m += blockDim.x) {
        const int qpos = q_start + m;
        float max_score = -INFINITY;

        if (qpos < T) {
            const int wi = w_idx(b, qpos, h, T);
            const float pw2 = __half2float(pre_w2[wi]);
            const float pdd = 1.0f + __half2float(pre_dd[wi]);

            for (int j = 0; j < KL; ++j) {
                const int kpos = k_start + j;
                float score = -INFINITY;
                if (valid_qk(qpos, kpos, T, window)) {
                    float s_acc = 0.0f;
                    #pragma unroll
                    for (int rr = 0; rr < kClusterDimX; ++rr) {
                        half* qk_remote = cluster.map_shared_rank(qk_tile, rr);
                        const int h2 = group * kHPG + rr;
                        const int wi2 = w_idx(b, qpos, h2, T);
                        const float w1 = __half2float(pre_w1[wi2]);
                        const float qk_val = __half2float(qk_remote[m * KL + j]);
                        s_acc += w1 * qk_val;
                    }
                    const float qk_h = __half2float(qk_tile[m * KL + j]);
                    score = pdd * qk_h + pw2 * s_acc;
                }
                probs[m * KL + j] = __float2half_rn(score);
                max_score = fmaxf(max_score, score);
            }

            float denom = 0.0f;
            for (int j = 0; j < KL; ++j) {
                const float score = __half2float(probs[m * KL + j]);
                const float p = isfinite(score) ? expf(score - max_score) : 0.0f;
                probs[m * KL + j] = __float2half_rn(p);
                denom += p;
            }
            const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
            for (int j = 0; j < KL; ++j) {
                const float p = __half2float(probs[m * KL + j]) * inv_denom;
                probs[m * KL + j] = __float2half_rn(p);
            }
        } else {
            for (int j = 0; j < KL; ++j) {
                probs[m * KL + j] = __float2half_rn(0.0f);
            }
        }
    }
    cluster.sync();

    for (int idx = tid; idx < BM * kHeadDim; idx += blockDim.x) {
        const int m = idx / kHeadDim;
        const int d = idx - m * kHeadDim;
        const int qpos = q_start + m;
        if (qpos < T) {
            const int wi = w_idx(b, qpos, h, T);
            const float pdd = 1.0f + __half2float(post_dd[wi]);
            const float pw2 = __half2float(post_w2[wi]);
            float acc = 0.0f;

            for (int j = 0; j < KL; ++j) {
                const int kpos = k_start + j;
                if (kpos >= 0 && kpos < T) {
                    float a_acc = 0.0f;
                    #pragma unroll
                    for (int rr = 0; rr < kClusterDimX; ++rr) {
                        half* p_remote = cluster.map_shared_rank(probs, rr);
                        const int h2 = group * kHPG + rr;
                        const int wi2 = w_idx(b, qpos, h2, T);
                        const float w1 = __half2float(post_w1[wi2]);
                        a_acc += w1 * __half2float(p_remote[m * KL + j]);
                    }
                    const float p = __half2float(probs[m * KL + j]);
                    const float mixed = pdd * p + pw2 * a_acc;
                    const float vv = __half2float(v[qkv_idx(b, kpos, h, d, T)]);
                    acc += mixed * vv;
                }
            }
            out[qkv_idx(b, qpos, h, d, T)] = __float2half_rn(acc);
        }
    }
}

__global__ void dc_hpg4_bm32_ref_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ pre_w1,
    const half* __restrict__ pre_w2,
    const half* __restrict__ pre_dd,
    const half* __restrict__ post_w1,
    const half* __restrict__ post_w2,
    const half* __restrict__ post_dd,
    half* __restrict__ out,
    int T,
    float scaling,
    int window,
    int BM,
    int KL) {
    const int b = blockIdx.x;
    const int m_block = blockIdx.y;
    const int group = blockIdx.z;
    const int tid = threadIdx.x;
    const int q_start = m_block * BM;
    const int k_start = q_start - window + 1;

    extern __shared__ half smem[];
    half* qk_smem = smem;                         // [HPG, BM, KL]
    half* probs = qk_smem + kHPG * BM * KL;        // [HPG, BM, KL]
    half* a_acc = probs + kHPG * BM * KL;          // [BM, KL]

    const int score_elems = kHPG * BM * KL;
    for (int idx = tid; idx < score_elems; idx += blockDim.x) {
        const int hloc = idx / (BM * KL);
        const int rem = idx - hloc * BM * KL;
        const int m = rem / KL;
        const int j = rem - m * KL;
        const int h = group * kHPG + hloc;
        const int qpos = q_start + m;
        const int kpos = k_start + j;

        float acc = 0.0f;
        if (valid_qk(qpos, kpos, T, window)) {
            #pragma unroll
            for (int d = 0; d < kHeadDim; ++d) {
                const float qv = __half2float(q[qkv_idx(b, qpos, h, d, T)]);
                const float kv = __half2float(k[qkv_idx(b, kpos, h, d, T)]);
                acc += qv * kv;
            }
            acc *= scaling;
        }
        qk_smem[idx] = __float2half_rn(acc);
    }
    __syncthreads();

    // Build DC-pre scores and softmax rows. One thread owns one (head, query row).
    for (int row = tid; row < kHPG * BM; row += blockDim.x) {
        const int hloc = row / BM;
        const int m = row - hloc * BM;
        const int h = group * kHPG + hloc;
        const int qpos = q_start + m;
        const int wi = w_idx(b, qpos, h, T);
        float max_score = -INFINITY;

        if (qpos < T) {
            const float pw2 = __half2float(pre_w2[wi]);
            const float pdd = 1.0f + __half2float(pre_dd[wi]);

            for (int j = 0; j < KL; ++j) {
                const int kpos = k_start + j;
                float score = -INFINITY;
                if (valid_qk(qpos, kpos, T, window)) {
                    float s_acc = 0.0f;
                    #pragma unroll
                    for (int hh = 0; hh < kHPG; ++hh) {
                        const int h2 = group * kHPG + hh;
                        const int wi2 = w_idx(b, qpos, h2, T);
                        const float w1 = __half2float(pre_w1[wi2]);
                        const float qk = __half2float(qk_smem[(hh * BM + m) * KL + j]);
                        s_acc += w1 * qk;
                    }
                    const float qk_h = __half2float(qk_smem[(hloc * BM + m) * KL + j]);
                    score = pdd * qk_h + pw2 * s_acc;
                }
                probs[(hloc * BM + m) * KL + j] = __float2half_rn(score);
                max_score = fmaxf(max_score, score);
            }

            float denom = 0.0f;
            for (int j = 0; j < KL; ++j) {
                const float score = __half2float(probs[(hloc * BM + m) * KL + j]);
                const float p = isfinite(score) ? expf(score - max_score) : 0.0f;
                probs[(hloc * BM + m) * KL + j] = __float2half_rn(p);
                denom += p;
            }
            const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
            for (int j = 0; j < KL; ++j) {
                const float p = __half2float(probs[(hloc * BM + m) * KL + j]) * inv_denom;
                probs[(hloc * BM + m) * KL + j] = __float2half_rn(p);
            }
        } else {
            for (int j = 0; j < KL; ++j) {
                probs[(hloc * BM + m) * KL + j] = __float2half_rn(0.0f);
            }
        }
    }
    __syncthreads();

    // Build post a_acc in shared memory.
    for (int idx = tid; idx < BM * KL; idx += blockDim.x) {
        const int m = idx / KL;
        const int j = idx - m * KL;
        const int qpos = q_start + m;
        float acc = 0.0f;
        if (qpos < T) {
            #pragma unroll
            for (int hh = 0; hh < kHPG; ++hh) {
                const int h2 = group * kHPG + hh;
                const int wi2 = w_idx(b, qpos, h2, T);
                const float w1 = __half2float(post_w1[wi2]);
                const float p = __half2float(probs[(hh * BM + m) * KL + j]);
                acc += w1 * p;
            }
        }
        a_acc[idx] = __float2half_rn(acc);
    }
    __syncthreads();

    // Final mixed-probability V dot.
    const int out_elems = kHPG * BM * kHeadDim;
    for (int idx = tid; idx < out_elems; idx += blockDim.x) {
        const int hloc = idx / (BM * kHeadDim);
        const int rem = idx - hloc * BM * kHeadDim;
        const int m = rem / kHeadDim;
        const int d = rem - m * kHeadDim;
        const int h = group * kHPG + hloc;
        const int qpos = q_start + m;

        if (qpos < T) {
            const int wi = w_idx(b, qpos, h, T);
            const float pdd = 1.0f + __half2float(post_dd[wi]);
            const float pw2 = __half2float(post_w2[wi]);
            float acc = 0.0f;
            for (int j = 0; j < KL; ++j) {
                const int kpos = k_start + j;
                if (kpos >= 0 && kpos < T) {
                    const float p = __half2float(probs[(hloc * BM + m) * KL + j]);
                    const float aa = __half2float(a_acc[m * KL + j]);
                    const float mixed = pdd * p + pw2 * aa;
                    const float vv = __half2float(v[qkv_idx(b, kpos, h, d, T)]);
                    acc += mixed * vv;
                }
            }
            out[qkv_idx(b, qpos, h, d, T)] = __float2half_rn(acc);
        }
    }
}

__global__ void dc_hpg4_bm16_w240_wmma_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ pre_w1,
    const half* __restrict__ pre_w2,
    const half* __restrict__ pre_dd,
    const half* __restrict__ post_w1,
    const half* __restrict__ post_w2,
    const half* __restrict__ post_dd,
    half* __restrict__ out,
    int T,
    float scaling) {
    const int b = blockIdx.x;
    const int m_block = blockIdx.y;
    const int group = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid & (kWarpSize - 1);
    const int q_start = m_block * kOptBM16;
    const int window = 240;
    const int k_start = q_start - window + 1;

    extern __shared__ unsigned char smem_raw[];
    half* smem_h = reinterpret_cast<half*>(smem_raw);
    half* q_smem = smem_h;                                                    // [4,16,128]
    half* qk_smem = q_smem + kHPG * kOptBM16 * kHeadDim;                     // [4,16,256]
    half* probs = qk_smem + kHPG * kOptBM16 * kOptKL256;                     // [4,16,256]
    half* a_acc = probs + kHPG * kOptBM16 * kOptKL256;                       // [16,256]
    half* k_warp = a_acc + kOptBM16 * kOptKL256;                             // [8,16,128]
    half* a_tile_warp = k_warp + kOptWarps * 16 * kHeadDim;                  // [8,16,16]
    half* v_tile_warp = a_tile_warp + kOptWarps * 16 * 16;                   // [8,16,16]
    float* out_tile_warp = reinterpret_cast<float*>(v_tile_warp + kOptWarps * 16 * 16); // [8,16,16]

    for (int idx = tid; idx < kHPG * kOptBM16 * kHeadDim; idx += blockDim.x) {
        const int hloc = idx / (kOptBM16 * kHeadDim);
        const int rem = idx - hloc * kOptBM16 * kHeadDim;
        const int m = rem / kHeadDim;
        const int d = rem - m * kHeadDim;
        const int h = group * kHPG + hloc;
        const int qpos = q_start + m;
        q_smem[idx] = qpos < T ? q[qkv_idx(b, qpos, h, d, T)] : __float2half_rn(0.0f);
    }
    __syncthreads();

    // QK: one warp computes one [16,16] tile, iterating over head and KL tiles.
    for (int task = warp_id; task < kHPG * 16; task += kOptWarps) {
        const int hloc = task / 16;
        const int ktile = task - hloc * 16;
        const int h = group * kHPG + hloc;
        half* kbuf = k_warp + warp_id * 16 * kHeadDim;

        for (int idx = lane; idx < 16 * kHeadDim; idx += kWarpSize) {
            const int kk_row = idx / kHeadDim;
            const int d = idx - kk_row * kHeadDim;
            const int kpos = k_start + ktile * 16 + kk_row;
            kbuf[idx] = (kpos >= 0 && kpos < T) ? k[qkv_idx(b, kpos, h, d, T)] : __float2half_rn(0.0f);
        }
        __syncwarp();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        #pragma unroll
        for (int kk = 0; kk < kHeadDim; kk += 16) {
            wmma::load_matrix_sync(a_frag, q_smem + hloc * kOptBM16 * kHeadDim + kk, kHeadDim);
            wmma::load_matrix_sync(b_frag, kbuf + kk, kHeadDim);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        float* otmp = out_tile_warp + warp_id * 16 * 16;
        wmma::store_matrix_sync(otmp, c_frag, 16, wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < 16 * 16; idx += kWarpSize) {
            const int m = idx / 16;
            const int j = idx - m * 16;
            const float val = otmp[idx] * scaling;
            qk_smem[(hloc * kOptBM16 + m) * kOptKL256 + ktile * 16 + j] = __float2half_rn(val);
        }
        __syncwarp();
    }
    __syncthreads();

    // DC-pre + softmax. One thread owns one (head,row).
    for (int row = tid; row < kHPG * kOptBM16; row += blockDim.x) {
        const int hloc = row / kOptBM16;
        const int m = row - hloc * kOptBM16;
        const int h = group * kHPG + hloc;
        const int qpos = q_start + m;
        float max_score = -INFINITY;

        if (qpos < T) {
            const int wi = w_idx(b, qpos, h, T);
            const float pw2 = __half2float(pre_w2[wi]);
            const float pdd = 1.0f + __half2float(pre_dd[wi]);
            for (int j = 0; j < kOptKL256; ++j) {
                const int kpos = k_start + j;
                float score = -INFINITY;
                if (valid_qk(qpos, kpos, T, window)) {
                    float s_acc = 0.0f;
                    #pragma unroll
                    for (int hh = 0; hh < kHPG; ++hh) {
                        const int h2 = group * kHPG + hh;
                        const int wi2 = w_idx(b, qpos, h2, T);
                        const float w1 = __half2float(pre_w1[wi2]);
                        const float qk_val = __half2float(qk_smem[(hh * kOptBM16 + m) * kOptKL256 + j]);
                        s_acc += w1 * qk_val;
                    }
                    const float qk_h = __half2float(qk_smem[(hloc * kOptBM16 + m) * kOptKL256 + j]);
                    score = pdd * qk_h + pw2 * s_acc;
                }
                probs[(hloc * kOptBM16 + m) * kOptKL256 + j] = __float2half_rn(score);
                max_score = fmaxf(max_score, score);
            }

            float denom = 0.0f;
            for (int j = 0; j < kOptKL256; ++j) {
                const float score = __half2float(probs[(hloc * kOptBM16 + m) * kOptKL256 + j]);
                const float p = isfinite(score) ? expf(score - max_score) : 0.0f;
                probs[(hloc * kOptBM16 + m) * kOptKL256 + j] = __float2half_rn(p);
                denom += p;
            }
            const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
            for (int j = 0; j < kOptKL256; ++j) {
                const float p = __half2float(probs[(hloc * kOptBM16 + m) * kOptKL256 + j]) * inv_denom;
                probs[(hloc * kOptBM16 + m) * kOptKL256 + j] = __float2half_rn(p);
            }
        } else {
            for (int j = 0; j < kOptKL256; ++j) {
                probs[(hloc * kOptBM16 + m) * kOptKL256 + j] = __float2half_rn(0.0f);
            }
        }
    }
    __syncthreads();

    for (int idx = tid; idx < kOptBM16 * kOptKL256; idx += blockDim.x) {
        const int m = idx / kOptKL256;
        const int j = idx - m * kOptKL256;
        const int qpos = q_start + m;
        float acc = 0.0f;
        if (qpos < T) {
            #pragma unroll
            for (int hh = 0; hh < kHPG; ++hh) {
                const int h2 = group * kHPG + hh;
                const int wi2 = w_idx(b, qpos, h2, T);
                const float w1 = __half2float(post_w1[wi2]);
                const float p = __half2float(probs[(hh * kOptBM16 + m) * kOptKL256 + j]);
                acc += w1 * p;
            }
        }
        a_acc[idx] = __float2half_rn(acc);
    }
    __syncthreads();

    // Final mixed @ V: one warp computes one [16,16] output tile.
    for (int task = warp_id; task < kHPG * 8; task += kOptWarps) {
        const int hloc = task / 8;
        const int dtile = task - hloc * 8;
        const int h = group * kHPG + hloc;
        half* abuf = a_tile_warp + warp_id * 16 * 16;
        half* vbuf = v_tile_warp + warp_id * 16 * 16;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int ktile = 0; ktile < 16; ++ktile) {
            for (int idx = lane; idx < 16 * 16; idx += kWarpSize) {
                const int m = idx / 16;
                const int j = idx - m * 16;
                const int qpos = q_start + m;
                float mixed = 0.0f;
                if (qpos < T) {
                    const int wi = w_idx(b, qpos, h, T);
                    const float pdd = 1.0f + __half2float(post_dd[wi]);
                    const float pw2 = __half2float(post_w2[wi]);
                    const int kidx = ktile * 16 + j;
                    const float p = __half2float(probs[(hloc * kOptBM16 + m) * kOptKL256 + kidx]);
                    const float aa = __half2float(a_acc[m * kOptKL256 + kidx]);
                    mixed = pdd * p + pw2 * aa;
                }
                abuf[idx] = __float2half_rn(mixed);
            }
            for (int idx = lane; idx < 16 * 16; idx += kWarpSize) {
                const int kk_row = idx / 16;
                const int d = idx - kk_row * 16;
                const int kpos = k_start + ktile * 16 + kk_row;
                const int dpos = dtile * 16 + d;
                vbuf[idx] = (kpos >= 0 && kpos < T) ? v[qkv_idx(b, kpos, h, dpos, T)] : __float2half_rn(0.0f);
            }
            __syncwarp();
            wmma::load_matrix_sync(a_frag, abuf, 16);
            wmma::load_matrix_sync(b_frag, vbuf, 16);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncwarp();
        }

        float* otmp = out_tile_warp + warp_id * 16 * 16;
        wmma::store_matrix_sync(otmp, c_frag, 16, wmma::mem_row_major);
        __syncwarp();
        for (int idx = lane; idx < 16 * 16; idx += kWarpSize) {
            const int m = idx / 16;
            const int d = idx - m * 16;
            const int qpos = q_start + m;
            const int dpos = dtile * 16 + d;
            if (qpos < T) {
                out[qkv_idx(b, qpos, h, dpos, T)] = __float2half_rn(otmp[idx]);
            }
        }
        __syncwarp();
    }
}

void check_fixed_inputs(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& pre_w1,
    const torch::Tensor& pre_w2,
    const torch::Tensor& pre_dd,
    const torch::Tensor& post_w1,
    const torch::Tensor& post_w2,
    const torch::Tensor& post_dd,
    int64_t window,
    int64_t chunk_size) {
    CHECK_CUDA(q);
    CHECK_CUDA(k);
    CHECK_CUDA(v);
    CHECK_CUDA(pre_w1);
    CHECK_CUDA(pre_w2);
    CHECK_CUDA(pre_dd);
    CHECK_CUDA(post_w1);
    CHECK_CUDA(post_w2);
    CHECK_CUDA(post_dd);
    CHECK_CONTIGUOUS(q);
    CHECK_CONTIGUOUS(k);
    CHECK_CONTIGUOUS(v);
    CHECK_CONTIGUOUS(pre_w1);
    CHECK_CONTIGUOUS(pre_w2);
    CHECK_CONTIGUOUS(pre_dd);
    CHECK_CONTIGUOUS(post_w1);
    CHECK_CONTIGUOUS(post_w2);
    CHECK_CONTIGUOUS(post_dd);
    CHECK_HALF(q);
    CHECK_HALF(k);
    CHECK_HALF(v);
    CHECK_HALF(pre_w1);
    CHECK_HALF(pre_w2);
    CHECK_HALF(pre_dd);
    CHECK_HALF(post_w1);
    CHECK_HALF(post_w2);
    CHECK_HALF(post_dd);

    TORCH_CHECK(q.dim() == 4, "q must have shape [B, T, 32, 128]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k must match q shape");
    TORCH_CHECK(v.sizes() == q.sizes(), "v must match q shape");
    TORCH_CHECK(q.size(2) == kNHeads && q.size(3) == kHeadDim, "only N=32,D=128 is supported");
    TORCH_CHECK(
        pre_w1.dim() == 3 && pre_w1.size(0) == q.size(0) && pre_w1.size(1) == q.size(1) &&
            pre_w1.size(2) == q.size(2),
        "pre_w1 must be [B,T,N]");
    TORCH_CHECK(pre_w2.sizes() == pre_w1.sizes(), "pre_w2 must match pre_w1");
    TORCH_CHECK(pre_dd.sizes() == pre_w1.sizes(), "pre_dd must match pre_w1");
    TORCH_CHECK(post_w1.sizes() == pre_w1.sizes(), "post_w1 must match pre_w1");
    TORCH_CHECK(post_w2.sizes() == pre_w1.sizes(), "post_w2 must match pre_w1");
    TORCH_CHECK(post_dd.sizes() == pre_w1.sizes(), "post_dd must match pre_w1");
    TORCH_CHECK(
        (chunk_size == 32 && window == 224) || (chunk_size == 16 && window == 240),
        "wide reference scaffold supports only BM=32,W=224 or BM=16,W=240");
}

}  // namespace

torch::Tensor forward_hpg4_wide_ref(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor pre_w1,
    torch::Tensor pre_w2,
    torch::Tensor pre_dd,
    torch::Tensor post_w1,
    torch::Tensor post_w2,
    torch::Tensor post_dd,
    double scaling,
    int64_t window,
    int64_t chunk_size) {
    check_fixed_inputs(
        q, k, v, pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd, window, chunk_size);

    const int B = static_cast<int>(q.size(0));
    const int T = static_cast<int>(q.size(1));
    const int BM = static_cast<int>(chunk_size);
    const int KL = 256;

    auto out = torch::empty_like(q);
    const int threads = 256;
    dim3 grid(B, (T + BM - 1) / BM, kG);
    const int smem_bytes = (2 * kHPG * BM * KL + BM * KL) * static_cast<int>(sizeof(half));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        dc_hpg4_bm32_ref_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));

    dc_hpg4_bm32_ref_kernel<<<grid, threads, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(pre_w1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(pre_w2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(pre_dd.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(post_w1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(post_w2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(post_dd.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        T,
        static_cast<float>(scaling),
        static_cast<int>(window),
        BM,
        KL);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor forward_hpg4_bm32_ref(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor pre_w1,
    torch::Tensor pre_w2,
    torch::Tensor pre_dd,
    torch::Tensor post_w1,
    torch::Tensor post_w2,
    torch::Tensor post_dd,
    double scaling,
    int64_t window) {
    TORCH_CHECK(window == 224, "BM=32 path is now reserved for the wide W=224 target");
    return forward_hpg4_wide_ref(
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
        window,
        32);
}

torch::Tensor forward_hpg4_wide_opt(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor pre_w1,
    torch::Tensor pre_w2,
    torch::Tensor pre_dd,
    torch::Tensor post_w1,
    torch::Tensor post_w2,
    torch::Tensor post_dd,
    double scaling,
    int64_t window,
    int64_t chunk_size) {
    check_fixed_inputs(
        q, k, v, pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd, window, chunk_size);
    TORCH_CHECK(
        chunk_size == 16 && window == 240,
        "current tensor-core opt path only supports BM=16,W=240");

    const int B = static_cast<int>(q.size(0));
    const int T = static_cast<int>(q.size(1));
    auto out = torch::empty_like(q);

    const int half_elems =
        kHPG * kOptBM16 * kHeadDim +          // q_smem
        kHPG * kOptBM16 * kOptKL256 +         // qk_smem
        kHPG * kOptBM16 * kOptKL256 +         // probs
        kOptBM16 * kOptKL256 +                // a_acc
        kOptWarps * 16 * kHeadDim +           // per-warp K tile
        kOptWarps * 16 * 16 +                 // per-warp A tile
        kOptWarps * 16 * 16;                  // per-warp V tile
    const int smem_bytes = half_elems * static_cast<int>(sizeof(half)) +
                           kOptWarps * 16 * 16 * static_cast<int>(sizeof(float));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        dc_hpg4_bm16_w240_wmma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));

    dim3 grid(B, (T + kOptBM16 - 1) / kOptBM16, kG);
    dc_hpg4_bm16_w240_wmma_kernel<<<
        grid, kOptThreads, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(pre_w1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(pre_w2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(pre_dd.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(post_w1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(post_w2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(post_dd.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        T,
        static_cast<float>(scaling));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor forward_hpg4_wide_cluster(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor pre_w1,
    torch::Tensor pre_w2,
    torch::Tensor pre_dd,
    torch::Tensor post_w1,
    torch::Tensor post_w2,
    torch::Tensor post_dd,
    double scaling,
    int64_t window,
    int64_t chunk_size) {
    check_fixed_inputs(
        q, k, v, pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd, window, chunk_size);

    const int B = static_cast<int>(q.size(0));
    const int T = static_cast<int>(q.size(1));
    const int BM = static_cast<int>(chunk_size);
    const int KL = 256;
    const int num_chunks = (T + BM - 1) / BM;
    const int num_clusters = B * num_chunks * kG;
    auto out = torch::empty_like(q);

    const int threads = 256;
    const int smem_bytes = 2 * BM * KL * static_cast<int>(sizeof(half));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        dc_hpg4_wide_cluster_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        dc_hpg4_wide_cluster_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_bytes));

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(static_cast<unsigned int>(num_clusters * kClusterDimX));
    config.blockDim = dim3(threads);
    config.dynamicSmemBytes = smem_bytes;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = kClusterDimX;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        dc_hpg4_wide_cluster_kernel,
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(pre_w1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(pre_w2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(pre_dd.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(post_w1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(post_w2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(post_dd.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        T,
        num_chunks,
        static_cast<float>(scaling),
        static_cast<int>(window),
        BM,
        KL));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor cluster_dsm_smoke(int64_t num_clusters) {
    TORCH_CHECK(num_clusters > 0, "num_clusters must be positive");
    auto opts = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    auto out = torch::empty({num_clusters, kClusterDimX}, opts);

    C10_CUDA_CHECK(cudaFuncSetAttribute(
        cluster_dsm_smoke_kernel,
        cudaFuncAttributeNonPortableClusterSizeAllowed,
        1));

    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(static_cast<unsigned int>(num_clusters * kClusterDimX));
    config.blockDim = dim3(32);
    config.dynamicSmemBytes = sizeof(float);
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = kClusterDimX;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    config.attrs = attrs;
    config.numAttrs = 1;

    C10_CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        cluster_dsm_smoke_kernel,
        out.data_ptr<float>(),
        static_cast<int>(num_clusters)));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
