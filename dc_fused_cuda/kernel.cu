/*
 * DC Residual Attention Forward — Multi-Kernel CUDA (A800 / sm_80)
 *
 * Uses nvcuda::wmma (Tensor Core) for QK^T matmul: [BM,D] @ [BK,D]^T -> [BM,BK]
 *
 * Kernel structure mirrors Triton:
 *   K0: Pre-aggregation  s_buf = sum_n pw1_n * QK_n      grid(tiles, B)
 *   K12: Fused softmax-stats + probs + PV + post-agg     grid(tiles, B*N)
 *   K3:  Final output                                     grid(tiles_f, B*N)
 *
 * WMMA: m16n16k16 half -> float accumulator.
 *   QK[BM, BK] = Q[BM, D] @ K[BK, D]^T
 *   BM=16, BK=64 => 4 output [16x16] tiles across BK dimension.
 *   D=128 => 8 accumulation steps (D/16=8).
 *   One warp computes one [16x16] output tile.
 *   Block has 4 warps => covers all 4 tiles => full [16x64] QK block.
 *
 * For PV: P[BM, BK] @ V[BK, D] -> O[BM, D]
 *   Similar: BK=64 => 4 accumulation steps; D=128 => 8 output [16x16] tiles.
 *   Each warp cycles through output tiles.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <float.h>

using namespace nvcuda;

#define HEAD_DIM  128
#define NUM_HEADS 32
#define WINDOW    256
#define BM        16    // query tile
#define BK        64    // key tile
#define WARP_SIZE 32
#define NWARPS    4     // BK / 16
#define BLOCK_THREADS (NWARPS * WARP_SIZE)  // 128

// WMMA dimensions
#define WM 16
#define WN 16
#define WK 16


// ======================================================================
// Shared memory helpers
// ======================================================================

// Load Q[BM, D] from global to shared, row-major
__device__ __forceinline__ void load_q_tile(
    const half* __restrict__ Q_base, long long stride_t, long long stride_n, int n,
    half* __restrict__ q_smem, int q_start, int q_end, int seq_len,
    int tid)
{
    // q_smem layout: [BM][HEAD_DIM], row-major
    const int elems = BM * HEAD_DIM;
    for (int i = tid; i < elems; i += BLOCK_THREADS) {
        int m = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int mg = q_start + m;
        if (mg < q_end && mg < seq_len) {
            q_smem[m * HEAD_DIM + d] = Q_base[mg * stride_t + n * stride_n + d];
        } else {
            q_smem[m * HEAD_DIM + d] = __float2half(0.0f);
        }
    }
}

// Load K[BK, D] from global to shared, row-major
__device__ __forceinline__ void load_k_tile(
    const half* __restrict__ K_base, long long stride_t, long long stride_n, int n,
    half* __restrict__ k_smem, int kb, int k_tile_len, int seq_len,
    int tid)
{
    const int elems = BK * HEAD_DIM;
    for (int i = tid; i < elems; i += BLOCK_THREADS) {
        int j = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int sg = kb + j;
        if (j < k_tile_len && sg < seq_len) {
            k_smem[j * HEAD_DIM + d] = K_base[sg * stride_t + n * stride_n + d];
        } else {
            k_smem[j * HEAD_DIM + d] = __float2half(0.0f);
        }
    }
}

// Load V[BK, D] from global to shared
__device__ __forceinline__ void load_v_tile(
    const half* __restrict__ V_base, long long stride_t, long long stride_n, int n,
    half* __restrict__ v_smem, int kb, int k_tile_len, int seq_len,
    int tid)
{
    const int elems = BK * HEAD_DIM;
    for (int i = tid; i < elems; i += BLOCK_THREADS) {
        int j = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int sg = kb + j;
        if (j < k_tile_len && sg < seq_len) {
            v_smem[j * HEAD_DIM + d] = V_base[sg * stride_t + n * stride_n + d];
        } else {
            v_smem[j * HEAD_DIM + d] = __float2half(0.0f);
        }
    }
}


// ======================================================================
// WMMA QK^T: compute QK[BM, BK] = Q[BM, D] @ K[BK, D]^T
//
// q_smem: [BM][D] row-major (A matrix)
// k_smem: [BK][D] row-major => K^T is [D][BK] col-major
//       => For wmma B-matrix (col-major), we load K as col-major [D][BK]
//       which is the same memory as K row-major [BK][D] transposed.
//
// Each warp computes one [16x16] output tile.
// warp_id selects which 16 columns of BK (since BK=64, 4 warps = 4 tiles).
// Accumulate over D/16 = 8 steps.
//
// Result stored in qk_smem[BM][BK] as float.
// ======================================================================
__device__ __forceinline__ void wmma_qk(
    const half* __restrict__ q_smem,    // [BM][D], row-major
    const half* __restrict__ k_smem,    // [BK][D], row-major
    float*      __restrict__ qk_smem,   // [BM][BK], row-major output
    int warp_id)
{
    // This warp handles output columns [warp_id*16, warp_id*16+16)
    const int col_offset = warp_id * WN;  // 0, 16, 32, 48

    wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    // Accumulate over D/WK = 8 steps
    #pragma unroll
    for (int kk = 0; kk < HEAD_DIM; kk += WK) {
        // A = Q_smem[0:BM, kk:kk+WK], row-major, ldm = HEAD_DIM
        wmma::load_matrix_sync(a_frag, q_smem + kk, HEAD_DIM);

        // B = K_smem[col_offset:col_offset+WN, kk:kk+WK]^T
        //   = col-major view of K[col_offset..., kk...], ldm = HEAD_DIM
        // K is stored row-major [BK][D], element (j,d) at k_smem[j*D+d].
        // For col-major B fragment [WK x WN], we want:
        //   B(d-kk, j-col) = K(col+j, kk+d) = k_smem[(col+j)*D + kk+d]
        // In memory: k_smem + col_offset*D + kk, with ldm = D (stride between columns = D)
        wmma::load_matrix_sync(b_frag, k_smem + col_offset * HEAD_DIM + kk, HEAD_DIM);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store to shared memory: qk_smem[0:BM, col_offset:col_offset+WN]
    wmma::store_matrix_sync(qk_smem + col_offset, c_frag, BK, wmma::mem_row_major);
}


// ======================================================================
// K0: Pre-aggregation
// Grid: (num_tiles, B), Block: 128 (4 warps)
//
// For each k-tile, loops over N heads:
//   1. Load K[BK, D] to smem
//   2. Load Q[BM, D] to smem (head n)
//   3. WMMA: QK = Q @ K^T => qk_smem[BM][BK]
//   4. Accumulate: s_acc[BM][BK] += pw1[m, n] * QK[m, k] * scaling
// After all heads, store s_acc -> s_buf.
// ======================================================================
__global__ void k0_preagg_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ pre_w1,
    half*       __restrict__ s_buf,
    const int*  __restrict__ seq_lens,
    float scaling, int T)
{
    const int tile_idx = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int q_start = tile_idx * BM;
    const int seq_len = seq_lens[b];

    if (q_start >= T) return;
    const int q_end = min(q_start + BM, T);

    const long long sb = (long long)T * NUM_HEADS * HEAD_DIM;
    const long long st = (long long)NUM_HEADS * HEAD_DIM;
    const long long sn = (long long)HEAD_DIM;
    const long long sb_w = (long long)T * NUM_HEADS;
    const long long st_w = (long long)NUM_HEADS;
    const long long sb_s = (long long)T * WINDOW;
    const long long st_s = (long long)WINDOW;

    const half* Q_b = Q + b * sb;
    const half* K_b = K + b * sb;
    const half* pw1_b = pre_w1 + b * sb_w;
    half* s_b = s_buf + b * sb_s;

    // Shared memory layout:
    //   q_smem: [BM * HEAD_DIM] half = 16*128*2 = 4096 bytes
    //   k_smem: [BK * HEAD_DIM] half = 64*128*2 = 16384 bytes
    //   qk_smem: [BM * BK] float = 16*64*4 = 4096 bytes
    //   s_acc: [BM * BK] float = 4096 bytes
    //   Total: 28672 bytes (~28 KB)
    extern __shared__ char smem[];
    half*  q_smem  = (half*)smem;                                 // 4096 B
    half*  k_smem  = (half*)(smem + BM * HEAD_DIM * sizeof(half)); // 16384 B
    float* qk_smem = (float*)(smem + (BM + BK) * HEAD_DIM * sizeof(half)); // 4096 B
    float* s_acc   = qk_smem + BM * BK;                          // 4096 B

    int q_k_lo[BM];
    #pragma unroll
    for (int m = 0; m < BM; m++)
        q_k_lo[m] = max(0, q_start + m - WINDOW + 1);

    int k_lo = max(0, q_start - WINDOW + 1);
    int k_hi = min(q_end, seq_len);
    if (k_hi <= 0) return;

    for (int kb = k_lo; kb < k_hi; kb += BK) {
        int k_tile_len = min(kb + BK, k_hi) - kb;

        // Zero s_acc
        for (int i = tid; i < BM * BK; i += BLOCK_THREADS)
            s_acc[i] = 0.0f;
        __syncthreads();

        for (int n = 0; n < NUM_HEADS; n++) {
            // Load Q tile for head n
            load_q_tile(Q_b, st, sn, n, q_smem, q_start, q_end, seq_len, tid);
            // Load K tile for head n
            load_k_tile(K_b, st, sn, n, k_smem, kb, k_tile_len, seq_len, tid);
            __syncthreads();

            // WMMA QK^T
            wmma_qk(q_smem, k_smem, qk_smem, warp_id);
            __syncthreads();

            // Accumulate pw1 * qk * scaling into s_acc
            for (int i = tid; i < BM * k_tile_len; i += BLOCK_THREADS) {
                int m = i / k_tile_len;
                int j = i % k_tile_len;
                int mg = q_start + m;
                int sg = kb + j;
                if (mg >= q_end || mg >= seq_len || sg > mg || mg - sg >= WINDOW) continue;

                float w1 = __half2float(pw1_b[mg * st_w + n]);
                s_acc[m * BK + j] += w1 * qk_smem[m * BK + j] * scaling;
            }
            __syncthreads();
        }

        // Store s_acc to s_buf
        for (int i = tid; i < BM * k_tile_len; i += BLOCK_THREADS) {
            int m = i / k_tile_len;
            int j = i % k_tile_len;
            int mg = q_start + m;
            int sg = kb + j;
            if (mg >= q_end || mg >= seq_len || sg > mg || mg - sg >= WINDOW) continue;

            int compact_s = sg - q_k_lo[m];
            s_b[mg * st_s + compact_s] = __float2half(s_acc[m * BK + j]);
        }
        __syncthreads();
    }
}


// ======================================================================
// K12 Fused: softmax-stats + probs + PV + post-agg
// Grid: (num_tiles, B*N), Block: 128
//
// Pass 1 (softmax stats): For each k-tile, compute QK via WMMA,
//   then score = (1+dd)*qk*sc + w2*s_buf, update online softmax m_i, l_i.
//
// Pass 2 (PV + post-agg): recompute QK and score, normalize to get probs,
//   accumulate P@V and a_buf.
//
// For P@V: we store prob tile as half in smem, then use WMMA:
//   O[BM, D] += P[BM, BK] @ V[BK, D]
//   This requires iterating over D in chunks of WN=16.
//   Each warp handles WN=16 columns of output.
//   4 warps => 64 columns. D=128 => 2 rounds.
// ======================================================================
__global__ void k12_fused_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ s_buf_in,
    const half* __restrict__ pre_w2,
    const half* __restrict__ pre_dd,
    const half* __restrict__ post_w1,
    half*       __restrict__ o_buf,
    float*      __restrict__ a_buf_out,
    float*      __restrict__ m_buf_out,
    float*      __restrict__ l_buf_out,
    const int*  __restrict__ seq_lens,
    float scaling, int T)
{
    const int tile_idx = blockIdx.x;
    const int bn = blockIdx.y;
    const int b = bn / NUM_HEADS;
    const int n = bn % NUM_HEADS;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int q_start = tile_idx * BM;
    const int seq_len = seq_lens[b];

    if (q_start >= T) return;
    const int q_end = min(q_start + BM, T);

    const long long sb = (long long)T * NUM_HEADS * HEAD_DIM;
    const long long st = (long long)NUM_HEADS * HEAD_DIM;
    const long long sn_d = (long long)HEAD_DIM;
    const long long sb_w = (long long)T * NUM_HEADS;
    const long long st_w = (long long)NUM_HEADS;
    const long long sb_s = (long long)T * WINDOW;
    const long long st_s = (long long)WINDOW;

    const half* Q_b = Q + b * sb;
    const half* K_b = K + b * sb;
    const half* V_b = V + b * sb;
    const half* s_b = s_buf_in + b * sb_s;

    int q_k_lo[BM];
    #pragma unroll
    for (int m = 0; m < BM; m++)
        q_k_lo[m] = max(0, q_start + m - WINDOW + 1);

    int k_lo = max(0, q_start - WINDOW + 1);
    int k_hi = min(q_end, seq_len);
    if (k_hi <= 0) return;

    // Load per-head weights
    float w2[BM], dd1[BM], po1[BM];
    for (int m = 0; m < BM; m++) {
        int mg = q_start + m;
        if (mg < q_end && mg < seq_len) {
            w2[m]  = __half2float(pre_w2[b * sb_w + mg * st_w + n]);
            dd1[m] = __half2float(pre_dd[b * sb_w + mg * st_w + n]) + 1.0f;
            po1[m] = __half2float(post_w1[b * sb_w + mg * st_w + n]);
        } else {
            w2[m] = 0.f; dd1[m] = 1.f; po1[m] = 0.f;
        }
    }

    // Shared memory:
    //  q_smem: BM*D*2        = 4096
    //  k_smem: BK*D*2        = 16384
    //  qk_smem: BM*BK*4      = 4096
    //  p_smem: BM*BK*2       = 2048  (half probs for WMMA PV)
    //  v_smem: BK*D*2         = 16384  (only in pass 2, reuse k_smem space)
    //  total pass1: q+k+qk = 24576
    //  total pass2: q+k(QKrecomp)+qk+p+v = but we can reuse...
    //
    //  Simpler approach: In pass2, do QK recompute and PV in same k-tile loop:
    //   Load Q (once before loop), K tile, compute QK, form P (half), load V tile, WMMA PV.
    //   smem: q(4096) + kv_double(16384) + qk(4096) + p(2048) = 26624 bytes.
    //   Actually we need K and V simultaneously for the same tile:
    //   q(4096) + k(16384) + v(16384) + qk(4096) + p(2048) = 42880 bytes. Close to 48KB limit.
    //   That's tight. Let's recompute differently:
    //   Load K first, WMMA QK -> qk_smem, form probs -> p_smem (half).
    //   Then load V (overwrite k_smem), WMMA PV.
    //   smem: q(4096) + kv_shared(16384) + qk(4096) + p(2048) = 26624 bytes. Good!

    extern __shared__ char smem[];
    half*  q_smem  = (half*)smem;                                          // 4096
    half*  kv_smem = (half*)(smem + BM * HEAD_DIM * sizeof(half));          // 16384
    float* qk_smem = (float*)(smem + (BM + BK) * HEAD_DIM * sizeof(half)); // 4096
    half*  p_smem  = (half*)(qk_smem + BM * BK);                           // 2048

    // ===== Load Q once =====
    load_q_tile(Q_b, st, sn_d, n, q_smem, q_start, q_end, seq_len, tid);
    __syncthreads();

    // ===== Pass 1: Online softmax stats =====
    // Per-thread: each thread "owns" certain (m,k) positions for score computation.
    // But softmax is per-row (per query position m).
    // Strategy: distribute query rows across warps. Each warp handles BM/NWARPS=4 rows.
    // Within each row, the warp lanes do nothing for softmax (it's scalar per row).
    // Only lane 0 of each warp updates m_i, l_i.
    // But we still need QK via WMMA which uses all lanes.
    // Solution: WMMA QK uses all warps, then each warp reads its rows from qk_smem.

    float m_i[BM], l_i[BM];
    #pragma unroll
    for (int m = 0; m < BM; m++) {
        m_i[m] = -FLT_MAX;
        l_i[m] = 0.0f;
    }

    for (int kb = k_lo; kb < k_hi; kb += BK) {
        int k_tile_len = min(kb + BK, k_hi) - kb;

        // Load K tile
        load_k_tile(K_b, st, sn_d, n, kv_smem, kb, k_tile_len, seq_len, tid);
        __syncthreads();

        // WMMA QK^T -> qk_smem[BM][BK]
        wmma_qk(q_smem, kv_smem, qk_smem, warp_id);
        __syncthreads();

        // Update online softmax (all threads read qk_smem but only update their rows)
        // Distribute rows across threads: each thread handles BM*k_tile_len/BLOCK_THREADS elements
        // But softmax is sequential per row. Let each thread handle a subset of rows.
        // With BM=16, 128 threads: 8 threads per row. But softmax within row is sequential over k.
        // Better: just let thread tid handle row tid%BM if tid < BM.
        if (tid < BM) {
            int m = tid;
            int mg = q_start + m;
            if (mg < q_end && mg < seq_len) {
                for (int j = 0; j < k_tile_len; j++) {
                    int sg = kb + j;
                    if (sg > mg || mg - sg >= WINDOW) continue;
                    int compact_s = sg - q_k_lo[m];
                    float qk_val = qk_smem[m * BK + j] * scaling;
                    float s_val = __half2float(s_b[mg * st_s + compact_s]);
                    float score = dd1[m] * qk_val + w2[m] * s_val;
                    float m_new = fmaxf(m_i[m], score);
                    l_i[m] = l_i[m] * expf(m_i[m] - m_new) + expf(score - m_new);
                    m_i[m] = m_new;
                }
            }
        }
        __syncthreads();
    }

    // Broadcast m_i, l_i from thread m to all threads via shared memory
    // Reuse qk_smem for this (BM*2 floats = small)
    float* ml_smem = qk_smem; // reuse
    if (tid < BM) {
        ml_smem[tid] = m_i[tid];
        ml_smem[BM + tid] = l_i[tid];
    }
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < BM; m++) {
        m_i[m] = ml_smem[m];
        l_i[m] = ml_smem[BM + m];
    }
    __syncthreads();

    // Store m_buf, l_buf
    if (tid < BM) {
        int mg = q_start + tid;
        if (mg < q_end && mg < seq_len) {
            m_buf_out[b * (long long)(T * NUM_HEADS) + mg * NUM_HEADS + n] = m_i[tid];
            l_buf_out[b * (long long)(T * NUM_HEADS) + mg * NUM_HEADS + n] = l_i[tid];
        }
    }

    // ===== Pass 2: Probs, PV, post-agg =====
    // Remap shared memory for pass 2:
    //   q_smem: BM*D*2 = 4096 (kept from pass 1)
    //   o_smem: BM*D*4 = 8192 (output accumulator, float)
    //   kv_smem: BK*D*2 = 16384 (reused for K then V)
    //   qk_smem: BM*BK*4 = 4096
    //   p_smem: BM*BK*2 = 2048 (prob half for WMMA PV)
    //   Total: 34816 bytes
    float* o_smem  = (float*)(smem + BM * HEAD_DIM * sizeof(half));
    half*  kv_smem2 = (half*)((char*)o_smem + BM * HEAD_DIM * sizeof(float));
    float* qk_smem2 = (float*)((char*)kv_smem2 + BK * HEAD_DIM * sizeof(half));
    half*  p_smem2  = (half*)(qk_smem2 + BM * BK);

    for (int i = tid; i < BM * HEAD_DIM; i += BLOCK_THREADS)
        o_smem[i] = 0.0f;
    __syncthreads();

    for (int kb = k_lo; kb < k_hi; kb += BK) {
        int k_tile_len = min(kb + BK, k_hi) - kb;

        // Load K, WMMA QK, form prob tile
        load_k_tile(K_b, st, sn_d, n, kv_smem2, kb, k_tile_len, seq_len, tid);
        __syncthreads();
        wmma_qk(q_smem, kv_smem2, qk_smem2, warp_id);
        __syncthreads();

        // Compute normalized probs -> p_smem2 (half), post-agg -> global a_buf
        for (int i = tid; i < BM * BK; i += BLOCK_THREADS) {
            int m = i / BK;
            int j = i % BK;
            int mg = q_start + m;
            int sg = kb + j;
            half p_val = __float2half(0.0f);
            if (mg < q_end && mg < seq_len && j < k_tile_len &&
                sg <= mg && mg - sg < WINDOW) {
                int compact_s = sg - q_k_lo[m];
                float qk_val = qk_smem2[m * BK + j] * scaling;
                float s_val = __half2float(s_b[mg * st_s + compact_s]);
                float score = dd1[m] * qk_val + w2[m] * s_val;
                float safe_l = (l_i[m] > 0.0f) ? l_i[m] : 1.0f;
                float prob = expf(score - m_i[m]) / safe_l;
                p_val = __float2half(prob);
                float a_contrib = po1[m] * prob;
                if (a_contrib != 0.0f)
                    atomicAdd(&a_buf_out[b * (long long)(T * WINDOW) + mg * WINDOW + compact_s],
                              a_contrib);
            }
            p_smem2[i] = p_val;
        }
        __syncthreads();

        // Load V (overwrites kv_smem2), WMMA P@V -> accumulate into o_smem
        load_v_tile(V_b, st, sn_d, n, kv_smem2, kb, k_tile_len, seq_len, tid);
        __syncthreads();

        // P@V via WMMA: O[BM,D] += P[BM,BK] @ V[BK,D]
        // 4 warps x 2 output column tiles each = 8 tiles covering D=128
        // Each warp gets exclusive output columns, so no conflicts between warps.
        {
            const int tiles_per_warp = HEAD_DIM / WN / NWARPS;  // 2
            for (int t = 0; t < tiles_per_warp; t++) {
                int col = (warp_id * tiles_per_warp + t) * WN;

                wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;
                wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);

                #pragma unroll
                for (int kk = 0; kk < BK; kk += WK) {
                    wmma::load_matrix_sync(a_frag, p_smem2 + kk, BK);
                    wmma::load_matrix_sync(b_frag, kv_smem2 + kk * HEAD_DIM + col, HEAD_DIM);
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                // Each warp has exclusive output columns, use per-warp temp in smem
                // qk_smem2 has BM*BK=1024 floats. Each warp needs WM*WN=256 floats.
                // 4 warps x 256 = 1024 floats. Fits exactly.
                float* pv_tmp = qk_smem2 + warp_id * WM * WN;
                wmma::store_matrix_sync(pv_tmp, c_frag, WN, wmma::mem_row_major);

                // Accumulate into o_smem (each warp touches exclusive columns)
                for (int r = lane; r < WM * WN; r += WARP_SIZE) {
                    int rm = r / WN;
                    int rd = r % WN;
                    o_smem[rm * HEAD_DIM + col + rd] += pv_tmp[r];
                }
            }
        }
        __syncthreads();
    }

    // Store o_buf
    for (int i = tid; i < BM * HEAD_DIM; i += BLOCK_THREADS) {
        int m = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int mg = q_start + m;
        if (mg < q_end && mg < seq_len)
            o_buf[b * sb + mg * st + n * sn_d + d] = __float2half(o_smem[i]);
    }
}


// ======================================================================
// K3: Final output
// Grid: (num_tiles_f, B*N), Block: 128
// out[m,n,d] = (1+post_dd) * o_buf[m,n,d] + post_w2 * sum_s(a_buf[m,s] * V[s,n,d])
// ======================================================================
__global__ void k3_final_kernel(
    const float* __restrict__ a_buf,
    const half*  __restrict__ V,
    const half*  __restrict__ o_buf,
    const half*  __restrict__ post_w2,
    const half*  __restrict__ post_dd,
    half*        __restrict__ out,
    const int*   __restrict__ seq_lens,
    int T)
{
    const int BM_F = 64;
    const int tile_idx = blockIdx.x;
    const int bn = blockIdx.y;
    const int b = bn / NUM_HEADS;
    const int n = bn % NUM_HEADS;
    const int tid = threadIdx.x;
    const int q_start = tile_idx * BM_F;
    const int seq_len = seq_lens[b];

    if (q_start >= T) return;
    const int q_end = min(q_start + BM_F, T);

    const long long sb = (long long)T * NUM_HEADS * HEAD_DIM;
    const long long st_qkv = (long long)NUM_HEADS * HEAD_DIM;
    const long long sn_qkv = (long long)HEAD_DIM;
    const long long sb_w = (long long)T * NUM_HEADS;
    const long long st_w = (long long)NUM_HEADS;
    const long long sb_a = (long long)T * WINDOW;
    const long long st_a = (long long)WINDOW;

    // Each thread handles a subset of (m, d) output elements
    for (int md = tid; md < (q_end - q_start) * HEAD_DIM; md += BLOCK_THREADS) {
        int m_local = md / HEAD_DIM;
        int d = md % HEAD_DIM;
        int m = q_start + m_local;
        if (m >= seq_len) continue;

        float pw2 = __half2float(post_w2[b * sb_w + m * st_w + n]);
        float pdd1 = __half2float(post_dd[b * sb_w + m * st_w + n]) + 1.0f;

        float oval = pdd1 * __half2float(o_buf[b * sb + m * st_qkv + n * sn_qkv + d]);

        int m_k_lo = max(0, m - WINDOW + 1);
        int m_k_hi = min(m + 1, seq_len);

        float agg = 0.0f;
        for (int s = m_k_lo; s < m_k_hi; s++) {
            int compact_s = s - m_k_lo;
            float a_val = a_buf[b * sb_a + m * st_a + compact_s];
            if (a_val != 0.0f) {
                agg += a_val * __half2float(V[b * sb + s * st_qkv + n * sn_qkv + d]);
            }
        }
        oval += pw2 * agg;

        out[b * sb + m * st_qkv + n * sn_qkv + d] = __float2half(oval);
    }
}


// ======================================================================
// Host wrapper
// ======================================================================
torch::Tensor dc_residual_fwd_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor pre_w1,
    torch::Tensor pre_w2,
    torch::Tensor pre_dd,
    torch::Tensor post_w1,
    torch::Tensor post_w2,
    torch::Tensor post_dd,
    torch::Tensor seq_lens,
    float scaling)
{
    int B = Q.size(0);
    int T = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    TORCH_CHECK(N == NUM_HEADS, "Expected N=", NUM_HEADS);
    TORCH_CHECK(D == HEAD_DIM, "Expected D=", HEAD_DIM);
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous());

    auto opts_half = Q.options();
    auto opts_float = Q.options().dtype(torch::kFloat32);

    auto out_t = torch::empty_like(Q);
    auto s_buf = torch::empty({B, T, WINDOW}, opts_half);
    auto o_buf = torch::empty({B, T, N, D}, opts_half);
    auto a_buf = torch::zeros({B, T, WINDOW}, opts_float);
    auto m_buf = torch::empty({B, T, N}, opts_float);
    auto l_buf = torch::empty({B, T, N}, opts_float);

    int num_tiles = (T + BM - 1) / BM;
    int BM_F = 64;
    int num_tiles_f = (T + BM_F - 1) / BM_F;

    // K0
    {
        dim3 grid(num_tiles, B);
        dim3 block(BLOCK_THREADS);
        size_t smem = (BM + BK) * HEAD_DIM * sizeof(half) + 2 * BM * BK * sizeof(float);
        cudaFuncSetAttribute(k0_preagg_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        k0_preagg_kernel<<<grid, block, smem>>>(
            (const half*)Q.data_ptr<at::Half>(),
            (const half*)K.data_ptr<at::Half>(),
            (const half*)pre_w1.data_ptr<at::Half>(),
            (half*)s_buf.data_ptr<at::Half>(),
            seq_lens.data_ptr<int>(), scaling, T);
    }

    // K12
    {
        dim3 grid(num_tiles, B * N);
        dim3 block(BLOCK_THREADS);
        // q_smem + o_smem + kv_smem + qk_smem + p_smem
        size_t smem = BM * HEAD_DIM * sizeof(half)      // q_smem: 4096
                    + BM * HEAD_DIM * sizeof(float)      // o_smem: 8192
                    + BK * HEAD_DIM * sizeof(half)       // kv_smem: 16384
                    + BM * BK * sizeof(float)            // qk_smem: 4096
                    + BM * BK * sizeof(half);            // p_smem: 2048
        // total: 34816 bytes
        cudaFuncSetAttribute(k12_fused_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        k12_fused_kernel<<<grid, block, smem>>>(
            (const half*)Q.data_ptr<at::Half>(),
            (const half*)K.data_ptr<at::Half>(),
            (const half*)V.data_ptr<at::Half>(),
            (const half*)s_buf.data_ptr<at::Half>(),
            (const half*)pre_w2.data_ptr<at::Half>(),
            (const half*)pre_dd.data_ptr<at::Half>(),
            (const half*)post_w1.data_ptr<at::Half>(),
            (half*)o_buf.data_ptr<at::Half>(),
            a_buf.data_ptr<float>(),
            m_buf.data_ptr<float>(),
            l_buf.data_ptr<float>(),
            seq_lens.data_ptr<int>(), scaling, T);
    }

    // K3
    {
        dim3 grid(num_tiles_f, B * N);
        dim3 block(BLOCK_THREADS);
        k3_final_kernel<<<grid, block>>>(
            a_buf.data_ptr<float>(),
            (const half*)V.data_ptr<at::Half>(),
            (const half*)o_buf.data_ptr<at::Half>(),
            (const half*)post_w2.data_ptr<at::Half>(),
            (const half*)post_dd.data_ptr<at::Half>(),
            (half*)out_t.data_ptr<at::Half>(),
            seq_lens.data_ptr<int>(), T);
    }

    return out_t;
}

// ======================================================================
// Single-kernel V4 CUDA reference/benchmark path
//
// This mirrors triton_dc_onekernel_v4.py for the useful cache-four-QK cases:
//   - N=32, D=128, fp16 contiguous tensors
//   - 4 <= G <= 8, so HPG is 8 or 4
//   - chunk_size is 16 or 32
//   - KL = next_power_of_2(chunk_size + window - 1) <= 128
//
// It intentionally does not try to cover G=1/2. Holding four [BM,KL] QK
// matrices live across long HPG=16/32 sweep-2 loops was slower in Triton too.
// ======================================================================

#define ONK_THREADS 256
#define ONK_LOG2E 1.4426950408889634f

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    return __shfl_sync(0xffffffff, v, 0);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return __shfl_sync(0xffffffff, v, 0);
}

__device__ __forceinline__ long long onk_qkv_offset(
    int b, int t, int n, int d, int T)
{
    return (((long long)b * T + t) * NUM_HEADS + n) * HEAD_DIM + d;
}

__device__ __forceinline__ long long onk_w_offset(
    int b, int t, int n, int T)
{
    return ((long long)b * T + t) * NUM_HEADS + n;
}

__device__ __forceinline__ bool onk_q_valid(int q, int T, int seq_len) {
    return q < T && q < seq_len;
}

__device__ __forceinline__ bool onk_k_valid(
    int j, int k, int T, int seq_len, int kspan)
{
    return j < kspan && k >= 0 && k < T && k < seq_len;
}

__device__ __forceinline__ bool onk_attn_valid(
    int q, int j, int k, int T, int seq_len, int window, int kspan)
{
    return onk_q_valid(q, T, seq_len)
        && onk_k_valid(j, k, T, seq_len, kspan)
        && k <= q
        && (q - k) < window;
}

__device__ void onk_compute_qk(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    float* __restrict__ qk,
    int b, int q_start, int k_start, int head,
    int T, int seq_len, int chunk_size, int window, int kspan, int KL,
    float scaling)
{
    const int total = chunk_size * KL;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int m = idx / KL;
        int j = idx - m * KL;
        int q = q_start + m;
        int k = k_start + j;
        float acc = 0.0f;
        if (onk_q_valid(q, T, seq_len) && onk_k_valid(j, k, T, seq_len, kspan)) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                float qv = __half2float(Q[onk_qkv_offset(b, q, head, d, T)]);
                float kv = __half2float(K[onk_qkv_offset(b, k, head, d, T)]);
                acc += qv * kv;
            }
            acc *= scaling;
        }
        qk[idx] = acc;
    }
}

__device__ __forceinline__ void onk_load_q_smem(
    const half* __restrict__ Q,
    half* __restrict__ q_smem,
    int b, int q_start, int head,
    int T, int seq_len, int chunk_size)
{
    const int elems = chunk_size * HEAD_DIM;
    for (int idx = threadIdx.x; idx < elems; idx += blockDim.x) {
        int m = idx / HEAD_DIM;
        int d = idx - m * HEAD_DIM;
        int q = q_start + m;
        half val = __float2half(0.0f);
        if (onk_q_valid(q, T, seq_len))
            val = Q[onk_qkv_offset(b, q, head, d, T)];
        q_smem[idx] = val;
    }
}

__device__ __forceinline__ void onk_load_k64_smem(
    const half* __restrict__ K,
    half* __restrict__ k_smem,
    int b, int k_base, int head,
    int T, int seq_len, int k_tile_len)
{
    const int elems = BK * HEAD_DIM;
    for (int idx = threadIdx.x; idx < elems; idx += blockDim.x) {
        int j = idx / HEAD_DIM;
        int d = idx - j * HEAD_DIM;
        int k = k_base + j;
        half val = __float2half(0.0f);
        if (j < k_tile_len && k >= 0 && k < T && k < seq_len)
            val = K[onk_qkv_offset(b, k, head, d, T)];
        k_smem[idx] = val;
    }
}

__device__ __forceinline__ void onk_load_v64_smem(
    const half* __restrict__ V,
    half* __restrict__ v_smem,
    int b, int k_base, int head,
    int T, int seq_len, int k_tile_len)
{
    const int elems = BK * HEAD_DIM;
    for (int idx = threadIdx.x; idx < elems; idx += blockDim.x) {
        int j = idx / HEAD_DIM;
        int d = idx - j * HEAD_DIM;
        int k = k_base + j;
        half val = __float2half(0.0f);
        if (j < k_tile_len && k >= 0 && k < T && k < seq_len)
            val = V[onk_qkv_offset(b, k, head, d, T)];
        v_smem[idx] = val;
    }
}

__device__ void onk_compute_qk_wmma(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    float* __restrict__ qk,
    char* __restrict__ temp,
    int b, int q_start, int k_start, int head,
    int T, int seq_len, int chunk_size, int kspan, int KL,
    float scaling)
{
    half* q_smem = (half*)temp;
    half* k_smem = q_smem + chunk_size * HEAD_DIM;
    const int warp_id = threadIdx.x >> 5;
    const int row_tile = warp_id >> 2;
    const int col_tile = warp_id & 3;
    const int row = row_tile * WM;
    const int col = col_tile * WN;

    onk_load_q_smem(Q, q_smem, b, q_start, head, T, seq_len, chunk_size);
    __syncthreads();

    const int num_ktiles = (KL + BK - 1) / BK;
    for (int kt = 0; kt < num_ktiles; kt++) {
        int k_off = kt * BK;
        int k_tile_len = kspan > k_off ? min(BK, kspan - k_off) : 0;
        onk_load_k64_smem(K, k_smem, b, k_start + k_off, head,
                          T, seq_len, k_tile_len);
        __syncthreads();

        if (row < chunk_size) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            #pragma unroll
            for (int kk = 0; kk < HEAD_DIM; kk += WK) {
                wmma::load_matrix_sync(a_frag, q_smem + row * HEAD_DIM + kk, HEAD_DIM);
                wmma::load_matrix_sync(b_frag, k_smem + col * HEAD_DIM + kk, HEAD_DIM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            #pragma unroll
            for (int e = 0; e < c_frag.num_elements; e++)
                c_frag.x[e] *= scaling;
            wmma::store_matrix_sync(qk + row * KL + k_off + col,
                                    c_frag, KL, wmma::mem_row_major);
        }
        __syncthreads();
    }
}

__device__ void onk_compute_qk_wmma_half(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ pre_w1,
    half* __restrict__ qk,
    float* __restrict__ s_acc,
    char* __restrict__ temp,
    int b, int q_start, int k_start, int head,
    int T, int seq_len, int chunk_size, int kspan, int KL,
    float scaling, bool add_pre)
{
    half* q_smem = (half*)temp;
    half* k_smem = q_smem + chunk_size * HEAD_DIM;
    float* tile_smem = (float*)(k_smem + BK * HEAD_DIM);
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_tile = warp_id >> 2;
    const int col_tile = warp_id & 3;
    const int row = row_tile * WM;
    const int col = col_tile * WN;

    onk_load_q_smem(Q, q_smem, b, q_start, head, T, seq_len, chunk_size);
    __syncthreads();

    const int num_ktiles = (KL + BK - 1) / BK;
    for (int kt = 0; kt < num_ktiles; kt++) {
        int k_off = kt * BK;
        int k_tile_len = kspan > k_off ? min(BK, kspan - k_off) : 0;
        onk_load_k64_smem(K, k_smem, b, k_start + k_off, head,
                          T, seq_len, k_tile_len);
        __syncthreads();

        if (row < chunk_size) {
            wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);

            #pragma unroll
            for (int kk = 0; kk < HEAD_DIM; kk += WK) {
                wmma::load_matrix_sync(a_frag, q_smem + row * HEAD_DIM + kk, HEAD_DIM);
                wmma::load_matrix_sync(b_frag, k_smem + col * HEAD_DIM + kk, HEAD_DIM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

            #pragma unroll
            for (int e = 0; e < c_frag.num_elements; e++)
                c_frag.x[e] *= scaling;

            float* tile = tile_smem + warp_id * WM * WN;
            wmma::store_matrix_sync(tile, c_frag, WN, wmma::mem_row_major);
            for (int r = lane; r < WM * WN; r += WARP_SIZE) {
                int rm = r / WN;
                int rd = r - rm * WN;
                int m = row + rm;
                int j = k_off + col + rd;
                if (m < chunk_size && j < KL) {
                    float val = tile[r];
                    int idx = m * KL + j;
                    qk[idx] = __float2half(val);
                    if (add_pre) {
                        int q = q_start + m;
                        float w = onk_q_valid(q, T, seq_len)
                            ? __half2float(pre_w1[onk_w_offset(b, q, head, T)])
                            : 0.0f;
                        s_acc[idx] += w * val;
                    }
                }
            }
        }
        __syncthreads();
    }
}

__device__ void onk_store_xv_wmma(
    const float* __restrict__ x,
    const half* __restrict__ V,
    const half* __restrict__ weight,
    half* __restrict__ OUT,
    char* __restrict__ temp,
    int b, int q_start, int k_start, int head,
    int T, int seq_len, int chunk_size, int kspan, int KL,
    int mode)
{
    const int total = chunk_size * KL;
    half* x_half = (half*)temp;
    half* v_smem = x_half + total;
    float* tile_smem = (float*)(v_smem + BK * HEAD_DIM);

    for (int idx = threadIdx.x; idx < total; idx += blockDim.x)
        x_half[idx] = __float2half(x[idx]);
    __syncthreads();

    const int warp_id = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;
    const int row_tiles = chunk_size / WM;
    const int col_tiles = HEAD_DIM / WN;
    const int num_ktiles = (KL + BK - 1) / BK;
    const int total_tiles = row_tiles * col_tiles;

    for (int tile_id = warp_id; tile_id < total_tiles; tile_id += num_warps) {
        int row_tile = tile_id / col_tiles;
        int col_tile = tile_id - row_tile * col_tiles;
        int row = row_tile * WM;
        int col = col_tile * WN;

        wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int kt = 0; kt < num_ktiles; kt++) {
            int k_off = kt * BK;
            int k_tile_len = kspan > k_off ? min(BK, kspan - k_off) : 0;
            onk_load_v64_smem(V, v_smem, b, k_start + k_off, head,
                              T, seq_len, k_tile_len);
            __syncthreads();

            #pragma unroll
            for (int kk = 0; kk < BK; kk += WK) {
                wmma::load_matrix_sync(a_frag, x_half + row * KL + k_off + kk, KL);
                wmma::load_matrix_sync(b_frag, v_smem + kk * HEAD_DIM + col, HEAD_DIM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            __syncthreads();
        }

        float* tile_out = tile_smem + warp_id * WM * WN;
        wmma::store_matrix_sync(tile_out, c_frag, WN, wmma::mem_row_major);
        for (int r = (threadIdx.x & 31); r < WM * WN; r += WARP_SIZE) {
            int rm = r / WN;
            int rd = r - rm * WN;
            int m = row + rm;
            int d = col + rd;
            int q = q_start + m;
            if (m < chunk_size && d < HEAD_DIM && onk_q_valid(q, T, seq_len)) {
                long long o = onk_qkv_offset(b, q, head, d, T);
                float coeff = __half2float(weight[onk_w_offset(b, q, head, T)]);
                float val = tile_out[r];
                if (mode == 0) {
                    OUT[o] = __float2half((coeff + 1.0f) * val);
                } else {
                    float prev = __half2float(OUT[o]);
                    OUT[o] = __float2half(prev + coeff * val);
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();
}

__device__ void onk_store_xv_wmma_half(
    const half* __restrict__ x_half,
    const half* __restrict__ V,
    const half* __restrict__ weight,
    half* __restrict__ OUT,
    char* __restrict__ temp,
    int b, int q_start, int k_start, int head,
    int T, int seq_len, int chunk_size, int kspan, int KL,
    int mode)
{
    half* v_smem = (half*)temp;
    float* tile_smem = (float*)(v_smem + BK * HEAD_DIM);

    const int warp_id = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;
    const int row_tiles = chunk_size / WM;
    const int col_tiles = HEAD_DIM / WN;
    const int num_ktiles = (KL + BK - 1) / BK;
    const int total_tiles = row_tiles * col_tiles;

    for (int tile_id = warp_id; tile_id < total_tiles; tile_id += num_warps) {
        int row_tile = tile_id / col_tiles;
        int col_tile = tile_id - row_tile * col_tiles;
        int row = row_tile * WM;
        int col = col_tile * WN;

        wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
        wmma::fill_fragment(c_frag, 0.0f);

        for (int kt = 0; kt < num_ktiles; kt++) {
            int k_off = kt * BK;
            int k_tile_len = kspan > k_off ? min(BK, kspan - k_off) : 0;
            onk_load_v64_smem(V, v_smem, b, k_start + k_off, head,
                              T, seq_len, k_tile_len);
            __syncthreads();

            #pragma unroll
            for (int kk = 0; kk < BK; kk += WK) {
                wmma::load_matrix_sync(a_frag, x_half + row * KL + k_off + kk, KL);
                wmma::load_matrix_sync(b_frag, v_smem + kk * HEAD_DIM + col, HEAD_DIM);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            __syncthreads();
        }

        float* tile_out = tile_smem + warp_id * WM * WN;
        wmma::store_matrix_sync(tile_out, c_frag, WN, wmma::mem_row_major);
        for (int r = (threadIdx.x & 31); r < WM * WN; r += WARP_SIZE) {
            int rm = r / WN;
            int rd = r - rm * WN;
            int m = row + rm;
            int d = col + rd;
            int q = q_start + m;
            if (m < chunk_size && d < HEAD_DIM && onk_q_valid(q, T, seq_len)) {
                long long o = onk_qkv_offset(b, q, head, d, T);
                float coeff = __half2float(weight[onk_w_offset(b, q, head, T)]);
                float val = tile_out[r];
                if (mode == 0) {
                    OUT[o] = __float2half((coeff + 1.0f) * val);
                } else {
                    float prev = __half2float(OUT[o]);
                    OUT[o] = __float2half(prev + coeff * val);
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();
}

__device__ void onk_add_pre(
    const half* __restrict__ pre_w1,
    const float* __restrict__ qk,
    float* __restrict__ s_acc,
    int b, int q_start, int head, int T, int seq_len, int chunk_size, int KL)
{
    const int total = chunk_size * KL;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int m = idx / KL;
        int q = q_start + m;
        float w = 0.0f;
        if (onk_q_valid(q, T, seq_len))
            w = __half2float(pre_w1[onk_w_offset(b, q, head, T)]);
        s_acc[idx] += w * qk[idx];
    }
}

__device__ void onk_consume_head(
    const half* __restrict__ V,
    const half* __restrict__ pre_w2,
    const half* __restrict__ pre_dd,
    const half* __restrict__ post_w1,
    const half* __restrict__ post_dd,
    half* __restrict__ OUT,
    half* __restrict__ qk,
    const float* __restrict__ s_acc,
    float* __restrict__ a_acc,
    char* __restrict__ temp,
    int b, int q_start, int k_start, int head,
    int T, int seq_len, int chunk_size, int window, int kspan, int KL)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    for (int row_base = 0; row_base < chunk_size; row_base += 8) {
        int m = row_base + warp_id;
        if (warp_id < 8 && m < chunk_size) {
            int q = q_start + m;
            float pw2 = 0.0f, pdd = 0.0f, pw1p = 0.0f;
            if (onk_q_valid(q, T, seq_len)) {
                pw2 = __half2float(pre_w2[onk_w_offset(b, q, head, T)]);
                pdd = __half2float(pre_dd[onk_w_offset(b, q, head, T)]);
                pw1p = __half2float(post_w1[onk_w_offset(b, q, head, T)]);
            }

            float row_max = -FLT_MAX;
            for (int j = lane; j < KL; j += 32) {
                int k = k_start + j;
                bool valid = onk_attn_valid(q, j, k, T, seq_len, window, kspan);
                float qkv = __half2float(qk[m * KL + j]);
                float score = (pdd + 1.0f) * qkv + pw2 * s_acc[m * KL + j];
                row_max = fmaxf(row_max, valid ? score : -FLT_MAX);
            }
            row_max = warp_reduce_max(row_max);

            float row_sum = 0.0f;
            for (int j = lane; j < KL; j += 32) {
                int k = k_start + j;
                bool valid = onk_attn_valid(q, j, k, T, seq_len, window, kspan);
                float qkv = __half2float(qk[m * KL + j]);
                float score = (pdd + 1.0f) * qkv + pw2 * s_acc[m * KL + j];
                float p = valid ? exp2f((score - row_max) * ONK_LOG2E) : 0.0f;
                qk[m * KL + j] = __float2half(p);
                row_sum += p;
            }
            row_sum = warp_reduce_sum(row_sum);
            float denom = row_sum > 0.0f ? row_sum : 1.0f;

            for (int j = lane; j < KL; j += 32) {
                float p = __half2float(qk[m * KL + j]) / denom;
                qk[m * KL + j] = __float2half(p);
                a_acc[m * KL + j] += pw1p * p;
            }
        }
        __syncthreads();
    }

    onk_store_xv_wmma_half(qk, V, post_dd, OUT, temp,
                           b, q_start, k_start, head,
                           T, seq_len, chunk_size, kspan, KL, 0);
}

__global__ void dc_onekernel_v4_cache4_kernel(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ pre_w1,
    const half* __restrict__ pre_w2,
    const half* __restrict__ pre_dd,
    const half* __restrict__ post_w1,
    const half* __restrict__ post_w2,
    const half* __restrict__ post_dd,
    half* __restrict__ OUT,
    const int* __restrict__ seq_lens,
    float scaling,
    int T, int window, int G, int HPG, int chunk_size, int KL)
{
    const int pid_c = blockIdx.x;
    const int pid_bg = blockIdx.y;
    const int b = pid_bg / G;
    const int g = pid_bg - b * G;
    const int q_start = pid_c * chunk_size;
    const int head_start = g * HPG;
    const int seq_len = seq_lens[b];
    const int kspan = chunk_size + window - 1;
    int k_start = q_start - window + 1;
    if (k_start < 0) k_start = 0;

    if (q_start >= T) return;

    const int total = chunk_size * KL;
    extern __shared__ float smem_f[];
    float* s_acc = smem_f;
    float* a_acc = s_acc + total;
    half* qk_cache = (half*)(a_acc + total);       // 4 * total
    half* qk_tmp = qk_cache;
    char* temp = (char*)(qk_cache + 4 * total);

    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        s_acc[idx] = 0.0f;
        a_acc[idx] = 0.0f;
    }
    __syncthreads();

    const int num_pairs = HPG / 2;

    for (int pair_idx = 0; pair_idx < num_pairs - 2; pair_idx++) {
        int h0 = head_start + pair_idx * 2;
        int h1 = h0 + 1;
        onk_compute_qk_wmma_half(Q, K, pre_w1, qk_tmp, s_acc, temp,
                                 b, q_start, k_start, h0,
                                 T, seq_len, chunk_size, kspan, KL, scaling, true);
        __syncthreads();

        onk_compute_qk_wmma_half(Q, K, pre_w1, qk_tmp, s_acc, temp,
                                 b, q_start, k_start, h1,
                                 T, seq_len, chunk_size, kspan, KL, scaling, true);
        __syncthreads();
    }

    int cached_heads[4];
    cached_heads[0] = head_start + (num_pairs - 2) * 2;
    cached_heads[1] = cached_heads[0] + 1;
    cached_heads[2] = head_start + (num_pairs - 1) * 2;
    cached_heads[3] = cached_heads[2] + 1;

    #pragma unroll
    for (int ci = 0; ci < 4; ci++) {
        half* qk = qk_cache + ci * total;
        onk_compute_qk_wmma_half(Q, K, pre_w1, qk, s_acc, temp,
                                 b, q_start, k_start, cached_heads[ci],
                                 T, seq_len, chunk_size, kspan, KL, scaling, true);
        __syncthreads();
    }

    // Same order as Triton V4: last pair, previous cached pair, then recomputed pairs.
    onk_consume_head(V, pre_w2, pre_dd, post_w1, post_dd, OUT,
                     qk_cache + 2 * total, s_acc, a_acc, temp,
                     b, q_start, k_start, cached_heads[2],
                     T, seq_len, chunk_size, window, kspan, KL);
    onk_consume_head(V, pre_w2, pre_dd, post_w1, post_dd, OUT,
                     qk_cache + 3 * total, s_acc, a_acc, temp,
                     b, q_start, k_start, cached_heads[3],
                     T, seq_len, chunk_size, window, kspan, KL);
    onk_consume_head(V, pre_w2, pre_dd, post_w1, post_dd, OUT,
                     qk_cache, s_acc, a_acc, temp,
                     b, q_start, k_start, cached_heads[0],
                     T, seq_len, chunk_size, window, kspan, KL);
    onk_consume_head(V, pre_w2, pre_dd, post_w1, post_dd, OUT,
                     qk_cache + total, s_acc, a_acc, temp,
                     b, q_start, k_start, cached_heads[1],
                     T, seq_len, chunk_size, window, kspan, KL);

    for (int pair_idx = 0; pair_idx < num_pairs - 2; pair_idx++) {
        int h0 = head_start + pair_idx * 2;
        int h1 = h0 + 1;
        onk_compute_qk_wmma_half(Q, K, pre_w1, qk_tmp, s_acc, temp,
                                 b, q_start, k_start, h0,
                                 T, seq_len, chunk_size, kspan, KL, scaling, false);
        __syncthreads();
        onk_consume_head(V, pre_w2, pre_dd, post_w1, post_dd, OUT,
                         qk_tmp, s_acc, a_acc, temp,
                         b, q_start, k_start, h0,
                         T, seq_len, chunk_size, window, kspan, KL);

        onk_compute_qk_wmma_half(Q, K, pre_w1, qk_tmp, s_acc, temp,
                                 b, q_start, k_start, h1,
                                 T, seq_len, chunk_size, kspan, KL, scaling, false);
        __syncthreads();
        onk_consume_head(V, pre_w2, pre_dd, post_w1, post_dd, OUT,
                         qk_tmp, s_acc, a_acc, temp,
                         b, q_start, k_start, h1,
                         T, seq_len, chunk_size, window, kspan, KL);
    }

    for (int h_local = 0; h_local < HPG; h_local++) {
        int head = head_start + h_local;
        onk_store_xv_wmma(a_acc, V, post_w2, OUT, temp,
                          b, q_start, k_start, head,
                          T, seq_len, chunk_size, kspan, KL, 1);
    }
}

static int next_power_of_2_int(int x) {
    int p = 1;
    while (p < x) p <<= 1;
    return p;
}

torch::Tensor dc_onekernel_v4_fwd_cuda(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor pre_w1,
    torch::Tensor pre_w2,
    torch::Tensor pre_dd,
    torch::Tensor post_w1,
    torch::Tensor post_w2,
    torch::Tensor post_dd,
    torch::Tensor seq_lens,
    float scaling,
    int window,
    int G,
    int chunk_size)
{
    int B = Q.size(0);
    int T = Q.size(1);
    int N = Q.size(2);
    int D = Q.size(3);

    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q/K/V must be CUDA tensors");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat16, "Q/K/V must be float16");
    TORCH_CHECK(pre_w1.scalar_type() == torch::kFloat16, "DC weights must be float16");
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt32, "seq_lens must be int32");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(), "Q/K/V must be contiguous");
    TORCH_CHECK(pre_w1.is_contiguous() && pre_w2.is_contiguous() && pre_dd.is_contiguous()
             && post_w1.is_contiguous() && post_w2.is_contiguous() && post_dd.is_contiguous(),
             "DC weights must be contiguous");
    TORCH_CHECK(N == NUM_HEADS, "Expected N=", NUM_HEADS);
    TORCH_CHECK(D == HEAD_DIM, "Expected D=", HEAD_DIM);
    TORCH_CHECK(NUM_HEADS % G == 0, "N must be divisible by G");
    int HPG = NUM_HEADS / G;
    TORCH_CHECK(G >= 4 && G <= 8, "CUDA V4 cache4 supports G=4 or G=8");
    TORCH_CHECK(HPG >= 4 && HPG <= 8 && (HPG % 2 == 0), "CUDA V4 cache4 supports HPG=4 or HPG=8");
    TORCH_CHECK(chunk_size == 16 || chunk_size == 32, "CUDA V4 supports chunk_size=16 or 32");

    int W = window < T ? window : T;
    int kspan = chunk_size + W - 1;
    int KL = next_power_of_2_int(kspan);
    TORCH_CHECK(KL <= 128, "CUDA V4 cache4 supports KL<=128 only");

    auto out = torch::empty_like(Q);
    int num_chunks = (T + chunk_size - 1) / chunk_size;
    dim3 grid(num_chunks, B * G);
    dim3 block(ONK_THREADS);
    size_t total = (size_t)chunk_size * KL;
    size_t block_warps = ONK_THREADS / WARP_SIZE;
    size_t base_smem = (size_t)2 * total * sizeof(float)
                     + (size_t)4 * total * sizeof(half);
    size_t qk_temp_smem =
        (size_t)chunk_size * HEAD_DIM * sizeof(half)
        + (size_t)BK * HEAD_DIM * sizeof(half)
        + block_warps * WM * WN * sizeof(float);
    size_t pv_temp_smem =
        (size_t)BK * HEAD_DIM * sizeof(half)
        + block_warps * WM * WN * sizeof(float);
    size_t av_temp_smem =
        total * sizeof(half)
        + (size_t)BK * HEAD_DIM * sizeof(half)
        + block_warps * WM * WN * sizeof(float);
    size_t temp_smem = qk_temp_smem;
    if (pv_temp_smem > temp_smem) temp_smem = pv_temp_smem;
    if (av_temp_smem > temp_smem) temp_smem = av_temp_smem;
    size_t smem = base_smem + temp_smem;
    cudaFuncSetAttribute(dc_onekernel_v4_cache4_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dc_onekernel_v4_cache4_kernel<<<grid, block, smem>>>(
        (const half*)Q.data_ptr<at::Half>(),
        (const half*)K.data_ptr<at::Half>(),
        (const half*)V.data_ptr<at::Half>(),
        (const half*)pre_w1.data_ptr<at::Half>(),
        (const half*)pre_w2.data_ptr<at::Half>(),
        (const half*)pre_dd.data_ptr<at::Half>(),
        (const half*)post_w1.data_ptr<at::Half>(),
        (const half*)post_w2.data_ptr<at::Half>(),
        (const half*)post_dd.data_ptr<at::Half>(),
        (half*)out.data_ptr<at::Half>(),
        seq_lens.data_ptr<int>(),
        scaling, T, W, G, HPG, chunk_size, KL);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dc_residual_fwd_cuda, "DC Residual Attention Forward (CUDA)");
    m.def("onekernel_v4_forward", &dc_onekernel_v4_fwd_cuda,
          "DC one-kernel V4 cache4 forward (CUDA)");
}
