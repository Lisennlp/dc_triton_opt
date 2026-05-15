"""Two-kernel post-only DC attention.

This version keeps the first pass group-serial to avoid atomics:
- K1: one program per (chunk, batch-group). It loops heads in the group,
  computes plain attention, stores direct post_dd-scaled PV output, and writes
  the group-level fp16 a_acc to A_BUF[B, G, C, BM, KL].
- K2: one program per (chunk, batch-group, head). It computes A_BUF @ V_h,
  applies post_w2, and adds it to OUT.

This tests whether parallelizing only the final AV pass can beat the single
kernel PostV1 without paying atomic-add costs.
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _post2k_k1_group_accum(
    Q, K, V,
    POST_W1, POST_DD,
    OUT, A_BUF, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_wb, stride_wt, stride_wn,
    stride_ob, stride_ot, stride_on, stride_od,
    stride_ab, stride_ag, stride_ac, stride_am, stride_ak,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BM: tl.constexpr, KL: tl.constexpr,
    G: tl.constexpr, HPG: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    head_start = g * HPG
    q_start = pid_c * BM

    q_offs = q_start + tl.arange(0, BM)
    d_offs = tl.arange(0, D)
    kl_offs = tl.arange(0, KL)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)
    q_mask = (q_offs < T) & (q_offs < seq_len)

    k_start = q_start - W + 1
    if k_start < 0:
        k_start = 0
    k_offs = k_start + kl_offs
    k_mask = (k_offs < T) & (k_offs >= 0) & (k_offs < seq_len) & (kl_offs < (BM + W - 1))
    causal = k_offs[None, :] <= q_offs[:, None]
    win = (q_offs[:, None] - k_offs[None, :]) < W
    valid = causal & win & q_mask[:, None] & k_mask[None, :]

    a_acc = tl.zeros([BM, KL], dtype=tl.float32)

    for h in range(HPG):
        ni = (head_start + h).to(tl.int64)

        q_ptrs = (Q + b * stride_qb + q_offs[:, None].to(tl.int64) * stride_qt
                  + ni * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
        q_blk = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        k_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                  + ni * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
        k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

        score = tl.dot(q_blk, tl.trans(k_blk)) * scaling
        score = tl.where(valid, score, float("-inf"))

        rm = tl.max(score, axis=1)
        p = tl.exp2((score - rm[:, None]) * LOG2E)
        p = tl.where(valid, p, 0.0)
        rs = tl.sum(p, axis=1)
        sl = tl.where(rs > 0.0, rs, 1.0)
        probs = p / sl[:, None]

        v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                  + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        pv = tl.dot(probs.to(v_blk.dtype), v_blk)

        pdd = tl.load(POST_DD + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                  + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
        tl.store(o_ptrs, (pdd[:, None] + 1.0) * pv, mask=q_mask[:, None])

        pw1 = tl.load(POST_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                      mask=q_mask, other=0.0).to(tl.float32)
        a_acc += pw1[:, None] * probs

    a_ptrs = (A_BUF + b * stride_ab + g * stride_ag + pid_c * stride_ac
              + tl.arange(0, BM)[:, None].to(tl.int64) * stride_am
              + kl_offs[None, :].to(tl.int64) * stride_ak)
    tl.store(a_ptrs, a_acc.to(tl.float16), mask=q_mask[:, None] & k_mask[None, :])


@triton.jit
def _post2k_k2_final_av(
    V, POST_W2, OUT, A_BUF, SEQ_LENS,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_wb, stride_wt, stride_wn,
    stride_ob, stride_ot, stride_on, stride_od,
    stride_ab, stride_ag, stride_ac, stride_am, stride_ak,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BM: tl.constexpr, KL: tl.constexpr,
    G: tl.constexpr, HPG: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bg = tl.program_id(1)
    h = tl.program_id(2)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    ni = (g * HPG + h).to(tl.int64)
    q_start = pid_c * BM

    q_offs = q_start + tl.arange(0, BM)
    d_offs = tl.arange(0, D)
    kl_offs = tl.arange(0, KL)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)
    q_mask = (q_offs < T) & (q_offs < seq_len)

    k_start = q_start - W + 1
    if k_start < 0:
        k_start = 0
    k_offs = k_start + kl_offs
    k_mask = (k_offs < T) & (k_offs >= 0) & (k_offs < seq_len) & (kl_offs < (BM + W - 1))

    a_ptrs = (A_BUF + b * stride_ab + g * stride_ag + pid_c * stride_ac
              + tl.arange(0, BM)[:, None].to(tl.int64) * stride_am
              + kl_offs[None, :].to(tl.int64) * stride_ak)
    a_blk = tl.load(a_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=0.0)

    v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
              + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
    v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
    av = tl.dot(a_blk, v_blk)

    pw2 = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                  mask=q_mask, other=0.0).to(tl.float32)
    o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
              + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
    o_prev = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    tl.store(o_ptrs, o_prev + pw2[:, None] * av, mask=q_mask[:, None])


@triton.jit
def _post2k_k2_final_av_wide2(
    V, POST_W2, OUT, A_BUF, SEQ_LENS,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_wb, stride_wt, stride_wn,
    stride_ob, stride_ot, stride_on, stride_od,
    stride_ab, stride_ag, stride_ac, stride_am, stride_ak,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BM: tl.constexpr, KL: tl.constexpr,
    G: tl.constexpr, HPG: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bg = tl.program_id(1)
    pair_idx = tl.program_id(2)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    n0 = (g * HPG + pair_idx * 2).to(tl.int64)
    q_start = pid_c * BM

    q_offs = q_start + tl.arange(0, BM)
    d2_offs = tl.arange(0, 256)
    head_delta = d2_offs // 128
    d_pair = d2_offs - head_delta * 128
    kl_offs = tl.arange(0, KL)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)
    q_mask = (q_offs < T) & (q_offs < seq_len)

    k_start = q_start - W + 1
    if k_start < 0:
        k_start = 0
    k_offs = k_start + kl_offs
    k_mask = (k_offs < T) & (k_offs >= 0) & (k_offs < seq_len) & (kl_offs < (BM + W - 1))

    a_ptrs = (A_BUF + b * stride_ab + g * stride_ag + pid_c * stride_ac
              + tl.arange(0, BM)[:, None].to(tl.int64) * stride_am
              + kl_offs[None, :].to(tl.int64) * stride_ak)
    a_blk = tl.load(a_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=0.0)

    n_pair = n0 + head_delta.to(tl.int64)
    v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
              + n_pair[None, :] * stride_vn + d_pair[None, :].to(tl.int64) * stride_vd)
    v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
    av = tl.dot(a_blk, v_blk)

    pw2_0 = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + n0 * stride_wn,
                    mask=q_mask, other=0.0).to(tl.float32)
    pw2_1 = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + (n0 + 1) * stride_wn,
                    mask=q_mask, other=0.0).to(tl.float32)
    pw2 = tl.where(head_delta[None, :] == 0, pw2_0[:, None], pw2_1[:, None])

    o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
              + n_pair[None, :] * stride_on + d_pair[None, :].to(tl.int64) * stride_od)
    o_prev = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
    tl.store(o_ptrs, o_prev + pw2 * av, mask=q_mask[:, None])


class TritonDCOneKernel:
    """Post-only DC using two kernels and a group-level A buffer."""

    @staticmethod
    def make_buffers(q, G=16, chunk_size=16, window=256):
        B, T, _, _ = q.shape
        W = min(int(window), T)
        KL = triton.next_power_of_2(chunk_size + W - 1)
        num_chunks = triton.cdiv(T, chunk_size)
        return {
            "a_buf": torch.empty(
                (B, G, num_chunks, chunk_size, KL),
                device=q.device,
                dtype=q.dtype,
            )
        }

    @staticmethod
    def forward(
        q, k, v, dc_weights, scaling, window,
        seq_lens=None, G=16, chunk_size=16, buffers=None,
    ):
        if len(dc_weights) == 6:
            _, _, _, post_w1, post_w2, post_dd = dc_weights
        elif len(dc_weights) == 3:
            post_w1, post_w2, post_dd = dc_weights
        else:
            raise ValueError("dc_weights must be either 6 full weights or 3 post-only weights")

        B, T, N, D = q.shape
        W = min(int(window), T)
        HPG = N // G
        assert N % G == 0
        assert HPG >= 1, f"HPG={HPG} must be >= 1"

        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        post_w1 = post_w1.contiguous()
        post_w2 = post_w2.contiguous()
        post_dd = post_dd.contiguous()

        KSPAN = chunk_size + W - 1
        KL = triton.next_power_of_2(KSPAN)
        num_chunks = triton.cdiv(T, chunk_size)

        if buffers is None or "a_buf" not in buffers:
            buffers = TritonDCOneKernel.make_buffers(q, G=G, chunk_size=chunk_size, window=W)
        a_buf = buffers["a_buf"]
        expected_shape = (B, G, num_chunks, chunk_size, KL)
        if tuple(a_buf.shape) != expected_shape:
            raise ValueError(f"a_buf shape {tuple(a_buf.shape)} != expected {expected_shape}")
        if a_buf.dtype != q.dtype:
            raise ValueError("a_buf must have the same dtype as q")

        out = torch.empty((B, T, N, D), device=q.device, dtype=q.dtype)
        grid_k1 = (num_chunks, B * G)
        use_wide2 = D == 128 and HPG % 2 == 0
        grid_k2 = (num_chunks, B * G, HPG // 2) if use_wide2 else (num_chunks, B * G, HPG)

        _post2k_k1_group_accum[grid_k1](
            q, k, v,
            post_w1, post_dd,
            out, a_buf, seq_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            post_w1.stride(0), post_w1.stride(1), post_w1.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            a_buf.stride(0), a_buf.stride(1), a_buf.stride(2), a_buf.stride(3), a_buf.stride(4),
            scaling,
            B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL, G=G, HPG=HPG,
            num_warps=4, num_stages=1,
        )

        if use_wide2:
            _post2k_k2_final_av_wide2[grid_k2](
                v, post_w2, out, a_buf, seq_lens,
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                post_w2.stride(0), post_w2.stride(1), post_w2.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                a_buf.stride(0), a_buf.stride(1), a_buf.stride(2), a_buf.stride(3), a_buf.stride(4),
                B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL, G=G, HPG=HPG,
                num_warps=4, num_stages=1,
            )
        else:
            _post2k_k2_final_av[grid_k2](
                v, post_w2, out, a_buf, seq_lens,
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                post_w2.stride(0), post_w2.stride(1), post_w2.stride(2),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                a_buf.stride(0), a_buf.stride(1), a_buf.stride(2), a_buf.stride(3), a_buf.stride(4),
                B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL, G=G, HPG=HPG,
                num_warps=4, num_stages=1,
            )
        return out
