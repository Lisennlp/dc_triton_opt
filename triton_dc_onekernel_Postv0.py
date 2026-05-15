"""Single-kernel post-only DC attention v0.

This variant removes the whole pre-DC path:
- no pre_w1 / pre_w2 / pre_dd
- score is plain scaled QK
- each head computes QK once, then immediately does softmax, direct PV, and
  post_w1 accumulation
- final pass applies post_w2 * (post_w1-aggregated probs @ V)

The forward API accepts either a full 6-tuple DC weight pack or a post-only
3-tuple. In both cases the kernel only receives post_w1, post_w2, post_dd.
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["B", "T", "N", "D", "W", "BM", "KL", "G", "HPG"],
)
@triton.jit
def _post_only_onekernel(
    Q, K, V,
    POST_W1, POST_W2, POST_DD,
    OUT, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_wb, stride_wt, stride_wn,
    stride_ob, stride_ot, stride_on, stride_od,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BM: tl.constexpr, KL: tl.constexpr,
    G: tl.constexpr, HPG: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    q_start = pid_c * BM
    head_start = g * HPG

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

        pw1p = tl.load(POST_W1 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                       mask=q_mask, other=0.0).to(tl.float32)
        a_acc += pw1p[:, None] * probs

        v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                  + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        pv = tl.dot(probs.to(v_blk.dtype), v_blk)

        pddp = tl.load(POST_DD + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                       mask=q_mask, other=0.0).to(tl.float32)
        o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                  + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
        tl.store(o_ptrs, (pddp[:, None] + 1.0) * pv, mask=q_mask[:, None])

    a_acc = a_acc.to(tl.float16)

    for h in range(HPG):
        ni = (head_start + h).to(tl.int64)

        v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                  + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
        v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
        av = tl.dot(a_acc, v_blk)

        pw2p = tl.load(POST_W2 + b * stride_wb + q_offs.to(tl.int64) * stride_wt + ni * stride_wn,
                       mask=q_mask, other=0.0).to(tl.float32)
        o_ptrs = (OUT + b * stride_ob + q_offs[:, None].to(tl.int64) * stride_ot
                  + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
        o_prev = tl.load(o_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)
        tl.store(o_ptrs, o_prev + pw2p[:, None] * av, mask=q_mask[:, None])


class TritonDCOneKernel:
    """Post-only DC: no pre path, one QK per head, post mixing only."""

    @staticmethod
    def forward(
        q, k, v, dc_weights, scaling, window,
        seq_lens=None, G=16, chunk_size=16,
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

        out = torch.empty((B, T, N, D), device=q.device, dtype=q.dtype)
        num_chunks = triton.cdiv(T, chunk_size)
        grid = (num_chunks, B * G)

        _post_only_onekernel[grid](
            q, k, v,
            post_w1, post_w2, post_dd,
            out, seq_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            post_w1.stride(0), post_w1.stride(1), post_w1.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            scaling,
            B=B, T=T, N=N, D=D, W=W, BM=chunk_size, KL=KL, G=G, HPG=HPG,
        )
        return out
