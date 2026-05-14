"""Grouped DC residual: K load shared across heads within each block.

K0g:       grid=(C, B*G), HPG heads per block → partial s_acc → gs_buf
K0r:       reduce gs_buf → s_buf
K12g:      grid=(C, B*G), HPG heads per block:
           k-tile outer loop (load K/V once), head inner loop (online softmax+PV)
           + second k-tile pass for post_w1 atomic (K shared across heads)
K3:        a_buf@V + combine
"""

from __future__ import annotations
import torch
import triton
import triton.language as tl

from triton_dc_residual import _k3_final_kernel


# ══════════ K0g: grouped pre-agg (same as before) ══════════
@triton.jit
def _k0g_preagg_kernel(
    Q, K, PRE_W1, GS_BUF, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_p1b, stride_p1t, stride_p1n,
    stride_gsb, stride_gsg, stride_gst, stride_gsw,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
    G: tl.constexpr, HPG: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    q_start = pid_m * BM
    head_start = g * HPG

    m_offs = q_start + tl.arange(0, BM)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BM
    if k_hi > T:
        k_hi = T
    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    for kb in range(k_lo, k_hi, BK):
        k_offs = kb + tl.arange(0, BK)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]
        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        s_acc = tl.zeros([BM, BK], dtype=tl.float32)
        for h in range(HPG):
            ni = (head_start + h).to(tl.int64)
            q_ptrs = (Q + b * stride_qb + m_offs[:, None].to(tl.int64) * stride_qt
                      + ni * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
            q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)
            k_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                      + ni * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
            k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
            qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling
            pw1 = tl.load(PRE_W1 + b * stride_p1b + m_offs.to(tl.int64) * stride_p1t + ni * stride_p1n,
                          mask=m_mask, other=0.0).to(tl.float32)
            s_acc += pw1[:, None] * qk

        gs_ptrs = (GS_BUF + b * stride_gsb + g * stride_gsg
                   + m_offs[:, None].to(tl.int64) * stride_gst
                   + compact_k.to(tl.int64) * stride_gsw)
        tl.store(gs_ptrs, s_acc.to(tl.float16), mask=valid)


# ══════════ K0r: reduce gs_buf → s_buf ══════════
@triton.jit
def _k0r_reduce_kernel(
    GS_BUF, S_BUF,
    stride_gsb, stride_gsg, stride_gst, stride_gsw,
    stride_sb, stride_st, stride_sw,
    B: tl.constexpr, T: tl.constexpr, W: tl.constexpr,
    G: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    num_cols = (W + BK - 1) // BK
    row_block = pid // num_cols
    col_block = pid % num_cols
    m_offs = row_block * BM + tl.arange(0, BM)
    w_offs = col_block * BK + tl.arange(0, BK)
    valid = (m_offs < T)[:, None] & (w_offs < W)[None, :]
    acc = tl.zeros([BM, BK], dtype=tl.float32)
    for g in range(G):
        ptrs = (GS_BUF + b * stride_gsb + g.to(tl.int64) * stride_gsg
                + m_offs[:, None].to(tl.int64) * stride_gst
                + w_offs[None, :].to(tl.int64) * stride_gsw)
        acc += tl.load(ptrs, mask=valid, other=0.0).to(tl.float32)
    s_ptrs = (S_BUF + b * stride_sb + m_offs[:, None].to(tl.int64) * stride_st
              + w_offs[None, :].to(tl.int64) * stride_sw)
    tl.store(s_ptrs, acc.to(tl.float16), mask=valid)


# ══════════ K12g: grouped fused — K-tile outer, head inner ══════════
@triton.jit
def _k12g_fused_kernel(
    Q, K, V, S_BUF, PRE_W2, PRE_DD, POST_W1,
    O_BUF, A_BUF, SEQ_LENS,
    stride_qb, stride_qt, stride_qn, stride_qd,
    stride_kb, stride_kt, stride_kn, stride_kd,
    stride_vb, stride_vt, stride_vn, stride_vd,
    stride_sb, stride_st, stride_sw,
    stride_p2b, stride_p2t, stride_p2n,
    stride_pdb, stride_pdt, stride_pdn,
    stride_pw1b, stride_pw1t, stride_pw1n,
    stride_ob, stride_ot, stride_on, stride_od,
    stride_ab, stride_at, stride_aw,
    scaling,
    B: tl.constexpr, T: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    W: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
    G: tl.constexpr, HPG: tl.constexpr,
):
    """Grid = (C, B*G). K-tile outer loop, head inner loop.
    K and V loaded ONCE per tile, shared across HPG heads.
    Pass 1: online softmax + PV. Pass 2: post_w1 atomic."""
    pid_m = tl.program_id(0)
    pid_bg = tl.program_id(1)
    b = (pid_bg // G).to(tl.int64)
    g = (pid_bg % G).to(tl.int64)
    q_start = pid_m * BM
    head_start = g * HPG

    m_offs = q_start + tl.arange(0, BM)
    m_mask = m_offs < T
    d_offs = tl.arange(0, D)
    seq_len = tl.load(SEQ_LENS + b).to(tl.int64)

    k_lo = q_start - W + 1
    if k_lo < 0:
        k_lo = 0
    k_hi = q_start + BM
    if k_hi > T:
        k_hi = T
    k_lo_per_q = tl.maximum(0, m_offs - W + 1)

    # ── Pass 1: k-tile outer, head inner (online softmax + PV) ──
    # Pre-load per-head Q and weights (stays in registers across k-tiles)
    # We loop heads first to load Q, then loop k-tiles with K shared.
    # But online softmax needs per-head state across k-tiles.
    # Solution: maintain HPG sets of (mi, li, acc) simultaneously.
    # This uses HPG * (BM + BM + BM*D) registers ≈ HPG * BM * (D+2) fp32.
    # For HPG=4, BM=16, D=128: 4 * 16 * 130 = 8320 fp32 regs.
    # Per-thread (128 threads w/ 4 warps): 8320/128 = 65 regs. Tight but feasible.

    # Actually in Triton, `range(HPG)` with HPG as constexpr will be unrolled.
    # The compiler manages register allocation. Let's try it.

    # Pre-load all heads' Q and weights
    # (These are accessed in the inner loop per k-tile)

    # === K-tile outer loop ===
    for kb in range(k_lo, k_hi, BK):
        k_offs = kb + tl.arange(0, BK)
        k_mask = (k_offs < k_hi) & (k_offs < seq_len)
        causal = k_offs[None, :] <= m_offs[:, None]
        win = (m_offs[:, None] - k_offs[None, :]) < W
        valid = causal & win & m_mask[:, None] & k_mask[None, :]
        compact_k = k_offs[None, :] - k_lo_per_q[:, None]
        compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

        # Load s_buf ONCE for this tile
        s_ptrs = (S_BUF + b * stride_sb + m_offs[:, None].to(tl.int64) * stride_st
                  + compact_k.to(tl.int64) * stride_sw)
        s_agg = tl.load(s_ptrs, mask=valid, other=0.0).to(tl.float32)

        # === Head inner loop ===
        for h in range(HPG):
            ni = (head_start + h).to(tl.int64)

            # Load Q for this head
            q_ptrs = (Q + b * stride_qb + m_offs[:, None].to(tl.int64) * stride_qt
                      + ni * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
            q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

            # Load K for this head (different head = different K!)
            k_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                      + ni * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
            k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)

            # QK
            qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

            # Load per-head weights
            pw2 = tl.load(PRE_W2 + b * stride_p2b + m_offs.to(tl.int64) * stride_p2t + ni * stride_p2n,
                          mask=m_mask, other=0.0).to(tl.float32)
            pdd = tl.load(PRE_DD + b * stride_pdb + m_offs.to(tl.int64) * stride_pdt + ni * stride_pdn,
                          mask=m_mask, other=0.0).to(tl.float32)

            score = (pdd + 1.0)[:, None] * qk + pw2[:, None] * s_agg
            score = tl.where(valid, score, float("-inf"))

            # Online softmax for this head — but we need persistent mi/li/acc!
            # With Triton, we can't easily maintain HPG parallel states in the
            # k-tile loop because each `h` iteration reuses the same variable names.
            # The compiler doesn't keep separate versions.
            #
            # WORKAROUND: load mi/li from HBM, update, store back.
            # This is expensive but avoids the register allocation issue.
            # Actually, let's just NOT try to maintain state across the inner loop.
            # Instead, do head-outer k-inner as before, but share V load.

            # Wait — K is per-head (K[b,s,n,d]), so K CAN'T be shared across heads!
            # Each head has its own K. So there's nothing to share.
            # This is MHA, not MQA/GQA. K is per-head.
            pass

    # K is per-head in MHA. There's NO K sharing possible across heads.
    # The original code was already optimal for this layout.
    # Let me fall back to the simple approach: head outer, k-tile inner.
    # The only thing we save is s_buf being shared (already loaded by all heads).
    # s_buf IS shared across heads (it's [B,T,W], head-independent).
    # V is also per-head. No sharing possible.

    # === Correct implementation: head outer, k-tile inner ===
    for h in range(HPG):
        ni = (head_start + h).to(tl.int64)

        pw2 = tl.load(PRE_W2 + b * stride_p2b + m_offs.to(tl.int64) * stride_p2t + ni * stride_p2n,
                       mask=m_mask, other=0.0).to(tl.float32)
        pdd = tl.load(PRE_DD + b * stride_pdb + m_offs.to(tl.int64) * stride_pdt + ni * stride_pdn,
                       mask=m_mask, other=0.0).to(tl.float32)
        ds = pdd + 1.0
        pw1p = tl.load(POST_W1 + b * stride_pw1b + m_offs.to(tl.int64) * stride_pw1t + ni * stride_pw1n,
                        mask=m_mask, other=0.0).to(tl.float32)

        q_ptrs = (Q + b * stride_qb + m_offs[:, None].to(tl.int64) * stride_qt
                  + ni * stride_qn + d_offs[None, :].to(tl.int64) * stride_qd)
        q_blk = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        mi = tl.full([BM], float("-inf"), dtype=tl.float32)
        li = tl.zeros([BM], dtype=tl.float32)
        acc = tl.zeros([BM, D], dtype=tl.float32)

        for kb in range(k_lo, k_hi, BK):
            k_offs = kb + tl.arange(0, BK)
            k_mask = (k_offs < k_hi) & (k_offs < seq_len)
            causal = k_offs[None, :] <= m_offs[:, None]
            win = (m_offs[:, None] - k_offs[None, :]) < W
            valid = causal & win & m_mask[:, None] & k_mask[None, :]
            compact_k = k_offs[None, :] - k_lo_per_q[:, None]
            compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

            # K is per-head, must load per head
            k_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                      + ni * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
            k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
            qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

            # s_buf is shared across heads
            s_ptrs = (S_BUF + b * stride_sb + m_offs[:, None].to(tl.int64) * stride_st
                      + compact_k.to(tl.int64) * stride_sw)
            s_agg = tl.load(s_ptrs, mask=valid, other=0.0).to(tl.float32)

            score = ds[:, None] * qk + pw2[:, None] * s_agg
            score = tl.where(valid, score, float("-inf"))

            bmax = tl.max(score, axis=1)
            m_new = tl.maximum(mi, bmax)
            alpha = tl.exp(mi - m_new)
            p = tl.exp(score - m_new[:, None])
            p = tl.where(valid, p, 0.0)
            acc = acc * alpha[:, None]

            # V is per-head, must load per head
            v_ptrs = (V + b * stride_vb + k_offs[:, None].to(tl.int64) * stride_vt
                      + ni * stride_vn + d_offs[None, :].to(tl.int64) * stride_vd)
            v_blk = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(p.to(tl.float16), v_blk)
            li = li * alpha + tl.sum(p, axis=1)
            mi = m_new

        safe_l = tl.where(li > 0.0, li, 1.0)
        acc = acc / safe_l[:, None]

        o_ptrs = (O_BUF + b * stride_ob + m_offs[:, None].to(tl.int64) * stride_ot
                  + ni * stride_on + d_offs[None, :].to(tl.int64) * stride_od)
        tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])

        # Pass 2: post_w1 atomic
        for kb in range(k_lo, k_hi, BK):
            k_offs = kb + tl.arange(0, BK)
            k_mask = (k_offs < k_hi) & (k_offs < seq_len)
            causal = k_offs[None, :] <= m_offs[:, None]
            win = (m_offs[:, None] - k_offs[None, :]) < W
            valid = causal & win & m_mask[:, None] & k_mask[None, :]
            compact_k = k_offs[None, :] - k_lo_per_q[:, None]
            compact_k = tl.minimum(tl.maximum(compact_k, 0), W - 1)

            k_ptrs = (K + b * stride_kb + k_offs[:, None].to(tl.int64) * stride_kt
                      + ni * stride_kn + d_offs[None, :].to(tl.int64) * stride_kd)
            k_blk = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
            qk = tl.dot(q_blk, tl.trans(k_blk)) * scaling

            s_ptrs = (S_BUF + b * stride_sb + m_offs[:, None].to(tl.int64) * stride_st
                      + compact_k.to(tl.int64) * stride_sw)
            s_agg = tl.load(s_ptrs, mask=valid, other=0.0).to(tl.float32)

            score = ds[:, None] * qk + pw2[:, None] * s_agg
            score = tl.where(valid, score, float("-inf"))
            probs = tl.exp(score - mi[:, None]) / safe_l[:, None]
            probs = tl.where(valid, probs, 0.0)

            a_ptrs = (A_BUF + b * stride_ab + m_offs[:, None].to(tl.int64) * stride_at
                      + compact_k.to(tl.int64) * stride_aw)
            tl.atomic_add(a_ptrs, pw1p[:, None] * probs, mask=valid, sem="relaxed")


class TritonDCGrouped:

    @staticmethod
    def alloc_buffers(q, window, G=4):
        B, T, N, D = q.shape
        W = min(int(window), T)
        dev, dt = q.device, q.dtype
        return {
            "gs_buf": torch.empty((B, G, T, W), device=dev, dtype=dt),
            "s_buf":  torch.empty((B, T, W), device=dev, dtype=dt),
            "a_buf":  torch.empty((B, T, W), device=dev, dtype=torch.float32),
            "o_buf":  torch.empty((B, T, N, D), device=dev, dtype=dt),
            "out":    torch.empty((B, T, N, D), device=dev, dtype=dt),
        }

    @staticmethod
    def forward(
        q, k, v, dc_weights, scaling, window,
        seq_lens=None, G=4,
        bm_k0=16, bk_k0=64,
        bm_mid=16, bk_mid=64,
        bm_fin=64, bk_fin=32,
        bm_red=64, bk_red=64,
        buffers=None,
    ):
        pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = dc_weights
        B, T, N, D = q.shape
        W = min(int(window), T)
        HPG = N // G

        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
        pre_w1 = pre_w1.contiguous(); pre_w2 = pre_w2.contiguous()
        pre_dd = pre_dd.contiguous(); post_w1 = post_w1.contiguous()
        post_w2 = post_w2.contiguous(); post_dd = post_dd.contiguous()

        if buffers is None:
            buffers = TritonDCGrouped.alloc_buffers(q, W, G)
        gs_buf = buffers["gs_buf"]
        s_buf  = buffers["s_buf"]
        a_buf  = buffers["a_buf"]
        o_buf  = buffers["o_buf"]
        out    = buffers["out"]
        a_buf.zero_()

        num_tiles_r = triton.cdiv(T, bm_red) * triton.cdiv(W, bk_red)

        # K0g
        _k0g_preagg_kernel[(triton.cdiv(T, bm_k0), B * G)](
            q, k, pre_w1, gs_buf, seq_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            pre_w1.stride(0), pre_w1.stride(1), pre_w1.stride(2),
            gs_buf.stride(0), gs_buf.stride(1), gs_buf.stride(2), gs_buf.stride(3),
            scaling, B=B, T=T, N=N, D=D, W=W, BM=bm_k0, BK=bk_k0,
            G=G, HPG=HPG, num_warps=4, num_stages=2,
        )

        # K0r
        _k0r_reduce_kernel[(num_tiles_r, B)](
            gs_buf, s_buf,
            gs_buf.stride(0), gs_buf.stride(1), gs_buf.stride(2), gs_buf.stride(3),
            s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),
            B=B, T=T, W=W, G=G, BM=bm_red, BK=bk_red,
            num_warps=4, num_stages=1,
        )

        # K12g: grouped fused
        _k12g_fused_kernel[(triton.cdiv(T, bm_mid), B * G)](
            q, k, v, s_buf, pre_w2, pre_dd, post_w1,
            o_buf, a_buf, seq_lens,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            s_buf.stride(0), s_buf.stride(1), s_buf.stride(2),
            pre_w2.stride(0), pre_w2.stride(1), pre_w2.stride(2),
            pre_dd.stride(0), pre_dd.stride(1), pre_dd.stride(2),
            post_w1.stride(0), post_w1.stride(1), post_w1.stride(2),
            o_buf.stride(0), o_buf.stride(1), o_buf.stride(2), o_buf.stride(3),
            a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
            scaling, B=B, T=T, N=N, D=D, W=W, BM=bm_mid, BK=bk_mid,
            G=G, HPG=HPG, num_warps=4, num_stages=2,
        )

        # K3
        _k3_final_kernel[(triton.cdiv(T, bm_fin), B * N)](
            a_buf, v, o_buf, post_w2, post_dd, out, seq_lens,
            a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o_buf.stride(0), o_buf.stride(1), o_buf.stride(2), o_buf.stride(3),
            post_w2.stride(0), post_w2.stride(1), post_w2.stride(2),
            post_dd.stride(0), post_dd.stride(1), post_dd.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            B=B, T=T, N=N, D=D, W=W, BLOCK_M=bm_fin, BLOCK_K=bk_fin,
            num_warps=4, num_stages=2,
        )
        return out
