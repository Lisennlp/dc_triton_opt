"""Decomposed DC attention prefill using cuBLAS matmuls (tensor cores).

Key insight: QK^T is per-head independent — just like standard MHA.
Only the mixing step needs cross-head communication. So:

  1. QK^T: standard batched matmul [B,N,T,D]@[B,N,D,T] -> [B,N,T,T]  (tensor core)
  2. Pre-mix: [B*T, N, N] @ [B*T, N, T] -> [B*T, N, T]  (tensor core)
  3. Causal mask + softmax
  4. Post-mix: same shape as pre-mix  (tensor core)
  5. P@V: [B,N,T,T] @ [B,N,T,D] -> [B,N,T,D]  (tensor core)

This materializes the full [B,N,T,T] attention matrix (O(T^2) memory),
which is fine for T <= 2048-4096 on 80GB cards.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dc_attention_decomposed_window_chunked(
    q: torch.Tensor,       # [B, T, N, D]
    k: torch.Tensor,       # [B, T, N, D]
    v: torch.Tensor,       # [B, T, N, D]
    pre_w: torch.Tensor,   # [B, T, N, N]
    post_w: torch.Tensor,  # [B, T, N, N]
    scaling: float,
    window: int,
    seq_lens: torch.Tensor,  # [B]
    chunk_size: int,
) -> torch.Tensor:
    """Chunked decomposed DC attention with local K windows.

    For a query chunk [q_start, q_end), keys are restricted to the union window
    [max(0, q_start-window+1), q_end). Attention/softmax are computed only over
    that local K range, not over the full sequence.
    """
    B, T, N, D = q.shape
    chunk_size = max(1, min(int(chunk_size), T))
    window = max(1, min(int(window), T))

    v_nt = v.transpose(1, 2)  # [B, N, T, D]
    out_chunks = []
    for q_start in range(0, T, chunk_size):
        q_end = min(q_start + chunk_size, T)
        C = q_end - q_start
        k_start = max(0, q_start - window + 1)
        k_end = q_end
        Kc = k_end - k_start

        q_chunk = q[:, q_start:q_end].transpose(1, 2)  # [B, N, C, D]
        k_chunk = k[:, k_start:k_end].transpose(1, 2)  # [B, N, Kc, D]
        v_chunk = v_nt[:, :, k_start:k_end]            # [B, N, Kc, D]

        logits = torch.matmul(q_chunk, k_chunk.transpose(-1, -2)) * scaling

        logits_qt = logits.permute(0, 2, 1, 3).contiguous()
        pre_w_bt = pre_w[:, q_start:q_end].reshape(B * C, N, N)
        logits_bt = logits_qt.reshape(B * C, N, Kc)
        pre_mixed = torch.bmm(pre_w_bt, logits_bt).reshape(B, C, N, Kc)

        q_idx = torch.arange(q_start, q_end, device=q.device)
        k_idx = torch.arange(k_start, k_end, device=q.device)
        causal = k_idx[None, :] <= q_idx[:, None]
        win_mask = (q_idx[:, None] - k_idx[None, :]) < window
        seq_mask = k_idx[None, :] < seq_lens[:, None]
        mask = causal[None, None, :, :] & win_mask[None, None, :, :]
        mask = mask & seq_mask[:, None, None, :]

        pre_mixed_nt = pre_mixed.permute(0, 2, 1, 3).masked_fill(~mask, float('-inf'))
        probs = torch.softmax(pre_mixed_nt, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)

        probs_qt = probs.permute(0, 2, 1, 3).contiguous()
        post_w_bt = post_w[:, q_start:q_end].reshape(B * C, N, N)
        probs_bt = probs_qt.reshape(B * C, N, Kc)
        post_mixed = torch.bmm(post_w_bt, probs_bt).reshape(B, C, N, Kc)
        post_mixed_nt = post_mixed.permute(0, 2, 1, 3)

        out_chunk = torch.matmul(post_mixed_nt, v_chunk)
        out_chunks.append(out_chunk.transpose(1, 2))

    return torch.cat(out_chunks, dim=1)



def dc_attention_window_chunked_residual(
    q: torch.Tensor,       # [B, T, N, D]
    k: torch.Tensor,       # [B, T, N, D]
    v: torch.Tensor,       # [B, T, N, D]
    dc_weights: tuple[torch.Tensor, torch.Tensor],   # [B, T, N]
    scaling: float,
    window: int=None,
    seq_lens: torch.Tensor=None,  # [B]
    chunk_size: int=None,
) -> torch.Tensor:
    # all shape is [B, T, N]
    pre_w1, pre_w2, pre_dd,post_w1, post_w2, post_dd = dc_weights
    B, T, N, D = q.shape
    if window is None:
        window = 256
    if seq_lens is None:
        seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)
    if chunk_size is None:
        chunk_size = 256

    out_chunks = []
    for q_start in range(0, T, chunk_size):
        q_end = min(q_start + chunk_size, T)
        k_start = max(q_start - window, 0)
        k_end = q_end

        q_chunk = q[:, q_start:q_end] # btnd
        k_chunk = k[:, k_start:k_end]
        v_chunk = v[:, k_start:k_end]
        
        logits_chunk = torch.einsum('btnd,bsnd->bnts', q_chunk, k_chunk) * scaling # btns

        pre_dd_chunk = pre_dd[:, q_start:q_end] # btn
        logits_pre_dd_chunk = torch.einsum('bnts,btn->bnts', logits_chunk, pre_dd_chunk)

        pre_w1_chunk = pre_w1[:, q_start:q_end] # btn
        pre_w1_logits_chunk = torch.einsum('bnts,btn->bts', logits_chunk, pre_w1_chunk) # bts
        pre_w2_chunk = pre_w2[:, q_start:q_end]
        pre_w2_logits_chunk = torch.einsum('bts,btn->bnts', pre_w1_logits_chunk, pre_w2_chunk)
        
        logits_chunk = logits_chunk + pre_w2_logits_chunk + logits_pre_dd_chunk
        
        # mask and softmax
        q_idx = torch.arange(q_start, q_end, device=q.device)
        k_idx = torch.arange(k_start, k_end, device=q.device)
        causal = k_idx[None, :] <= q_idx[:, None]
        win_mask = (q_idx[:, None] - k_idx[None, :]) < window
        seq_mask = k_idx[None, :] < seq_lens[:, None]
        mask = causal[None, None, :, :] & win_mask[None, None, :, :]
        mask = mask & seq_mask[:, None, None, :] # b1ts
        logits_chunk = logits_chunk.masked_fill(~mask, float('-inf'))
        probs_chunk = torch.softmax(logits_chunk, dim=-1)
        probs_chunk = torch.nan_to_num(probs_chunk, nan=0.0)

        post_dd_chunk = post_dd[:, q_start:q_end] # btn
        post_dd_probs_chunk = torch.einsum('bnts,btn->bnts', probs_chunk, post_dd_chunk)

        post_w1_chunk = post_w1[:, q_start:q_end] # btn
        post_w1_probs_chunk = torch.einsum('bnts,btn->bts', probs_chunk, post_w1_chunk) # bts
        post_w2_chunk = post_w2[:, q_start:q_end]
        post_w2_probs_chunk = torch.einsum('bts,btn->bnts', post_w1_probs_chunk, post_w2_chunk)

        probs_chunk = probs_chunk + post_w2_probs_chunk + post_dd_probs_chunk

        post_w2_probs_chunk = torch.einsum('bnts,bsnd->bntd', probs_chunk, v_chunk)
        
        out_chunks.append(post_w2_probs_chunk)

    return torch.cat(out_chunks, dim=2)


def dc_attention_window_chunked_residual_grouped(
    q: torch.Tensor,       # [B, T, N, D]
    k: torch.Tensor,       # [B, T, N, D]
    v: torch.Tensor,       # [B, T, N, D]
    dc_weights: tuple,     # 6 × [B, T, N]
    scaling: float,
    window: int = None,
    seq_lens: torch.Tensor = None,
    chunk_size: int = None,
    G: int = 1,
) -> torch.Tensor:
    """DC attention with intra-group head mixing.

    Heads are divided into G groups of HPG = N/G heads each.
    Pre/post cross-head mixing (w1/w2) only happens within each group.
    G=1 is equivalent to full cross-head mixing (same as dc_attention_window_chunked_residual).
    """
    pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = dc_weights
    B, T, N, D = q.shape
    HPG = N // G
    if window is None:
        window = 256
    if seq_lens is None:
        seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)
    if chunk_size is None:
        chunk_size = 256

    out_all = torch.empty(B, N, T, D, device=q.device, dtype=q.dtype)

    for g in range(G):
        h_start = g * HPG
        h_end = h_start + HPG

        # Slice heads for this group
        q_g = q[:, :, h_start:h_end, :]          # [B, T, HPG, D]
        k_g = k[:, :, h_start:h_end, :]
        v_g = v[:, :, h_start:h_end, :]
        pw1_g = pre_w1[:, :, h_start:h_end]      # [B, T, HPG]
        pw2_g = pre_w2[:, :, h_start:h_end]
        pdd_g = pre_dd[:, :, h_start:h_end]
        qw1_g = post_w1[:, :, h_start:h_end]
        qw2_g = post_w2[:, :, h_start:h_end]
        qdd_g = post_dd[:, :, h_start:h_end]

        out_chunks = []
        for q_start in range(0, T, chunk_size):
            q_end = min(q_start + chunk_size, T)
            k_start = max(q_start - window, 0)
            k_end = q_end

            q_chunk = q_g[:, q_start:q_end]
            k_chunk = k_g[:, k_start:k_end]
            v_chunk = v_g[:, k_start:k_end]

            # QK per-head: [B, HPG, ql, kl]
            logits = torch.einsum('btnd,bsnd->bnts', q_chunk, k_chunk) * scaling

            # Pre-mix (intra-group only)
            pdd_c = pdd_g[:, q_start:q_end]
            logits_dd = torch.einsum('bnts,btn->bnts', logits, pdd_c)

            pw1_c = pw1_g[:, q_start:q_end]
            pw1_logits = torch.einsum('bnts,btn->bts', logits, pw1_c)
            pw2_c = pw2_g[:, q_start:q_end]
            pw2_logits = torch.einsum('bts,btn->bnts', pw1_logits, pw2_c)

            logits = logits + pw2_logits + logits_dd

            # Mask + softmax
            q_idx = torch.arange(q_start, q_end, device=q.device)
            k_idx = torch.arange(k_start, k_end, device=q.device)
            causal = k_idx[None, :] <= q_idx[:, None]
            win_mask = (q_idx[:, None] - k_idx[None, :]) < window
            seq_mask = k_idx[None, :] < seq_lens[:, None]
            mask = causal[None, None, :, :] & win_mask[None, None, :, :]
            mask = mask & seq_mask[:, None, None, :]
            logits = logits.masked_fill(~mask, float('-inf'))
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)

            # Post-mix (intra-group only)
            qdd_c = qdd_g[:, q_start:q_end]
            probs_dd = torch.einsum('bnts,btn->bnts', probs, qdd_c)

            qw1_c = qw1_g[:, q_start:q_end]
            qw1_probs = torch.einsum('bnts,btn->bts', probs, qw1_c)
            qw2_c = qw2_g[:, q_start:q_end]
            qw2_probs = torch.einsum('bts,btn->bnts', qw1_probs, qw2_c)

            probs = probs + qw2_probs + probs_dd

            # PV
            out_chunk = torch.einsum('bnts,bsnd->bntd', probs, v_chunk)
            out_chunks.append(out_chunk)

        # [B, HPG, T, D]
        out_g = torch.cat(out_chunks, dim=2)
        out_all[:, h_start:h_end, :, :] = out_g

    return out_all
