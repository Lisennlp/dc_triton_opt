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


def dc_attention_decomposed(
    q: torch.Tensor,       # [B, T, N, D]
    k: torch.Tensor,       # [B, T, N, D]
    v: torch.Tensor,       # [B, T, N, D]
    pre_w: torch.Tensor,   # [B, T, N, N]
    post_w: torch.Tensor,  # [B, T, N, N]
    scaling: float,
    window: int,
    seq_lens: torch.Tensor,  # [B]
) -> torch.Tensor:
    """DC attention using pure PyTorch matmuls — all ops use tensor cores via cuBLAS."""
    B, T, N, D = q.shape

    # --- Step 1: QK^T (per-head, cuBLAS) ---
    # [B, T, N, D] -> [B, N, T, D]
    q_nt = q.transpose(1, 2)  # [B, N, T, D]
    k_nt = k.transpose(1, 2)  # [B, N, T, D]
    v_nt = v.transpose(1, 2)  # [B, N, T, D]

    # [B, N, T, D] @ [B, N, D, T] -> [B, N, T, T]
    logits = torch.matmul(q_nt, k_nt.transpose(-1, -2)) * scaling

    # --- Step 2: Pre-mix (cross-head, batched matmul) ---
    # logits: [B, N, T_q, T_k] -> rearrange for per-query mixing
    # For each query position q: pre_w[b, q, :, :] @ logits[b, :, q, :]
    #   = [N, N] @ [N, T_k] -> [N, T_k]
    # Batch over (B, T_q): [B*T, N, N] @ [B*T, N, T] -> [B*T, N, T]
    logits_qt = logits.permute(0, 2, 1, 3).contiguous()  # [B, T_q, N, T_k]
    pre_w_bt = pre_w.reshape(B * T, N, N)                # [B*T, N, N]
    logits_bt = logits_qt.reshape(B * T, N, T)           # [B*T, N, T_k]

    pre_mixed = torch.bmm(pre_w_bt, logits_bt)  # [B*T, N, T_k]
    pre_mixed = pre_mixed.reshape(B, T, N, T)   # [B, T_q, N, T_k]

    # --- Step 3: Causal mask + window mask + softmax ---
    q_idx = torch.arange(T, device=q.device)
    k_idx = torch.arange(T, device=q.device)
    causal = k_idx[None, :] <= q_idx[:, None]           # [T_q, T_k]
    win_mask = (q_idx[:, None] - k_idx[None, :]) < window  # [T_q, T_k]

    # seq_lens mask: k_idx < seq_lens[b]
    seq_mask = k_idx[None, :] < seq_lens[:, None]  # [B, T_k]

    # Combine masks
    mask = causal[None, None, :, :] & win_mask[None, None, :, :]  # [1, 1, T_q, T_k]
    mask = mask & seq_mask[:, None, None, :]  # [B, 1, T_q, T_k]

    # pre_mixed: [B, T_q, N, T_k] -> [B, N, T_q, T_k] for masking
    pre_mixed_nt = pre_mixed.permute(0, 2, 1, 3)  # [B, N, T_q, T_k]
    pre_mixed_nt = pre_mixed_nt.masked_fill(~mask, float('-inf'))

    probs = torch.softmax(pre_mixed_nt, dim=-1)  # [B, N, T_q, T_k]
    probs = torch.nan_to_num(probs, nan=0.0)

    # --- Step 4: Post-mix (cross-head, batched matmul) ---
    # Same structure as pre-mix but on probs
    probs_qt = probs.permute(0, 2, 1, 3).contiguous()  # [B, T_q, N, T_k]
    post_w_bt = post_w.reshape(B * T, N, N)             # [B*T, N, N]
    probs_bt = probs_qt.reshape(B * T, N, T)            # [B*T, N, T_k]

    post_mixed = torch.bmm(post_w_bt, probs_bt)  # [B*T, N, T_k]
    post_mixed = post_mixed.reshape(B, T, N, T)  # [B, T_q, N, T_k]
    post_mixed_nt = post_mixed.permute(0, 2, 1, 3)  # [B, N, T_q, T_k]

    # --- Step 5: P@V (per-head, cuBLAS) ---
    # [B, N, T_q, T_k] @ [B, N, T_k, D] -> [B, N, T_q, D]
    out_nt = torch.matmul(post_mixed_nt, v_nt)  # [B, N, T, D]

    # Back to [B, T, N, D]
    return out_nt.transpose(1, 2)


def dc_attention_decomposed_chunked(
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
    """Chunked decomposed DC attention over the query sequence dimension.

    This keeps K/V full length but only materializes [B, N, chunk, T] attention
    tiles at a time. It is useful for sliding-window shapes where full [T, T]
    materialization costs more memory bandwidth than needed.
    """
    B, T, N, D = q.shape
    chunk_size = max(1, min(int(chunk_size), T))

    k_nt = k.transpose(1, 2)  # [B, N, T, D]
    v_nt = v.transpose(1, 2)  # [B, N, T, D]
    k_t = k_nt.transpose(-1, -2)

    k_idx = torch.arange(T, device=q.device)
    seq_mask = k_idx[None, :] < seq_lens[:, None]  # [B, T]

    out_chunks = []
    for q_start in range(0, T, chunk_size):
        q_end = min(q_start + chunk_size, T)
        C = q_end - q_start

        q_chunk = q[:, q_start:q_end].transpose(1, 2)  # [B, N, C, D]
        logits = torch.matmul(q_chunk, k_t) * scaling  # [B, N, C, T]

        logits_qt = logits.permute(0, 2, 1, 3).contiguous()  # [B, C, N, T]
        pre_w_bt = pre_w[:, q_start:q_end].reshape(B * C, N, N)
        logits_bt = logits_qt.reshape(B * C, N, T)
        pre_mixed = torch.bmm(pre_w_bt, logits_bt).reshape(B, C, N, T)

        q_idx = torch.arange(q_start, q_end, device=q.device)
        causal = k_idx[None, :] <= q_idx[:, None]
        win_mask = (q_idx[:, None] - k_idx[None, :]) < window
        mask = causal[None, None, :, :] & win_mask[None, None, :, :]
        mask = mask & seq_mask[:, None, None, :]

        pre_mixed_nt = pre_mixed.permute(0, 2, 1, 3).masked_fill(~mask, float('-inf'))
        probs = torch.softmax(pre_mixed_nt, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)

        probs_qt = probs.permute(0, 2, 1, 3).contiguous()  # [B, C, N, T]
        post_w_bt = post_w[:, q_start:q_end].reshape(B * C, N, N)
        probs_bt = probs_qt.reshape(B * C, N, T)
        post_mixed = torch.bmm(post_w_bt, probs_bt).reshape(B, C, N, T)
        post_mixed_nt = post_mixed.permute(0, 2, 1, 3)

        out_chunk = torch.matmul(post_mixed_nt, v_nt)  # [B, N, C, D]
        out_chunks.append(out_chunk.transpose(1, 2))

    return torch.cat(out_chunks, dim=1)


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