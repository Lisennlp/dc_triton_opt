"""DC attention with M-matrix formulation — torch.bmm based.

Instead of the rank-1 decomposition (s_buf/a_buf + atomics + K3),
precompute M_pre[B,T,N,N] and M_post[B,T,N,N] and use batched matmul
to apply the cross-head mixing. This uses cuBLAS for the [N,N] matmuls.

Pipeline:
  1. Precompute M_pre, M_post from DC weights  (cheap, O(B*T*N²))
  2. Per-head QK → online softmax → PV          (Triton kernel, same as before)
  3. Apply M_post to PV results via bmm          (cuBLAS, [B*T, N, N] @ [B*T, N, D])
  
For pre-mixing: apply M_pre to per-chunk logits via chunked windowed approach.
Since we need cross-head mixed logits for softmax, K0 is still needed.
But M_pre formulation allows us to skip a_buf/atomic/K3 for post-mix.

Wait — let me reconsider. The post-mix in output space is:
  out[b,n,t,d] = Σ_n' M_post[b,t,n,n'] * pv[b,n',t,d]
  where pv[b,n',t,d] = Σ_s probs[b,n',t,s] * V[b,s,n',d]  (per-head PV using n' V)

This IS just M_post @ pv_per_head (in head dimension)!
Because V uses n' (same head as probs), pv is the standard per-head PV.

Verification: original formula says
  out_n = Σ_s (M_post_n,n' * probs_n') * V_n
But V_n means V at head n (output head). So:
  out_n = Σ_s Σ_n' M_post[n,n'] * probs[n',t,s] * V[s,n,d]
        = Σ_n' M_post[n,n'] * Σ_s probs[n',t,s] * V[s,n,d]
        = Σ_n' M_post[n,n'] * (probs_n' @ V_n)

Here probs_n' @ V_n uses head n' probs but head n V. NOT the same as pv_n'.

So out_n ≠ Σ_n' M_post[n,n'] * pv_n'. The M_post on PV doesn't work.

HOWEVER: look at this differently. The ORIGINAL code does:
  probs_final[n,t,s] = (1+dd_n) * probs[n,t,s] + w2_n * Σ_n'(w1_n' * probs[n',t,s])
  out[n,t,d] = Σ_s probs_final[n,t,s] * V[s,n,d]

V[s,n,d] uses output head n. So:
  out_n = (1+dd_n) * (probs_n @ V_n) + w2_n * Σ_n'(w1_n' * (probs_n' @ V_n))

Now: probs_n' @ V_n is NOT pv_n' (which would be probs_n' @ V_n'). 
It's probs of head n' applied to head n's values.

UNLESS n = n' in which case probs_n @ V_n = pv_n.

So the cross term is: w2_n * Σ_n'≠n (w1_n' * (probs_n' @ V_n)) + w2_n * w1_n * pv_n

This still requires cross-head PV, which means a_buf is necessary.

CONCLUSION: M_post cannot be simply applied to per-head PV results to get
the correct output. The a_buf approach (or equivalent) is REQUIRED.

The only optimization M-matrix formulation provides is a cleaner way to 
express and potentially compute the cross-head mixing, but it doesn't
change the fundamental computational requirements.

Let me instead focus on what we CAN do: use M_pre to replace s_buf
computation, and measure if torch.bmm for the [N,N]@[N,kl] mixing
is faster than the current Triton K0 + s_buf approach.
"""

from __future__ import annotations
import math
import torch
import triton
import triton.language as tl

from triton_dc_residual import TritonDCResidual


class TritonDCMMatrix:
    """DC attention with M_pre via torch.bmm for pre-mixing."""

    @staticmethod
    def forward(q, k, v, dc_weights, scaling, window, seq_lens=None, chunk_size=256):
        pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = dc_weights
        B, T, N, D = q.shape
        W = min(int(window), T)

        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        # Just use the existing TritonDCResidual for now
        return TritonDCResidual.forward(q, k, v, dc_weights, scaling, W, seq_lens)
