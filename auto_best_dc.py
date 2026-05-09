from __future__ import annotations

import torch

from triton_atg_dc import TritonDCRank1D0Atg, T_THRESH
from triton_3k_dc_rank1_group import TritonDCRank1_3K_Grouped



class TritonDCRank1D0BestGraph(object):
    """Auto-dispatch D0 CUDA Graph: picks the fastest pipeline per (B,T).

    Small tokens (T<=1024 AND B*T<=BT_CROSSOVER):
        → AtG: atomic K1+K2+K3 with [B,T,T] buffers

    Large tokens (otherwise):
        → 3K-Grouped: compact [B,G,T,W] fp16 buffers
    """

    BT_CROSSOVER = 4096

    def __init__(self, B, T, N, D, W, scaling,
                 n_groups=1, block_m_atg=16, block_k_atg=32,
                 block_m_3k=32, block_k_3k=None, impl="auto"):
        device = 'cuda'
        dtype = torch.float16
        self.B, self.T, self.N, self.D, self.W = B, T, N, D, W
        self.scaling = scaling

        auto_use_atg = (T <= T_THRESH and B * T <= self.BT_CROSSOVER)
        if impl == "auto":
            self.use_atg = auto_use_atg
        elif impl == "atg":
            self.use_atg = True
        elif impl == "3k":
            self.use_atg = False
        else:
            raise ValueError("impl must be one of: auto, atg, 3k")

        self.q_s = torch.randn(B, T, N, D, device=device, dtype=dtype)
        self.k_s = torch.randn(B, T, N, D, device=device, dtype=dtype)
        self.v_s = torch.randn(B, T, N, D, device=device, dtype=dtype)
        self.pw1_pre_s = torch.randn(B, T, N, device=device, dtype=dtype)
        self.pw2_pre_s = torch.randn(B, T, N, device=device, dtype=dtype)
        self.pw1_post_s = torch.randn(B, T, N, device=device, dtype=dtype)
        self.pw2_post_s = torch.randn(B, T, N, device=device, dtype=dtype)
        self.out_s = torch.empty(B, T, N, D, device=device, dtype=dtype)

        if self.use_atg:
            self.bm, self.bk = block_m_atg, block_k_atg
            use_atomic = (T <= T_THRESH)
            if use_atomic:
                self.agg_s = torch.zeros(B, T, T, device=device,
                                         dtype=torch.float32)
                self.agg_post_s = torch.zeros(B, T, T, device=device,
                                              dtype=torch.float32)
            else:
                self.agg_s = torch.full((B, T, T), float('-inf'),
                                        device=device, dtype=dtype)
                self.agg_post_s = torch.zeros(B, T, T, device=device,
                                              dtype=dtype)
        else:
            self.G = n_groups
            self.H = N // n_groups
            assert N % n_groups == 0
            self.bm = block_m_3k
            self.bk = block_k_3k
            if self.bk is None:
                self.bk = 64 if B * T < 163840 else 128
            self.s_buf_s = torch.empty(B, self.G, T, W, device=device,
                                       dtype=dtype)
            self.m_buf_s = torch.full((B, T, N), float('-inf'),
                                      device=device, dtype=torch.float32)
            self.l_buf_s = torch.zeros(B, T, N, device=device,
                                       dtype=torch.float32)
            self.u_buf_s = torch.empty(B, self.G, T, W, device=device,
                                       dtype=dtype)

        self._run_pipeline()
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._run_pipeline()
        torch.cuda.synchronize()

    def _run_pipeline(self):
        if self.use_atg:
            TritonDCRank1D0Atg.forward(
                self.q_s, self.k_s, self.v_s,
                self.pw1_pre_s, self.pw2_pre_s,
                self.pw1_post_s, self.pw2_post_s,
                self.scaling, self.W,
                block_m=self.bm, block_k=self.bk,
                agg=self.agg_s, agg_post=self.agg_post_s, out=self.out_s)
        else:
            TritonDCRank1_3K_Grouped.forward(
                self.q_s, self.k_s, self.v_s,
                self.pw1_pre_s, self.pw2_pre_s,
                self.pw1_post_s, self.pw2_post_s,
                self.scaling, self.W,
                n_groups=self.G, block_m=self.bm, block_k=self.bk,
                s_buf=self.s_buf_s, m_buf=self.m_buf_s,
                l_buf=self.l_buf_s, u_buf=self.u_buf_s, out=self.out_s)

    def __call__(self, q, k, v, pw1_pre, pw2_pre, pw1_post, pw2_post):
        self.q_s.copy_(q)
        self.k_s.copy_(k)
        self.v_s.copy_(v)
        self.pw1_pre_s.copy_(pw1_pre)
        self.pw2_pre_s.copy_(pw2_pre)
        self.pw1_post_s.copy_(pw1_post)
        self.pw2_post_s.copy_(pw2_post)
        self.graph.replay()
        return self.out_s

    def replay_only(self):
        self.graph.replay()
