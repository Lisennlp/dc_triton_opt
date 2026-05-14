"""Python wrapper for the CUDA DC one-kernel V4 benchmark path."""

from __future__ import annotations

import torch


class CudaDCOneKernelV4:
    """CUDA cache-four-QK version of DC one-kernel V4.

    Supported benchmark cases:
    - q/k/v and DC weights are contiguous fp16 tensors
    - shape [B, T, 32, 128]
    - G is 4 or 8
    - chunk_size is 16 or 32
    - next_power_of_2(chunk_size + window - 1) <= 128
    """

    @staticmethod
    def forward(
        q, k, v, dc_weights, scaling, window,
        seq_lens=None, G=8, chunk_size=16,
    ):
        pre_w1, pre_w2, pre_dd, post_w1, post_w2, post_dd = dc_weights
        B, T, N, D = q.shape

        if q.dtype is not torch.float16:
            raise NotImplementedError("CUDA V4 only supports fp16")
        if N != 32 or D != 128:
            raise NotImplementedError("CUDA V4 only supports N=32, D=128")
        if G not in (4, 8):
            raise NotImplementedError("CUDA V4 cache4 benchmark path supports G=4 or G=8")
        if chunk_size not in (16, 32):
            raise NotImplementedError("CUDA V4 supports chunk_size=16 or 32")

        W = min(int(window), T)
        kspan = int(chunk_size) + W - 1
        kl = 1 << (kspan - 1).bit_length()
        if kl > 128:
            raise NotImplementedError("CUDA V4 supports KL<=128")

        if seq_lens is None:
            seq_lens = torch.full((B,), T, device=q.device, dtype=torch.int32)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        pre_w1 = pre_w1.contiguous()
        pre_w2 = pre_w2.contiguous()
        pre_dd = pre_dd.contiguous()
        post_w1 = post_w1.contiguous()
        post_w2 = post_w2.contiguous()
        post_dd = post_dd.contiguous()
        seq_lens = seq_lens.contiguous()

        import dc_fused_cuda

        return dc_fused_cuda.onekernel_v4_forward(
            q, k, v,
            pre_w1, pre_w2, pre_dd,
            post_w1, post_w2, post_dd,
            seq_lens,
            float(scaling),
            int(W),
            int(G),
            int(chunk_size),
        )
