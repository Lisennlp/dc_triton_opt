"""Profile V3 kernel with Nsight Compute to check register spill.

Run (lightweight, just spill + regs):
  CUDA_VISIBLE_DEVICES=2 ncu --metrics \
    launch__registers_per_thread,\
    l1tex__data_pipe_lsu_wavefronts_mem_local_op_ld.sum,\
    l1tex__data_pipe_lsu_wavefronts_mem_local_op_st.sum \
    python profile_ncu.py

Run (full report):
  CUDA_VISIBLE_DEVICES=2 ncu --set full -o profile_v3 python profile_ncu.py
"""

import math
import torch
from triton_dc_onekernel_v3 import TritonDCOneKernel as V3

N, D = 32, 128
sc = 1.0 / math.sqrt(D)

# Test both KL=128 and KL=256 configs
configs = [
    (16, 4096, 112, 16, 16),   # BM=16, W=112, KL=128, G=16
    (16, 4096, 240, 16, 16),   # BM=16, W=240, KL=256, G=16
]

for B, T, W, G, BM in configs:
    print(f"\n=== B={B} T={T} W={W} G={G} BM={BM} KL={BM+W} ===")
    q = torch.randn(B, T, N, D, device='cuda', dtype=torch.float16).contiguous()
    k = torch.randn(B, T, N, D, device='cuda', dtype=torch.float16).contiguous()
    v = torch.randn(B, T, N, D, device='cuda', dtype=torch.float16).contiguous()
    ws = tuple(torch.randn(B, T, N, device='cuda', dtype=torch.float16).contiguous() for _ in range(6))
    sl = torch.full((B,), T, device='cuda', dtype=torch.int32)

    # Warmup (triggers autotune + compilation)
    for _ in range(3):
        V3.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM)
    torch.cuda.synchronize()

    # Profiled run
    V3.forward(q, k, v, ws, sc, W, sl, G=G, chunk_size=BM)
    torch.cuda.synchronize()
    print("  Kernel launched. Check ncu output above.")

print("\nKey metrics:")
print("  launch__registers_per_thread  — should be <= 255")
print("  local_op_ld.sum = 0           — no register spill reads")
print("  local_op_st.sum = 0           — no register spill writes")
