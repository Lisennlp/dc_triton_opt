import torch, math, time
import dc_fused_cuda
from triton_dc_residual import TritonDCResidual

device = 'cuda'
dtype = torch.float16
N, D, W = 32, 128, 256
sc = 1.0 / math.sqrt(D)
warmup, repeat = 10, 50

configs = [
    (1, 256),
    (1, 1024),
    (1, 4096),
    (4, 4096),
    (16, 4096),
    (1, 16384),
    (1, 65536),
]

def bench(fn):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1e6

print(f"{'B':>3s} {'T':>6s} {'B*T':>7s} | {'Triton':>8s} {'CUDA':>8s} {'CUDA/Tri':>8s}")
print('-' * 55)

for B, T in configs:
    q = torch.randn(B, T, N, D, device=device, dtype=dtype)
    k = torch.randn(B, T, N, D, device=device, dtype=dtype)
    v = torch.randn(B, T, N, D, device=device, dtype=dtype)
    dc = tuple(torch.randn(B, T, N, device=device, dtype=dtype) for _ in range(6))
    sl = torch.full((B,), T, device=device, dtype=torch.int32)

    # Triton
    bufs = TritonDCResidual.alloc_buffers(q, W)
    TritonDCResidual.forward(q, k, v, dc, sc, W, sl, buffers=bufs)
    us_triton = bench(lambda: TritonDCResidual.forward(q, k, v, dc, sc, W, sl, buffers=bufs))

    # CUDA
    dc_fused_cuda.forward(q, k, v, *dc, sl, sc)
    us_cuda = bench(lambda: dc_fused_cuda.forward(q, k, v, *dc, sl, sc))

    ratio = us_cuda / us_triton
    print(f'{B:3d} {T:6d} {B*T:7d} | {us_triton:8.0f} {us_cuda:8.0f} {ratio:8.2f}x')
    del q, k, v, dc, sl, bufs
    torch.cuda.empty_cache()