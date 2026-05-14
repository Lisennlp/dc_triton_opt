"""Exhaustive search over all tile/warp params for 3-kernel fused DC."""
import math, time, torch, itertools
from flash_attn import flash_attn_func
from dc_attention_torch import dc_attention_window_chunked_residual
from triton_dc_fused import TritonDCResidualFused
from triton_dc_residual import TritonDCResidual

device = "cuda"; dtype = torch.float16
N, D, W = 32, 128, 256
sc = 1.0 / math.sqrt(D)

def make(B, T):
    q = torch.randn(B, T, N, D, device=device, dtype=dtype)
    k = torch.randn(B, T, N, D, device=device, dtype=dtype)
    v = torch.randn(B, T, N, D, device=device, dtype=dtype)
    ws = tuple(torch.randn(B, T, N, device=device, dtype=dtype) for _ in range(6))
    sl = torch.full((B,), T, device=device, dtype=torch.int32)
    return q, k, v, ws, sl

def bench(fn, warmup=10, repeat=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1e6

# ── Correctness ──
print("Correctness check:")
for Bc, Tc in [(1, 512), (2, 1024)]:
    q, k, v, ws, sl = make(Bc, Tc)
    ref = dc_attention_window_chunked_residual(q, k, v, ws, sc, W, sl, 256)
    fused = TritonDCResidualFused.forward(q, k, v, ws, sc, W, sl)
    diff = (fused.permute(0,2,1,3).float() - ref.float()).abs()
    print(f"  B={Bc} T={Tc}: max={diff.max().item():.4e} mean={diff.mean().item():.4e}")
    del q,k,v,ws,sl,ref,fused; torch.cuda.empty_cache()

q, k, v, ws, sl = make(16, 4096)
opt_out = TritonDCResidual.forward(q, k, v, ws, sc, W, sl)
fused_out = TritonDCResidualFused.forward(q, k, v, ws, sc, W, sl)
diff = (fused_out.float() - opt_out.float()).abs()
print(f"  B=16 T=4096 vs Opt: max={diff.max().item():.4e} mean={diff.mean().item():.4e}")

# ── References ──
bufs_o = TritonDCResidual.alloc_buffers(q, W)
us_opt = bench(lambda: TritonDCResidual.forward(q, k, v, ws, sc, W, sl, buffers=bufs_o))
us_faw = bench(lambda: flash_attn_func(q, k, v, softmax_scale=sc, causal=True, window_size=(W-1, 0)))
us_faf = bench(lambda: flash_attn_func(q, k, v, softmax_scale=sc, causal=True))
print(f"\nOpt(4k): {us_opt:.0f} us ({us_opt/us_faw:.2f}x FA2-w, {us_opt/us_faf:.2f}x FA2-f)")
print(f"FA2-w:   {us_faw:.0f} us")
print(f"FA2-f:   {us_faf:.0f} us")

# ── Grid search ──
print(f"\nGrid search (B=16 T=4096):")
bufs_f = TritonDCResidualFused.alloc_buffers(q, W)
best_us = float('inf')
best_cfg = None

configs = list(itertools.product(
    [(64,64), (64,32), (32,64)],     # K0: (bm, bk)
    [(16,64), (16,32), (32,32)],     # K12f: (bm, bk)
    [(64,32), (32,32), (64,64)],     # K3: (bm, bk)
    [4, 8],                           # num_warps for K0
    [4],                              # num_warps for K12f
    [4],                              # num_warps for K3
))

for cfg in configs:
    (bm0,bk0), (bmm,bkm), (bmf,bkf), nw0, nwm, nwf = cfg
    try:
        us = bench(lambda: TritonDCResidualFused.forward(
            q, k, v, ws, sc, W, sl,
            bm_k0=bm0, bk_k0=bk0, bm_mid=bmm, bk_mid=bkm,
            bm_fin=bmf, bk_fin=bkf, nw_k0=nw0, nw_mid=nwm, nw_fin=nwf,
            buffers=bufs_f), warmup=5, repeat=20)
        tag = ""
        if us < best_us:
            best_us = us
            best_cfg = cfg
            tag = " ★"
        if us < us_opt:
            print(f"  K0=({bm0},{bk0}) K12=({bmm},{bkm}) K3=({bmf},{bkf}) nw=({nw0},{nwm},{nwf})"
                  f" → {us:.0f} us ({us/us_faw:.2f}x FA2-w){tag}")
    except Exception:
        pass

print(f"\nBest fused: {best_us:.0f} us ({best_us/us_faw:.2f}x FA2-w, {best_us/us_faf:.2f}x FA2-f)")
print(f"Best cfg: {best_cfg}")
print(f"Speedup vs Opt: {us_opt/best_us:.2f}x")

# ── Full benchmark with best config ──
(bm0,bk0), (bmm,bkm), (bmf,bkf), nw0, nwm, nwf = best_cfg
print(f"\nFull benchmark with best config:")
hdr = f"{'B':>3} {'T':>5} {'B*T':>7} | {'Fused':>8} {'Opt':>8} {'FA2-w':>8} {'FA2-f':>8} | {'F/FA2w':>7} {'O/FA2w':>7} {'F/FA2f':>7}"
print(hdr)
print("-" * len(hdr))
del q,k,v,ws,sl; torch.cuda.empty_cache()

for Bb, Tb in [(16, 4096), (32, 2048), (32, 4096), (64, 2048), (64, 4096)]:
    q, k, v, ws, sl = make(Bb, Tb)
    bf = TritonDCResidualFused.alloc_buffers(q, W)
    bo = TritonDCResidual.alloc_buffers(q, W)
    uf = bench(lambda: TritonDCResidualFused.forward(q, k, v, ws, sc, W, sl,
        bm_k0=bm0, bk_k0=bk0, bm_mid=bmm, bk_mid=bkm,
        bm_fin=bmf, bk_fin=bkf, nw_k0=nw0, nw_mid=nwm, nw_fin=nwf, buffers=bf))
    uo = bench(lambda: TritonDCResidual.forward(q, k, v, ws, sc, W, sl, buffers=bo))
    fw = bench(lambda: flash_attn_func(q, k, v, softmax_scale=sc, causal=True, window_size=(W-1, 0)))
    ff = bench(lambda: flash_attn_func(q, k, v, softmax_scale=sc, causal=True))
    print(f"{Bb:3d} {Tb:5d} {Bb*Tb:7d} | {uf:7.0f}u {uo:7.0f}u {fw:7.0f}u {ff:7.0f}u | "
          f"{uf/fw:6.2f}x {uo/fw:6.2f}x {uf/ff:6.2f}x")
    del q,k,v,ws,sl,bf,bo; torch.cuda.empty_cache()
