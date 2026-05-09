"""DC benchmark: D0-Best vs D1(PreQPostOut) vs baselines vs FA2.

D0-Best  = auto-dispatch AtG/3K-Grouped CUDA Graph
D1       = TritonDCRank1PreQPostOut (pre-mix Q, post-mix output)
WinChunk = dc_attention_decomposed_window_chunked
Compile  = torch.compile(WinChunk)
"""
import torch, time, sys, math
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64
sys.path.insert(0, '/home/lishengping/dc_triton_test')

from auto_best_dc import TritonDCRank1D0BestGraph
from triton_dc_PostAfterPV import TritonDCRank1PreQPostOut
from dc_attention_torch import dc_attention_decomposed_window_chunked
from flash_attn import flash_attn_func

device = 'cuda'
dtype = torch.float16
W = 256
CHUNK = W
warmup, repeat = 15, 50

decomp_compiled = torch.compile(dc_attention_decomposed_window_chunked)

def make(B, T, N=32, D=128):
    q = torch.randn(B, T, N, D, device=device, dtype=dtype)
    k = torch.randn(B, T, N, D, device=device, dtype=dtype)
    v = torch.randn(B, T, N, D, device=device, dtype=dtype)
    pw1 = torch.randn(B, T, N, 1, device=device, dtype=dtype)
    pw2 = torch.randn(B, T, 1, N, device=device, dtype=dtype)
    pw3 = torch.randn(B, T, N, 1, device=device, dtype=dtype)
    pw4 = torch.randn(B, T, 1, N, device=device, dtype=dtype)
    pw1p = pw1.squeeze(-1).contiguous()
    pw2p = pw2.squeeze(2).contiguous()
    pw3p = pw3.squeeze(-1).contiguous()
    pw4p = pw4.squeeze(2).contiguous()
    pre_w = (pw1 @ pw2).contiguous()
    post_w = (pw3 @ pw4).contiguous()
    return q, k, v, pw1, pw2, pw3, pw4, pw1p, pw2p, pw3p, pw4p, pre_w, post_w

def bench(fn):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1e6

def fmt_us(us):
    if us >= 1e18:
        return "     OOM"
    return f"{us:7.0f}\u00b5"

def fmt_ratio(us, ref):
    if us >= 1e18:
        return "    OOM"
    return f"{us/ref:6.2f}\u00d7"

N, D = 32, 128
sc = 1.0 / (D ** 0.5)

configs = [
    (1, 256), (1, 512), (1, 1024), (1, 2048), (1, 4096),
    (4, 256), (4, 512), (4, 1024), (4, 2048), (4, 4096),
    (8, 256), (8, 512), (8, 1024), (8, 2048), (8, 4096),
    (16, 256), (16, 512), (16, 1024), (16, 2048), (16, 4096),
    (32, 256), (32, 512), (32, 1024), (32, 2048), (32, 4096),
    (64, 256), (64, 512), (64, 1024), (64, 2048), (64, 4096),
]

# configs = [
#     (1, 256), (1, 512), (1, 1024), (1, 2048), (1, 4096),
#     (32, 256), (32, 512), (32, 1024), (32, 2048), (32, 4096),
# ]


hdr = (f"{'B':>3s} {'T':>5s} {'B*T':>7s} | "
       f"{'D0-Bst':>8s} {'D1':>8s} {'WinChk':>8s} {'Compile':>8s} | "
       f"{'FA2-w':>8s} {'FA2-f':>8s} | "
       f"{'D0/FA2-w':>7s} {'D1/FA2-w':>7s} {'WC/FA2-w':>7s} {'Cmp/FA2-w':>7s} {'D0/FA2-f':>7s}")
print(hdr)
print("-" * len(hdr))

for B, T in configs:
    tokens = B * T
    q, k, v, pw1, pw2, pw3, pw4, pw1p, pw2p, pw3p, pw4p, pre_w, post_w = make(B, T)
    sl = torch.full((B,), T, device=device, dtype=torch.int32)

    # D0-BestGraph (auto-dispatch)
    try:
        best_eng = TritonDCRank1D0BestGraph(B, T, N, D, W, sc)
        best_eng(q, k, v, pw1p, pw2p, pw3p, pw4p)
        us_d0 = bench(lambda: best_eng.replay_only())
        del best_eng
    except Exception:
        us_d0 = float('inf')

    # D1 - PreQPostOut
    try:
        TritonDCRank1PreQPostOut.forward(
            q, k, v, pw1, pw2, pw3, pw4, sc, W, sl)
        us_d1 = bench(lambda: TritonDCRank1PreQPostOut.forward(
            q, k, v, pw1, pw2, pw3, pw4, sc, W, sl))
    except Exception:
        us_d1 = float('inf')
        torch.cuda.empty_cache()

    # Window-chunked baseline
    try:
        dc_attention_decomposed_window_chunked(q, k, v, pre_w, post_w, sc, W, sl, CHUNK)
        us_wc = bench(lambda: dc_attention_decomposed_window_chunked(
            q, k, v, pre_w, post_w, sc, W, sl, CHUNK))
    except Exception:
        us_wc = float('inf')
        torch.cuda.empty_cache()

    # torch.compile version
    try:
        decomp_compiled(q, k, v, pre_w, post_w, sc, W, sl, CHUNK)
        us_compile = bench(lambda: decomp_compiled(q, k, v, pre_w, post_w, sc, W, sl, CHUNK))
    except Exception:
        us_compile = float('inf')
        torch.cuda.empty_cache()

    # FA2
    us_faw = bench(lambda: flash_attn_func(q, k, v, softmax_scale=sc,
                                            causal=True, window_size=(W-1, 0)))
    us_faf = bench(lambda: flash_attn_func(q, k, v, softmax_scale=sc, causal=True))

    print(f"{B:3d} {T:5d} {tokens:7d} | "
          f"{fmt_us(us_d0)} {fmt_us(us_d1)} {fmt_us(us_wc)} {fmt_us(us_compile)} | "
          f"{fmt_us(us_faw)} {fmt_us(us_faf)} | "
          f"{fmt_ratio(us_d0, us_faw)} {fmt_ratio(us_d1, us_faw)} "
          f"{fmt_ratio(us_wc, us_faw)} {fmt_ratio(us_compile, us_faw)} {fmt_ratio(us_d0, us_faf)}")

    torch.cuda.empty_cache()

print()
print("D0-Bst  = TritonDCRank1D0BestGraph (auto-dispatch AtG/3K-Grp CUDA Graph)")
print("D1      = TritonDCRank1PreQPostOut (pre-mix Q, post-mix output)")
print(f"WinChk  = dc_attention_decomposed_window_chunked (chunk={CHUNK})")
print("Compile = torch.compile(WinChk)")
print(f"FA2-w   = FlashAttention-2 window_size={W}")
print("FA2-f   = FlashAttention-2 full causal")
