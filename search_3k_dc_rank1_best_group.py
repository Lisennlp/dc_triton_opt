"""Search optimal group size G for DC attention given B*T, model_dim, num_heads.

Usage:
    python search_best_group.py --tokens 32768 --model_dim 4096 --num_heads 32
    python search_best_group.py --tokens 65536 --model_dim 4096 --num_heads 32 --window 256
"""
import argparse
import torch
import time
import sys
import math
sys.path.insert(0, '/home/lishengping/dc_triton_test')

from triton_3k_dc_rank1_group import TritonDCRank1_3K_Grouped
from flash_attn import flash_attn_func


def get_divisors(n):
    """All divisors of n, sorted."""
    divs = []
    for i in range(1, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs


def pick_BT(total_tokens, D_head):
    """Pick (B, T) split for benchmarking. Prefer T as a power-of-2 >= 512."""
    candidates = []
    for t_log2 in range(9, 16):  # T from 512 to 32768
        T = 1 << t_log2
        if total_tokens % T == 0:
            B = total_tokens // T
            if B >= 1:
                candidates.append((B, T))
    if not candidates:
        for T in [512, 1024, 2048, 4096]:
            B = max(1, total_tokens // T)
            if B * T == total_tokens:
                candidates.append((B, T))
    if not candidates:
        T = min(2048, total_tokens)
        B = max(1, total_tokens // T)
        candidates = [(B, T)]
    # prefer moderate B (8-32) for stable benchmarks
    candidates.sort(key=lambda bt: abs(bt[0] - 16))
    return candidates[0]


def bench(fn, warmup=15, repeat=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeat * 1e3


def main():
    parser = argparse.ArgumentParser(description="Search best group size G for DC attention")
    parser.add_argument("--tokens", type=int, required=True,
                        help="Total tokens = batch_size × seq_len")
    parser.add_argument("--model_dim", type=int, default=4096,
                        help="Model dimension (default: 4096)")
    parser.add_argument("--num_heads", type=int, default=32,
                        help="Number of attention heads (default: 32)")
    parser.add_argument("--window", type=int, default=256,
                        help="Sliding window size (default: 256)")
    parser.add_argument("--block_m", type=int, default=32,
                        help="Triton BLOCK_M (default: 32)")
    parser.add_argument("--block_k", type=int, default=64,
                        help="Triton BLOCK_K (default: 64)")
    parser.add_argument("--repeat", type=int, default=50,
                        help="Benchmark repetitions (default: 50)")
    args = parser.parse_args()

    N = args.num_heads
    D = args.model_dim // N
    W = args.window
    total = args.tokens
    # B, T = pick_BT(total, D)
    T = 2048
    B = total // T
    scaling = 1.0 / math.sqrt(D)

    device = 'cuda'
    dtype = torch.float16

    print("=" * 70)
    print(f"DC Attention Group Size Search")
    print(f"  total_tokens = {total}  (B={B}, T={T})")
    print(f"  model_dim    = {args.model_dim}  (N={N}, D={D})")
    print(f"  window       = {W}")
    print(f"  tile         = BM={args.block_m}, BK={args.block_k}")
    print("=" * 70)

    q = torch.randn(B, T, N, D, device=device, dtype=dtype)
    k = torch.randn(B, T, N, D, device=device, dtype=dtype)
    v = torch.randn(B, T, N, D, device=device, dtype=dtype)
    pw1 = torch.randn(B, T, N, 1, device=device, dtype=dtype)
    pw2 = torch.randn(B, T, 1, N, device=device, dtype=dtype)
    pw3 = torch.randn(B, T, N, 1, device=device, dtype=dtype)
    pw4 = torch.randn(B, T, 1, N, device=device, dtype=dtype)

    # FA2 baselines
    ms_faw = bench(lambda: flash_attn_func(q, k, v, softmax_scale=scaling,
                                            causal=True, window_size=(W-1, 0)),
                   repeat=args.repeat)
    ms_faf = bench(lambda: flash_attn_func(q, k, v, softmax_scale=scaling,
                                            causal=True),
                   repeat=args.repeat)

    # G=1 means ungrouped (all N heads interact)
    possible_G = get_divisors(N)

    results = []

    for G in possible_G:
        H = N // G
        label = f"G={G:2d} (H={H:2d})"

        ms = bench(lambda G=G: TritonDCRank1_3K_Grouped.forward(
            q, k, v, pw1, pw2, pw3, pw4, scaling, W,
            n_groups=G, block_m=args.block_m, block_k=args.block_k),
            repeat=args.repeat)

        sbuf_mb = B * G * T * W * 2 / 1e6 if G > 1 else B * T * W * 2 / 1e6
        results.append((G, H, ms, sbuf_mb))

    # Sort by latency
    results.sort(key=lambda x: x[2])
    best_G, best_H, best_ms, _ = results[0]

    print(f"\n{'G':>4s} {'H':>4s} | {'latency':>10s} {'vs FA2-w':>9s} {'vs FA2-f':>9s} | {'s_buf MB':>9s} | {'rank':>4s}")
    print("-" * 70)

    for rank, (G, H, ms, sbuf_mb) in enumerate(results, 1):
        vs_faw = ms / ms_faw
        vs_faf = ms / ms_faf
        marker = " ★ BEST" if G == best_G else ""
        print(f"{G:4d} {H:4d} | {ms:8.3f}ms {vs_faw:8.2f}× {vs_faf:8.2f}× | {sbuf_mb:8.1f}MB | #{rank:d}{marker}")

    print("-" * 70)
    print(f"FA2-win{W}:  {ms_faw:.3f}ms")
    print(f"FA2-full:   {ms_faf:.3f}ms")
    print()
    print(f">>> Best group size: G={best_G} (H={best_H} heads/group)")
    print(f"    Latency: {best_ms:.3f}ms  ({best_ms/ms_faw:.2f}× FA2-w{W},  {best_ms/ms_faf:.2f}× FA2-full)")

    speedup_vs_ungrouped = [r for r in results if r[0] == 1][0][2] / best_ms
    if best_G != 1:
        print(f"    Speedup vs ungrouped (G=1): {speedup_vs_ungrouped:.2f}×")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
