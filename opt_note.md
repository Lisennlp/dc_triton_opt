这次优化过程可以总结为：

  1. 明确 full DC 语义

  我先对齐了 dc_attention_window_chunked_residual 的数学形式。相比之前的 TritonDCRank1_3K_Grouped，full DC 多了两部分：

  - pre 端：logits + pre_w2 * sum(pre_w1 * logits) + pre_dd * logits
  - post 端：probs + post_w2 * sum(post_w1 * probs) + post_dd * probs

  也就是说它不是单纯 rank-1 分解，必须保留每个 head 自己的 direct qk 和 pV 分支。

  2. 第一版 Triton full DC

  我新增了 triton_dc_residual.py，初始实现是 4 阶段：

  - K0：计算并保存 per-head qk_buf，同时生成 pre 聚合 s_buf
  - K1：基于 qk_buf + s_buf 做 softmax stats
  - K2：生成 per-head pV，并用 atomic 做 post 聚合 a_buf
  - K3：再做一次 a_buf @ V，合成 post_dd direct 分支和 post_w2 聚合分支

  这个版本 correctness 通过，但大尺寸大约是 7.2x FA2-w。

  3. 针对瓶颈做阶段计时

  在 B=16,T=4096,N=32,D=128,W=256 下分阶段看，初始版本大概是：

  K0 ~8.7ms
  K1 ~2.8ms
  K2 ~8.5ms
  K3 ~4.1ms
  full ~24.3ms

  主要瓶颈是 K0 的 per-head qk_buf 写入，以及 K2 的概率生成 + post 聚合 atomic。

  4. 试过的优化路径

  我尝试了几类改法：

  - 搜索 BLOCK_M/BLOCK_K，发现主路径较好的组合是中小 tile，final PV 更适合 BM=64/BK=32。
  - 尝试 K0 按 head 并行，用 atomic 累加 pre 聚合，结果比串行 head 更慢。
  - 去掉 qk_buf，改为只保存 s_buf，后续 K1/K2 重算 direct qk。这个减少了大规模读写，整体从约 24.3ms 降到约 21.5ms。
  - 尝试把 post 聚合 a_buf 从 fp32 改 fp16，结果 fp16 atomic 更慢，回退 fp32。
  - 尝试显式保存 [B,N,T,W] 的 post prob，再单独 reduce head，避免 atomic。结果多出的写入和 reduce 比 atomic 更慢，保留 atomic 路径。

  5. 最终结果

  最终 benchmark 使用新 TritonDCResidual，脚本在 bench_atg_vs_opt.py。大尺寸结果大致是：

  B=16,T=4096  Opt/FA2-w ~6.42x, Opt/FA2-f ~1.75x
  B=32,T=4096  Opt/FA2-w ~6.53x, Opt/FA2-f ~1.76x
  B=64,T=4096  Opt/FA2-w ~6.60x, Opt/FA2-f ~1.78x

  也就是说：相比 torch 版本已经快很多，也接近或优于 FA2-full 的 2x 内，但没有达到你设定的 <3x FA2-w 目标。

  6. 当前瓶颈判断

  现在主要瓶颈不是简单 tile 参数，而是 full DC 的结构性开销：

  - pre 阶段需要 cross-head 聚合；
  - post 阶段也需要 cross-head 聚合；
  - post 聚合当前依赖 atomic_add 到 [B,T,W]；
  - full DC 实际上至少包含多次 QK/PV 或中间 buffer 读写，很难用普通 Triton kernel 达到 FA2-w 附近。

  如果继续优化，下一步更可能需要写更深度 fused 的 CUDA kernel，重点解决 K2 的 post_w1 * probs cross-head reduce 和第二次 AV。

  7. 第二轮 Triton 优化尝试

  目标 <3× FA2-w (~10ms at B=16,T=4096)，当前 ~21.6ms (6.5× FA2-w)。

  分阶段精确计时 (B=16, T=4096, BM=16, BK=64, BM_F=64, BK_F=32):
    K0 (pre-agg, grid=C×B):        3818 us   17.7%
    K1 (stats, grid=C×B×N):        4707 us   21.8%
    K2 (probs+PV+post, grid=C×B×N): 9778 us   45.2%  ← 最大瓶颈
    K3 (final, grid=C×B×N):        3316 us   15.3%

  尝试的方案和结果：

  a) 单 kernel 方案 (grid=C×B, 循环 N heads):
     - 所有 head 串行处理，无 atomic，s_buf/a_buf 不竞争
     - 结果：1.26-1.41× 慢于 4-kernel Opt
     - 原因：grid 只有 C×B = 4096 blocks，SM 利用率低

  b) Fused K12 (合并 K1+K2，online softmax):
     - grid=C×B×N, pass1: online softmax + PV, pass2: QK recompute + post_w1 atomic
     - 结果：与 K1+K2 分开基本持平（~22.4ms），甚至略慢
     - 原因：总 QK 计算量不变 (2× QK)，只省了 m_buf/l_buf 的 HBM 读写 (< 1%)

  c) K0 per-head 并行 + fp32 atomic:
     - grid=C×B×N, 每个 head 单独计算 QK, atomic_add 到 fp32 s_buf
     - 结果：比串行 K0 更慢 (25.7ms)，精度也下降
     - 原因：32 个 head 争抢同一 s_buf 位置，atomic 争用严重

  d) 不同 tile 大小搜索:
     - K0: BM=64,BK=64 最佳 (3212us, 比默认快 17%)
     - K1: BM=32,BK=32 最佳 (4017us, 比默认快 13%)
     - K2: BM=16,BK=64 仍是最佳（更大 tile 反而慢）
     - K3: BM=64,BK=32 最佳 (3294us)
     - 但 forward() API 共用 BM 参数，K0 和 K1/K2 不能独立设置

  e) Noatomic K12 (grid=C×B, 循环 N, 无 atomic):
     - online softmax + PV + a_buf load-add-store
     - 结果：26ms+，比 atomic 版本慢 ~18%
     - 原因：grid=C×B 并行度太低

  f) 预计算 q_pw1 重构 pre-agg 为 windowed matmul:
     - s_buf = (pw1⊙Q).reshape(B,T,ND) @ K.reshape(B,T,ND)^T
     - 结果：23.6ms，比 Opt 慢，精度也下降
     - 原因：ND=4096 维度的 matmul 仍需 32 个 inner loop tile，等价于 N loop

  8. 结论

  理论下限分析:
    DC per head 需要: 1× QK (pre-agg 均摊) + 1× QK (softmax) + 1× QK (post-agg) + 1× PV + 1× AV = 5 ops
    FA2-w per head:   1× QK + 1× PV = 2 ops
    理论最低: DC/FA2-w = 5/2 = 2.5×

  当前 6.5× ÷ 理论 2.5× = 2.6×，即当前实现在理论效率的 ~38%。
  要达到 <3× FA2-w = 理论效率的 83%，需要接近 FlashAttention2 级别的内核优化。

  Triton 的局限：
    - 无法做 block 间同步 (inter-block barrier)
    - 无法手动管理 shared memory 或 warp-level 通信
    - 无法做 persistent kernel (单次 launch，多阶段复用 SM)
    - register pressure 控制粗粒度

  9. CUDA 原型尝试

  实现了 dc_fused_cuda/kernel.cu，persistent kernel 设计:
  - Grid = (C, B), Block = (32, 32) = 1024 threads, 每个 warp 一个 head
  - Cross-head 聚合通过 shared memory + __syncthreads()
  - 三个 phase 在同一 kernel 内完成

  结果: correctness OK (max_abs vs torch ~0.18), 但速度极慢 (304ms vs Opt 21.7ms)
  原因: scalar code (每元素循环 + warp_reduce_sum) 无法利用 tensor core

  要达到目标 <3× FA2-w, 需要:
  - 使用 wmma/mma.sync 做 QK 和 PV 的矩阵乘
  - 向量化内存加载 (float4 / half8)
  - 类 FlashAttention 的 register tiling + shared memory 管理
  - 这是 ~1000+ 行高度优化 CUDA，工程量与 FlashAttention-2 相当

  10. 总结和下一步

  当前 Triton 实现: ~6.5× FA2-w, ~1.8× FA2-f (21.6ms at B=16, T=4096)
  理论下限: ~2.5× FA2-w
  目标: <3× FA2-w ≈ 理论效率 83%

  可行路线:
  a) 基于 FlashAttention-2 源码做 fork, 在其 kernel 中插入 DC mixing 逻辑
     - 复用 FA2 的 wmma tiling、memory 管理、softmax 实现
     - 只需添加 pre/post cross-head aggregation 的 shared memory 通信
     - 这是最现实的路线, 工作量约 2-3 天

  b) 从零写 CUDA kernel with wmma
     - 完全控制但工程量巨大
     - 需要 mma.sync、register tiling、double buffering、swizzle 等
     - 约 1-2 周

  11. FA2 Triton 改造尝试

  基于 flash_attn/flash_attn_triton.py 的 _fwd_kernel, 创建了 dc_attn_fa2_triton.py:
  - K0 + K_dc(FA2风格) + K3
  - K_dc 使用 FA2 的 online softmax + PV, 添加 DC pre-mix 和 post_w1 atomic
  - 修复了 DC 分数的 NaN 问题 (direct_scale < 0 时 -inf * negative = +inf)
  - 修复了 window-masked tiles 导致 exp(-inf - (-inf)) = NaN 的问题

  结果:
    - Correctness 完美 (max_diff vs Opt = 0.0625, 与 Triton 4-kernel 一致)
    - 最佳 tile: BM=64 BN=64 → 25.5ms (8.14× FA2-w)
    - 比 4-kernel Opt (21.7ms, 6.6× FA2-w) 更慢

  原因: FA2 框架下仍需 2 pass QK (softmax+PV 和 post_w1 atomic), 加上 BLOCK_M/N 较大导致
  window 边界处浪费更多计算. FA2 的优势 (大 tile, tensor core) 被 DC 的 2nd pass + window
  浪费抵消了.

  12. 最终结论

  DC residual attention 的根本瓶颈是:
  - 2 个 cross-head reduction (pre-agg 和 post-agg) 需要多次 QK 计算
  - post-agg 的 a_buf 依赖所有 head 的 softmax 结果, 无法在 per-head kernel 中消除
  - 总计 3× QK + PV + AV = 5 ops vs FA2 的 QK + PV = 2 ops
  - 理论下限 2.5× FA2-w, 当前 6.5× = 理论 38%

  在 Triton 框架下, 4-kernel 方案 (K0 + K1 + K2 + K3) 已接近最优:
  - 各 kernel 独立 tile 优化
  - K0 BM=64 比默认 BM=16 快 17%
  - 但受限于 K0 的低并行度 (grid=C×B) 和 K2 的 atomic 开销

  要达到 <3× FA2-w (理论 83%):
  - 需要 warp-cooperative cross-head reduce + 单次 launch persistent kernel
  - 这意味着必须用 CUDA C++ + wmma, 而非 Triton
  - 或者修改算法本身 (如近似 post-agg, 减少 cross-head 依赖)



  B     T  BM    W   KL   G  HPG |       1K      Opt     FA2w     FA2f |  1K/FA2w  Opt/FA2w  1K/FA2f  1K/Opt
------------------------------------------------------------------------------------------------------------
 16  4096  16  112  128   1   32 |   10611u   11366u    2419u   12549u |    4.39x    4.70x    0.85x    0.93x
 16  4096  16  112  128   2   16 |   10700u   11366u    2419u   12549u |    4.42x    4.70x    0.85x    0.94x
 16  4096  16  112  128   4    8 |   10872u   11366u    2419u   12549u |    4.49x    4.70x    0.87x    0.96x
 16  4096  16  112  128   8    4 |   11144u   11366u    2419u   12549u |    4.61x    4.70x    0.89x    0.98x
 16  4096  16  112  128  16    2 |   10331u   11366u    2419u   12549u |    4.27x    4.70x    0.82x    0.91x
 16  4096  16  112  128  32    1 |   11717u   11366u    2419u   12549u |    4.84x    4.70x    0.93x    1.03x
 16  4096  32   96  128   1   32 |   13367u   10499u    2449u   13080u |    5.46x    4.29x    1.02x    1.27x
 16  4096  32   96  128   2   16 |   13092u   10499u    2449u   13080u |    5.35x    4.29x    1.00x    1.25x
 16  4096  32   96  128   4    8 |   13444u   10499u    2449u   13080u |    5.49x    4.29x    1.03x    1.28x
 16  4096  32   96  128   8    4 |   14217u   10499u    2449u   13080u |    5.81x    4.29x    1.09x    1.35x
 16  4096  32   96  128  16    2 |   14300u   10499u    2449u   13080u |    5.84x    4.29x    1.09x    1.36x
 16  4096  32   96  128  32    1 |   14701u   10499u    2449u   13080u |    6.00x    4.29x    1.12x    1.40x
 16  4096  64   64  128   1   32 |   15359u    9451u    1935u   12672u |    7.94x    4.88x    1.21x    1.63x
 16  4096  64   64  128   2   16 |   15920u    9451u    1935u   12672u |    8.23x    4.88x    1.26x    1.68x
 16  4096  64   64  128   4    8 |   16181u    9451u    1935u   12672u |    8.36x    4.88x    1.28x    1.71x
 16  4096  64   64  128   8    4 |   17375u    9451u    1935u   12672u |    8.98x    4.88x    1.37x    1.84x
 16  4096  64   64  128  16    2 |   19450u    9451u    1935u   12672u |   10.05x    4.88x    1.53x    2.06x
 16  4096  64   64  128  32    1 |   18275u    9451u    1935u   12672u |    9.44x    4.88x    1.44x    1.93x
 16  4096  16  240  256   1   32 |   38875u   20254u    3397u   12558u |   11.44x    5.96x    3.10x    1.92x
 16  4096  16  240  256   2   16 |   37128u   20254u    3397u   12558u |   10.93x    5.96x    2.96x    1.83x
 16  4096  16  240  256   4    8 |   37686u   20254u    3397u   12558u |   11.09x    5.96x    3.00x    1.86x
 16  4096  16  240  256   8    4 |   40304u   20254u    3397u   12558u |   11.86x    5.96x    3.21x    1.99x
 16  4096  16  240  256  16    2 |   31211u   20254u    3397u   12558u |    9.19x    5.96x    2.49x    1.54x
 16  4096  16  240  256  32    1 |   27304u   20254u    3397u   12558u |    8.04x    5.96x    2.17x    1.35x
 16  4096  32  224  256   1   32 |   44000u   19259u    3460u   12603u |   12.72x    5.57x    3.49x    2.28x
 16  4096  32  224  256   2   16 |   43298u   19259u    3460u   12603u |   12.51x    5.57x    3.44x    2.25x
 16  4096  32  224  256   4    8 |   43674u   19259u    3460u   12603u |   12.62x    5.57x    3.47x    2.27x
 16  4096  32  224  256   8    4 |   45599u   19259u    3460u   12603u |   13.18x    5.57x    3.62x    2.37x
 16  4096  32  224  256  16    2 |   41500u   19259u    3460u   12603u |   11.99x    5.57x    3.29x    2.15x
 16  4096  32  224  256  32    1 |   43022u   19259u    3460u   12603u |   12.43x    5.57x    3.41x    2.23x
 16  4096  64  192  256   1   32 |  352845u   18277u    3113u   12730u |  113.36x    5.87x   27.72x   19.30x
 16  4096  64  192  256   2   16 |  347522u   18277u    3113u   12730u |  111.65x    5.87x   27.30x   19.01x
 16  4096  64  192  256   4    8 |  346892u   18277u    3113u   12730u |  111.45x    5.87x   27.25x   18.98x
 16  4096  64  192  256   8    4 |  351387u   18277u    3113u   12730u |  112.89x    5.87x   27.60x   19.23x