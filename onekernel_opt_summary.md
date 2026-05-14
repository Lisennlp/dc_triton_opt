# Single-Kernel DC Attention 优化总结

## 背景

DC (Decomposed Cross-head) Attention 在标准 MHA 基础上增加了跨 head 的 mixing：

- **Pre-mix**: `score[n] = (1+pre_dd[n]) * QK[n] + pre_w2[n] * Σ_n'(pre_w1[n'] * QK[n'])`
- **Post-mix**: `final_probs[n] = (1+post_dd[n]) * probs[n] + post_w2[n] * Σ_n'(post_w1[n'] * probs[n'])`

核心挑战：两个 cross-head reduction（pre-agg 的 s_buf 和 post-agg 的 a_buf）。

### 之前的最优方案（TritonDCResidual，4-kernel）


| Kernel | Grid     | 功能                                  | 耗时占比 |
| ------ | -------- | ----------------------------------- | ---- |
| K0     | (C, B)   | pre-agg: 循环 N heads → s_buf         | 18%  |
| K1     | (C, B×N) | softmax stats (重算 QK)               | 22%  |
| K2     | (C, B×N) | probs + PV + post_w1 atomic (重算 QK) | 45%  |
| K3     | (C, B×N) | a_buf @ V + combine                 | 15%  |


B=16, T=4096, W=256: **21.6ms = 6.6× FA2-w**

---

## 核心思路

用户提出的关键想法：

1. **单 kernel**：grid = (q_chunks, B×G)，G 为 head 分组数，组内 mix
2. **Key 不循环**：W 足够小时，key 维度一次 `tl.dot` 算完，不需要 online softmax
3. **BM + W = 2^k**：避免 `tl.arange` 的 power-of-2 限制导致的 padding 浪费

---

## V0：三 sweep 基础版

**文件**: `triton_dc_onekernel_v0.py`

**结构**：

```
Sweep 1: loop HPG heads → QK → s_acc (寄存器)    → 写 s_buf (HBM)
Sweep 2: loop HPG heads → 重读 Q/K → QK → score → softmax → PV → a_buf
Sweep 3: loop HPG heads → 读 a_buf → a_buf@V → combine → OUT
```

**关键改进**（vs 4-kernel）：

- 单 kernel launch，零多 kernel overhead
- s_buf/a_buf 是 per-group 的，无 atomic 竞争
- key 不循环（BM+W=128 时 KL=128，一次 `tl.dot`），直接 softmax

**问题**：

- Q/K 读了 2 次（sweep 1 + sweep 2）
- V 读了 2 次（sweep 2 + sweep 3）
- s_buf/a_buf 经过 HBM（虽然小，但增加延迟）

**结果** (B=16, T=4096, BM=16, W=112, G=16 HPG=2):

- **6.4ms = 2.67× FA2-w**（vs 4-kernel Opt 的 11.2ms = 4.70×）

---

## V1：融合 Sweep 1+2，消除 Q/K 第二次读取

**文件**: `triton_dc_onekernel_v1.py`

**核心优化**：

- s_acc 和 a_acc **全部留在寄存器**，不写 HBM（零中间 buffer）
- Sweep 1 循环 heads 时，**缓存最后 2 个 head 的 QK 在寄存器里**
- Sweep 2 先处理缓存的最后 2 个 head（零重读），再重读其余 head


**结构**：

```
Sweep 1: loop HPG heads in pairs
  - 非末 pair: Q/K → QK → s_acc += pw1*QK → 丢弃 QK
  - 末 pair:   Q/K → QK → s_acc += pw1*QK → 保留 QK
  → s_acc 完成 (寄存器)

Sweep 2:
  - 末 pair (QK 在寄存器): score → softmax → PV → a_acc (零重读!)
  - 非末 pairs: 重读 Q/K → QK → score → softmax → PV → a_acc

Sweep 3: loop HPG heads → V → a_acc@V → combine → OUT
```

**HBM 流量**：

- Q 读: 1× (sweep 1) + (HPG-2)/HPG × (sweep 2 重读)
- K 读: 同上
- V 读: 2× (sweep 2 + sweep 3)
- OUT 写: 2× (sweep 2 direct + sweep 3 combine)
- **零中间 buffer**

对于 HPG=2: Q/K 完全不重读！总 HBM = Q 1× + K 1× + V 2× + OUT 2×

**结果**:


| BM  | W   | G   | HPG | V0    | V1         | FA2-w | V1/FA2-w  |
| --- | --- | --- | --- | ----- | ---------- | ----- | --------- |
| 16  | 112 | 16  | 2   | 6.4ms | **5.3ms**  | 2.4ms | **2.22×** |
| 32  | 96  | 16  | 2   | 4.9ms | **4.3ms**  | 2.5ms | **1.72×** |
| 16  | 112 | 8   | 4   | 6.8ms | **6.3ms**  | 2.4ms | **2.63×** |
| 16  | 240 | 16  | 2   | 24ms  | **12.8ms** | 3.4ms | **3.76×** |


**最佳**: BM=32, W=96, G=16, HPG=2: **1.72× FA2-w** (vs 目标 <3×)

---

## V2：大 W 优化尝试（num_warps=2）

**文件**: `triton_dc_onekernel_v2.py`

**动机**：V1 在 W=240 (KL=256) 时 register spill 严重（s_acc/a_acc `[16,256]` 太大）。

**尝试**：用 `num_warps=2`（64 threads）增加每线程可用寄存器数。

**结果**：W=240 时 413ms，完全不可用。原因：

- `num_warps=2` 的 SM 占用率太低
- 寄存器仍然不够（循环体内 qk/score/probs `[16,256]` 临时变量 + s_acc + a_acc 远超 255 regs）
- Triton 编译器在 `num_warps=2` 下的调度效率差

**结论**：对于 KL=256，`num_warps` 调整不能解决 register spill。

---

## 关键发现总结

### 1. BM + W = 2^k 是必须的

`tl.arange` 要求 power-of-2。选 BM 和 W 使其和恰好是 2 的幂，零 padding 浪费。

### 2. 寄存器是硬约束

KL=128 时 s_acc+a_acc = 32 regs/thread → fit。KL=256 时 = 64 regs/thread → spill → 性能崩掉 3-5×。

### 3. 单 kernel + 零 HBM buffer 是关键突破

V0 → V1 的提升（40%）主要来自消除 s_buf/a_buf 的 HBM 读写。

### 4. 融合 sweep 1+2 省 Q/K 重读

对 HPG=2，完全消除了 Q/K 的第二次读取。对 HPG=4+，省了最后 2 个 head 的重读。

### 5. G 越大（HPG 越小）越快

HPG=2 (G=16) 最佳：每 block 只处理 2 heads，并行度高，Q/K 无重读。

---

## 最终推荐配置


| 场景      | BM  | W   | KL  | G   | HPG | 版本  | vs FA2-w  |
| ------- | --- | --- | --- | --- | --- | --- | --------- |
| **小窗口** | 32  | 96  | 128 | 16  | 2   | V1  | **1.72×** |
| **中窗口** | 16  | 112 | 128 | 16  | 2   | V1  | **2.22×** |
| **中窗口** | 16  | 112 | 128 | 8   | 4   | V1  | **2.63×** |
| **大窗口** | 16  | 240 | 256 | 16  | 2   | V1  | **3.76×** |


大窗口 (W=240) 因 register spill 性能下降，暂无有效 Triton 解决方案。

---

## V3：exp2 softmax + autotune

**文件**: `triton_dc_onekernel_v3.py`

### 自己理解
- exp 换成 exp2，softmax 更快。
- 加了 @triton.autotune，让 num_warps/num_stages 自动选。
- 按 pair 处理 head，增加一点指令级并行。
- 最后一对 head 的 QK 缓存在寄存器里，省掉这两个 head 在 Sweep 2 的 Q/K 读和 QK 计算。

**基于 V1 的两项优化**：

### 1. exp2 替代 exp

GPU 原生支持 `exp2` 指令（1 cycle），而 `exp` 需要先转换为 `exp2(x * log2e)`（编译器隐式做，但可能不如手动高效）。

```python
# 之前 (V1)
p = tl.exp(score - rm[:, None])

# V3: 手动用 exp2
LOG2E = tl.constexpr(1.4426950408889634)
p = tl.exp2((score - rm[:, None]) * LOG2E)
```

### 2. autotune

对 `num_warps` 和 `num_stages` 做自动搜索：

```python
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['B', 'T', 'N', 'D', 'W', 'BM', 'KL', 'G', 'HPG'],
)
```

注意：`num_warps=2` 会导致大 W (KL=256) 崩溃，不包含在搜索空间中。

### V3 结果

| B | BM | W | G | HPG | V1 | V3 | FA2-w | V3/FA2-w | V3 vs V1 |
|---|-----|-----|-----|------|------|------|-------|---------|---------|
| 16 | 16 | 112 | 16 | 2 | 5232us | **4990us** | 2400us | **2.08×** | 5% faster |
| 16 | 32 | 96 | 16 | 2 | 4298us | **4143us** | 2426us | **1.71×** | 4% faster |
| 16 | 16 | 240 | 16 | 2 | 13108us | **11002us** | 3356us | **3.28×** | 16% faster |
| 32 | 16 | 112 | 16 | 2 | 10538us | **10119us** | 5026us | **2.01×** | 4% faster |
| 32 | 32 | 96 | 16 | 2 | 8697us | **8410us** | 5073us | **1.66×** | 3% faster |
| 32 | 16 | 240 | 16 | 2 | — | **~22000us** | 6988us | **~3.1×** | — |

**W=112 提升 3-5%，W=240 提升 16%**。autotune 在大 W 时选择了更优的 `num_warps/stages` 组合。

---

## 最终推荐配置（V3）

| 场景 | BM | W | KL | G | HPG | 版本 | vs FA2-w |
|------|-----|-----|-----|-----|------|------|---------|
| **小窗口** | 32 | 96 | 128 | 16 | 2 | V3 | **1.66×** |
| **中窗口** | 16 | 112 | 128 | 16 | 2 | V3 | **2.08×** |
| **中窗口** | 16 | 112 | 128 | 8 | 4 | V3 | **2.43×** |
| **大窗口** | 16 | 240 | 256 | 16 | 2 | V3 | **3.28×** |

---

## vs 原方案对比 (B=16, T=4096)

| 方案 | W=112 最佳 | W=240 最佳 |
|------|-----------|-----------|
| TritonDCResidual (4-kernel) | 11.3ms (4.72× FA2-w) | 20.1ms (5.99× FA2-w) |
| V1 single-kernel | 5.2ms (2.18× FA2-w) | 13.1ms (3.91× FA2-w) |
| **V3 single-kernel** | **5.0ms (2.08× FA2-w)** | **11.0ms (3.28× FA2-w)** |
| V3 vs 4-kernel 提升 | **2.3× faster** | **1.8× faster** |




## Raw Benchmark Data (latest, V0-V3)

```
  B  BM    W   G  HPG |       V0       V1       V2       V3      4k     FA2w |  V0/fw  V1/fw  V2/fw  V3/fw  4k/fw
----------------------------------------------------------------------------------------------------------------
 16  16  112   2   16 |    7518u    7232u    7170u    6493u   11321u    2400u |   3.13x   3.01x   2.99x   2.71x   4.72x
 16  16  112   4    8 |    7309u    6891u    6833u    6319u   11321u    2400u |   3.05x   2.87x   2.85x   2.63x   4.72x
 16  16  112   8    4 |    6769u    6258u    6246u    5836u   11321u    2400u |   2.82x   2.61x   2.60x   2.43x   4.72x
 16  16  112  16    2 |    6332u    5232u    5324u    4990u   11321u    2400u |   2.64x   2.18x   2.22x   2.08x   4.72x
 16  32   96   2   16 |    6481u    5846u    5822u    5337u   10470u    2426u |   2.67x   2.41x   2.40x   2.20x   4.32x
 16  32   96   4    8 |    6376u    6102u    5951u    5500u   10470u    2426u |   2.63x   2.51x   2.45x   2.27x   4.32x
 16  32   96   8    4 |    6477u    5704u    5679u    5242u   10470u    2426u |   2.67x   2.35x   2.34x   2.16x   4.32x
 16  32   96  16    2 |    4969u    4298u    4383u    4143u   10470u    2426u |   2.05x   1.77x   1.81x   1.71x   4.32x
 16  16  240   2   16 |   22676u   18675u  413413u   13122u   20101u    3356u |   6.76x   5.57x 123.2x   3.91x   5.99x
 16  16  240   4    8 |   22987u   19037u  392787u   12719u   20101u    3356u |   6.85x   5.67x 117.1x   3.79x   5.99x
 16  16  240   8    4 |   24350u   21777u  365253u   12724u   20101u    3356u |   7.26x   6.49x 108.9x   3.79x   5.99x
 16  16  240  16    2 |   24834u   13108u  295760u   11002u   20101u    3356u |   7.40x   3.91x  88.1x   3.28x   5.99x
 32  16  112   2   16 |   15221u   14587u   14528u   13091u   22808u    5026u |   3.03x   2.90x   2.89x   2.60x   4.54x
 32  16  112   4    8 |   14865u   13887u   13853u   12777u   22808u    5026u |   2.96x   2.76x   2.76x   2.54x   4.54x
 32  16  112   8    4 |   13786u   12667u   12680u   11754u   22808u    5026u |   2.74x   2.52x   2.52x   2.34x   4.54x
 32  16  112  16    2 |   13012u   10538u   10735u   10119u   22808u    5026u |   2.59x   2.10x   2.14x   2.01x   4.54x
 32  32   96   2   16 |   13009u   11980u   12062u   10792u   21213u    5073u |   2.56x   2.36x   2.38x   2.13x   4.18x
 32  32   96   4    8 |   13211u   12165u   12140u   10937u   21213u    5073u |   2.60x   2.40x   2.39x   2.16x   4.18x
 32  32   96   8    4 |   13545u   11470u   11662u   10560u   21213u    5073u |   2.67x   2.26x   2.30x   2.08x   4.18x
 32  32   96  16    2 |   10238u    8697u    8701u    8410u   21213u    5073u |   2.02x   1.71x   1.72x   1.66x   4.18x
 32  16  240   2   16 |   46357u   37873u  825646u   26521u   40644u    6988u |   6.63x   5.42x 118.2x   3.80x   5.82x
 32  16  240   4    8 |   47395u   38830u  787742u   25638u   40644u    6988u |   6.78x   5.56x 112.7x   3.67x   5.82x
 32  16  240   8    4 |   50098u   44055u  730701u   25600u   40644u    6988u |   7.17x   6.30x 104.56x   3.66x   5.82x
 32  16  240  16    2 |  113922u   26286u  589706u   21960u   40644u    6988u |  16.30x   3.76x  84.39x   3.14x   5.82x
```