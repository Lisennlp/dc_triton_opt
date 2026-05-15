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
### v4 在之前v4的基础上增加
- autotune 加 num_warps=2
- valid 移到 _consume_qk 内部
- Sweep 1 非缓存部分改成逐 head 累加，降低 live range

B  BM    W   G  HPG |       V0       V1       V2       V3       V4       4k     sATN     FA2w |  V3/fw  V4/fw  V4/V3  4k/fw  sA/fw
------------------------------------------------------------------------------------------------------------------------------------
 16  16  112   1   32 |    7438u    7322u    7336u    6462u    6453u   11535u    2592u    2400u |   2.69x   2.69x   1.00x   4.81x   1.08x
 16  16  112   2   16 |    7601u    7260u    7242u    6450u    6427u   16222u    2631u    2400u |   2.69x   2.68x   1.00x   6.76x   1.10x
 16  16  112   4    8 |    7309u    6825u    6830u    6352u    6099u   16175u    2764u    2400u |   2.65x   2.54x   0.96x   6.74x   1.15x
 16  16  112   8    4 |    6774u    6214u    6252u    5977u    5346u   16510u    2999u    2400u |   2.49x   2.23x   0.89x   6.88x   1.25x
 16  16  112  16    2 |    6335u    5222u    5359u    5070u    5069u   17458u    3151u    2400u |   2.11x   2.11x   1.00x   7.27x   1.31x
 16  16  112  32    1 |     FAIL     FAIL     FAIL     FAIL     FAIL   19322u    3333u    2400u |    N/A    N/A    N/A   8.05x   1.39x
 16  32   96   1   32 |    6458u    5979u    5982u    5591u    5626u   10698u    1957u    2511u |   2.23x   2.24x   1.01x   4.26x   0.78x
 16  32   96   2   16 |    6357u    5858u    6033u    5683u    5688u   15472u    1927u    2511u |   2.26x   2.27x   1.00x   6.16x   0.77x
 16  32   96   4    8 |    6412u    5940u    6098u    5772u    5503u   15583u    1957u    2511u |   2.30x   2.19x   0.95x   6.21x   0.78x
 16  32   96   8    4 |    6526u    5694u    5795u    5331u    4776u   15789u    2052u    2511u |   2.12x   1.90x   0.90x   6.29x   0.82x
 16  32   96  16    2 |    4948u    4307u    4290u    4315u    4308u   16530u    2218u    2511u |   1.72x   1.72x   1.00x   6.58x   0.88x
 16  32   96  32    1 |     FAIL     FAIL     FAIL     FAIL     FAIL   17979u    2758u    2511u |    N/A    N/A    N/A   7.16x   1.10x
 16  16  240   1   32 |   22651u   20281u  424572u   13351u   13333u   20143u    6277u    3421u |   3.90x   3.90x   1.00x   5.89x   1.83x
 16  16  240   2   16 |   22594u   18671u  410726u   13056u   13022u   25283u    6286u    3421u |   3.82x   3.81x   1.00x   7.39x   1.84x
 16  16  240   4    8 |   23086u   19061u  390703u   13162u   13234u   25039u    6432u    3421u |   3.85x   3.87x   1.01x   7.32x   1.88x
 16  16  240   8    4 |   24327u   21968u  363145u   12778u   12778u   25809u    6840u    3421u |   3.73x   3.73x   1.00x   7.54x   2.00x
 16  16  240  16    2 |   24597u   12957u  291479u   11159u   11186u   27564u    7846u    3421u |   3.26x   3.27x   1.00x   8.06x   2.29x
 16  16  240  32    1 |     FAIL     FAIL     FAIL     FAIL     FAIL   31685u    6737u    3421u |    N/A    N/A    N/A   9.26x   1.97x
 32  16  112   1   32 |   15165u   14802u   14835u   12903u   13017u   22945u    5173u    5034u |   2.56x   2.59x   1.01x   4.56x   1.03x
 32  16  112   2   16 |   15408u   14556u   14648u   13083u   13118u   32855u    5394u    5034u |   2.60x   2.61x   1.00x   6.53x   1.07x
 32  16  112   4    8 |   14989u   13936u   13983u   12903u   12261u   32457u    5581u    5034u |   2.56x   2.44x   0.95x   6.45x   1.11x
 32  16  112   8    4 |   13790u   12622u   12727u   12098u   10751u   33080u    5992u    5034u |   2.40x   2.14x   0.89x   6.57x   1.19x
 32  16  112  16    2 |   12826u   10567u   10586u   10242u   10379u   34715u    6347u    5034u |   2.03x   2.06x   1.01x   6.90x   1.26x
 32  16  112  32    1 |     FAIL     FAIL     FAIL     FAIL     FAIL   38543u    6756u    5034u |    N/A    N/A    N/A   7.66x   1.34x
 32  32   96   1   32 |   12867u   12070u   11945u   11217u   11274u   21304u    3944u    5054u |   2.22x   2.23x   1.01x   4.21x   0.78x
 32  32   96   2   16 |   12820u   11886u   11980u   11427u   11467u   31094u    3968u    5054u |   2.26x   2.27x   1.00x   6.15x   0.79x


### 测试
 B  BM    W   G  HPG |       V0       V1       V2       V3       V4       4k     sATN     FA2w |  V3/fw  V4/fw  V4/V3  4k/fw  sA/fw
------------------------------------------------------------------------------------------------------------------------------------
  8  16  112   1   32 |    3781u    3710u       0u    3297u    3315u    5698u    1307u    1121u |   2.94x   2.96x   1.01x   5.08x   1.17x
  8  16  112   2   16 |    3799u    3622u       0u    3258u    3267u    8041u    1315u    1121u |   2.91x   2.91x   1.00x   7.17x   1.17x
  8  16  112   4    8 |    3665u    3408u       0u    3172u    3000u    8030u    1374u    1121u |   2.83x   2.68x   0.95x   7.16x   1.23x
  8  16  112   8    4 |    3393u    3085u       0u    2987u    2655u    8196u    1479u    1121u |   2.66x   2.37x   0.89x   7.31x   1.32x
  8  16  240   1   32 |   11312u    9977u       0u    6605u    6617u   10150u    3182u    1615u |   4.09x   4.10x   1.00x   6.29x   1.97x
  8  16  240   2   16 |   11275u    9407u       0u    6481u    6477u   12672u    3158u    1615u |   4.01x   4.01x   1.00x   7.85x   1.96x
  8  16  240   4    8 |   11586u    9633u       0u    6556u    6554u   12338u    3210u    1615u |   4.06x   4.06x   1.00x   7.64x   1.99x
  8  16  240   8    4 |   12110u   10790u       0u    6360u    6361u   12745u    3399u    1615u |   3.94x   3.94x   1.00x   7.89x   2.11x
 16  16  112   1   32 |    7401u    7258u       0u    6402u    6380u   11590u    2567u    2440u |   2.62x   2.61x   1.00x   4.75x   1.05x
 16  16  112   2   16 |    7537u    7199u       0u    6480u    6458u   16331u    2659u    2440u |   2.66x   2.65x   1.00x   6.69x   1.09x
 16  16  112   4    8 |    7323u    6861u       0u    6350u    6067u   16209u    2769u    2440u |   2.60x   2.49x   0.96x   6.64x   1.13x
 16  16  112   8    4 |    6772u    6242u       0u    6002u    5308u   16604u    2990u    2440u |   2.46x   2.18x   0.88x   6.81x   1.23x
 16  16  240   1   32 |   22685u   20408u       0u   13342u   13311u   20290u    6241u    3361u |   3.97x   3.96x   1.00x   6.04x   1.86x
 16  16  240   2   16 |   22456u   18496u       0u   12984u   13013u   25128u    6196u    3361u |   3.86x   3.87x   1.00x   7.48x   1.84x
 16  16  240   4    8 |   22959u   18866u       0u   13110u   13080u   24983u    6360u    3361u |   3.90x   3.89x   1.00x   7.43x   1.89x
 16  16  240   8    4 |   24259u   21771u       0u   12714u   12747u   25775u    6760u    3361u |   3.78x   3.79x   1.00x   7.67x   2.01x
 32  16  112   1   32 |   15089u   14730u       0u   12883u   12979u   22813u    5154u    4951u |   2.60x   2.62x   1.01x   4.61x   1.04x
 32  16  112   2   16 |   15314u   14463u       0u   13023u   13186u   32730u    5417u    4951u |   2.63x   2.66x   1.01x   6.61x   1.09x
 32  16  112   4    8 |   14854u   13887u       0u   12802u   12272u   32583u    5597u    4951u |   2.59x   2.48x   0.96x   6.58x   1.13x
 32  16  112   8    4 |   13818u   12663u       0u   12105u   10765u   33309u    6017u    4951u |   2.44x   2.17x   0.89x   6.73x   1.22x
 32  16  240   1   32 |   46090u   41445u       0u   26836u   26825u   40187u   12493u    6790u |   3.95x   3.95x   1.00x   5.92x   1.84x
 32  16  240   2   16 |   45066u   37372u       0u   26201u   26287u   50636u   12584u    6790u |   3.86x   3.87x   1.00x   7.46x   1.85x
 32  16  240   4    8 |   46107u   37939u       0u   26701u   26574u   50248u   12948u    6790u |   3.93x   3.91x   1.00x   7.40x   1.91x
 32  16  240   8    4 |   48749u   43854u       0u   25718u   25737u   51761u   13744u    6790u |   3.79x   3.79x   1.00x   7.62x   2.02x
 64  16  112   1   32 |   30476u   29839u       0u   25976u   25953u   45740u   10381u   10187u |   2.55x   2.55x   1.00x   4.49x   1.02x
 64  16  112   2   16 |   30623u   29223u       0u   26258u   26101u   65443u   10797u   10187u |   2.58x   2.56x   0.99x   6.42x   1.06x
 64  16  112   4    8 |   29814u   27875u       0u   25739u   24529u   64770u   11163u   10187u |   2.53x   2.41x   0.95x   6.36x   1.10x
 64  16  112   8    4 |   27542u   25318u       0u   24236u   21373u   66109u   12004u   10187u |   2.38x   2.10x   0.88x   6.49x   1.18x
 64  16  240   1   32 |   91595u   82564u       0u   52929u   52976u   81139u   24760u   13541u |   3.91x   3.91x   1.00x   5.99x   1.83x
 64  16  240   2   16 |   89426u   74008u       0u   51730u   51817u  101303u   24684u   13541u |   3.82x   3.83x   1.00x   7.48x   1.82x
 64  16  240   4    8 |   91390u   75581u       0u   52430u   52419u   99370u   25492u   13541u |   3.87x   3.87x   1.00x   7.34x   1.88x
 64  16  240   8    4 |   96384u   87428u       0u   50998u   50984u  102466u   27083u   13541u |   3.77x   3.77x   1.00x   7.57x   2.00x

## V5: fp16 cached QK + HPG=8 cache8 + hoisted valid mask

文件: `triton_dc_onekernel_v5.py`

V5 在 V4 cache-four-QK specialization 上继续做实验优化。整体仍然是一套 forward/一份 kernel 文件，`HPG` 是 `tl.constexpr`，因此 Triton 会在编译期选择分支:
- `HPG=4/6`: 继续使用 V4 的 cache4 结构
- `HPG=8`: 使用 cache8 分支，把 8 个 head 的 QK 都以 fp16 缓存下来，Sweep 2 不再重算 QK

适用范围保持与 V4 fast path 一致:
- `G <= 8`
- `4 <= HPG <= 8` 且 `HPG` 为偶数
- `KL = next_power_of_2(chunk_size + W - 1) <= 128`
- 其他情况继续 fallback 到 V3，避免影响 `HPG=2` 和 `KL=256` 路径

### 优化点

1. cached QK 从 fp32 live matrix 改成 fp16 live matrix，并对 `HPG=8` 尝试 cache8。

V4 会缓存最后 4 个 head 的 QK，用于 Sweep 2 免重算。原实现中这 4 个 QK 都是 `tl.dot` 输出的 fp32 tensor，并且一直 live 到 softmax/PV 阶段。V5 在每个 cached QK 完成 `s_acc += pre_w1 * qk` 之后立即转成 fp16:

```python
qk_c0 = qk_c0.to(tl.float16)
qk_c1 = qk_c1.to(tl.float16)
qk_l0 = qk_l0.to(tl.float16)
qk_l1 = qk_l1.to(tl.float16)
```

`s_acc` 仍然保持 fp32，因此 pre-aggregation 不降精度。Sweep 2 消费 QK 时再转回 fp32 参与 score:

```python
qk_f = qk.to(tl.float32)
score = (pdd + 1.0)[:, None] * qk_f + pw2[:, None] * s_acc
```

动机是降低 register pressure。V4 cache4 的主要压力来自同时持有:
- `s_acc [BM, KL] fp32`
- `a_acc [BM, KL] fp32`
- 4 个 cached `qk [BM, KL] fp32`

将 cache4 的 cached QK 改为 fp16 不改变理论 dot 数量，只降低 live state 和 register/local memory 压力。由于 QK 本身来自 fp16 Q/K 的 tensor-core dot，且 softmax 前仍会转 fp32 参与 score，这个改动的精度风险比把 `s_acc` 或 `a_acc` 改 fp16 小。

对 `HPG=8`，V5 进一步增加一个静态 cache8 分支:
- Sweep 1 计算 8 个 QK，并全部累加到 fp32 `s_acc8`
- 每个 QK 累加后立即转 fp16 缓存
- Sweep 2 直接消费 8 个 cached QK，不再重算前 4 个 head 的 QK

理论 dot 数量上，`HPG=8` 的 cache4 路径需要 `8 QK + 4 recomputed QK + 8 PV + 8 AV = 28` 次 `[BM, KL] x [KL, D]` 等价 dot；cache8 路径变成 `8 QK + 8 PV + 8 AV = 24` 次，理论计算项从 `1.75 * KL/W` 降到 `1.50 * KL/W`。但 cache8 同时让 8 个 fp16 QK matrix live 到 Sweep 2，寄存器/本地内存压力明显更高，实际是否更快必须看 profile。

2. valid mask 从 `_consume_qk` 内部挪到 kernel 主体中计算一次。

V4 每次 `_consume_qk` 都重新计算:

```python
causal = k_offs[None, :] <= q_offs[:, None]
win = (q_offs[:, None] - k_offs[None, :]) < W
valid = causal & win & q_mask[:, None] & k_mask[None, :]
```

这些值只依赖 query/key offsets 和 masks，与 head 无关。V5 在 CTA 开始处计算一次 `valid`，然后传给所有 cached/recomputed head 的 `_consume_qk`。这可以减少重复 integer/vector op，也略微缩短 `_consume_qk` 内部 live range。

3. `a_acc` 继续保持 fp32。

实验版 `triton_dc_onekernel_v5_hpg4.py` 曾测试 `a_acc_half=True`，但在 HPG=4 测试中没有带来额外速度收益，且相对 V4 的差异略增。因此正式 V5 保持 fp32 `a_acc`，只在 cached QK 精度、valid-hoist 和 `HPG=8` cache8 分支上做实验。

### benchmark 脚本更新

`bench_onekernel.py` 已更新:
- 移除 V2 import、计时代码和表格列
- 新增 `triton_dc_onekernel_v5.TritonDCOneKernel`
- 输出新增 `V5`、`V5/fw`、`V5/V4`

运行方式:

```bash
CUDA_VISIBLE_DEVICES=2 /home/lishengping/miniconda3/bin/python bench_onekernel.py
```

### cache8 短测结果

单点短测配置:

```text
B=16 T=4096 BM=16 W=112 G=4 HPG=8
CUDA_VISIBLE_DEVICES=2 /home/lishengping/miniconda3/bin/python
```

结果:

```text
FA2w 2183 us
V4   6106 us  2.80x FA2w
V5   7786 us  3.57x FA2w  1.28x V4
diff max_vs_v4=3.1250e-02 mean=3.3650e-04
```

结论: cache8 的理论 dot 数更低，但这个点上实际慢于 V4，说明 8 个 cached QK 的 live state 很可能造成了额外 register spill / occupancy 损失。当前 V5 保留该实现作为实验分支，后续完整 bench 需要重点观察 `HPG=8, W=112, KL=128` 的表现；如果整体仍慢，应该考虑把 `HPG=8` 回退到 fp16-cache4，或者只在特定 BM/W/autotune 配置下启用 cache8。

### 待补 benchmark 结果

把完整 bench 输出粘贴在这里:

```text
TODO: paste bench_onekernel.py V5 results here.
```

## Post-only one-kernel: Post0 / Post1

文件:
- `triton_dc_onekernel_Postv0.py`
- `triton_dc_onekernel_Postv1.py`

Post-only 版本去掉完整 pre path:
- 不使用 `pre_w1 / pre_w2 / pre_dd`
- score 退化为 plain scaled QK
- 每个 head 只计算一次 QK
- 保留 post path: `post_dd * probs`、`post_w1` 聚合到 `a_acc`、最终 `post_w2 * (a_acc @ V)`

理论 dot 数:

```text
sATN:      H QK + H PV        = 2H
Post-only: H QK + H PV + H AV = 3H
```

所以 post-only 相比 head-serial attention 的理论下限约为 `1.5x`。如果实际 `P0/sA` 或 `P1/sA` 已接近 `1.5x`，继续去掉 pre 不会带来数量级收益，主要瓶颈已经是 post 的 final AV pass。

### Post1 优化点

`Post1` 在 `Post0` 上做两个 AV-pass 相关实验:

1. 延后 `a_acc += post_w1 * probs`。

`Post0` 在 softmax 后先更新 `a_acc`，再做 direct PV。`Post1` 改成先用 `probs` 做 direct PV/store，再加载 `post_w1` 更新 `a_acc`。数学完全等价，目标是减少 direct PV 阶段与大矩阵 `a_acc` 更新交织造成的 live range / scheduling 压力。

2. `HPG=8/16, D=128` 使用 wide2 final AV。

final AV 中所有 head 共享同一个 `a_acc`，区别只在 `V_h`。`Post1` 对 `HPG=8/16` 尝试一次 `tl.dot(a_acc, V_pair)` 产出两个 head 的 AV，块形状从 `[BM, KL] x [KL, 128]` 变成 `[BM, KL] x [KL, 256]`，希望提高对同一个 `a_acc` 的复用。

短测中 `HPG=32` 的 wide2 会变慢，`HPG=4` 基本没有收益且在 `W=240` 略慢，原因大概率是 `[BM, 256]` output、`pw2` 和 `o_prev` 同时 live 带来更高寄存器压力。因此当前只在 `HPG=8/16` 启用 wide2，`HPG=32/4` 走普通 final AV。

### benchmark 脚本更新

`bench_onekernel.py` 已新增:
- `Post0`
- `Post1`
- `P1/fw`
- `P1/P0`
- `P1/sA`

## Post2K: two-kernel post-only attempt

文件: `triton_dc_onekernel_Post2K.py`

目标是验证: 是否可以把 PostV1 中串行的 final AV pass 拆出去，用额外 A_BUF 换 head 并行度。

最终保留的实现:
- K1: grid = `(num_chunks, B * G)`，仍然 group-serial loop `HPG`，计算 QK/softmax/direct PV，同时累加 `a_acc`
- K1 结束时把 `a_acc` 以 fp16 写入 `A_BUF[B, G, C, BM, KL]`
- K2: grid = `(num_chunks, B * G, HPG/2)`，使用 wide2 final AV，一次读取同一个 `A_BUF` 并计算两个 head 的 `A_BUF @ V_h`
- K2 把 `post_w2 * AV` 加回 K1 已写好的 direct output

尝试过但不保留的方案:
- K1 head-parallel + `tl.atomic_add(A_BUF, post_w1 * probs)`。这个版本要清零 A_BUF，而且 atomic/global 写开销很高，短测比 PostV1 慢约 `1.36x-1.81x`。
- group-serial K1 + fp32 A_BUF。比 fp16 A_BUF 更慢，原因是 A_BUF 写读带宽翻倍。

短测配置:

```text
B=16 T=4096 BM=16 N=32 D=128
CUDA_VISIBLE_DEVICES=2 /home/lishengping/miniconda3/bin/python
```

fp16 A_BUF + wide2 K2 结果:

```text
W=112 FA2w=2170 us
G=1 HPG=32 Post1=4337 us Post2K=5644 us P2K/P1=1.30 P2K/fw=2.60
G=2 HPG=16 Post1=4118 us Post2K=5754 us P2K/P1=1.40 P2K/fw=2.65
G=4 HPG=8  Post1=4077 us Post2K=5745 us P2K/P1=1.41 P2K/fw=2.65
G=8 HPG=4  Post1=4374 us Post2K=5843 us P2K/P1=1.34 P2K/fw=2.69

W=240 FA2w=3253 us
G=1 HPG=32 Post1=10101 us Post2K=13390 us P2K/P1=1.33 P2K/fw=4.12
G=2 HPG=16 Post1=9914  us Post2K=14071 us P2K/P1=1.42 P2K/fw=4.33
G=4 HPG=8  Post1=10100 us Post2K=13972 us P2K/P1=1.38 P2K/fw=4.29
G=8 HPG=4  Post1=10552 us Post2K=14530 us P2K/P1=1.38 P2K/fw=4.47
```

结论: 对当前 `BM=16, D=128, W=112/240`，Post2K 不值得。PostV1 把 `a_acc` 留在寄存器里复用，虽然 final AV 是 group 内串行，但避免了 `B*G*T*KL` 级别的 A_BUF 全局写读。Post2K 的 K2 head 并行收益抵不过 A_BUF 的全局内存流量和第二个 kernel launch。

## Pre-only one-kernel: Pre0

文件: `triton_dc_onekernel_Prev0.py`

语义:
- 保留 pre DC: `pre_w1 / pre_w2 / pre_dd`
- 去掉 post DC: 不使用 `post_w1 / post_w2 / post_dd`
- 输出为 pre-mixed logits softmax 后的 `PV`

实现:
- `HPG=4/8, KL<=128`: 缓存所有 QK，QK 累加到 fp32 `s_acc` 后转 fp16 保存，Sweep 2 直接消费 cached QK 做 pre-mixed softmax + PV
- 其他配置: 通用 two-pass。第一遍计算所有 QK 累加 `s_acc`，第二遍重算 QK 做 pre-mixed softmax + PV
- 接口兼容完整 6-tuple DC weights，也支持只传 `(pre_w1, pre_w2, pre_dd)`

理论 dot 数:

```text
Pre-only generic: H QK + H QK(recompute) + H PV = 3H
Pre-only cached HPG=4/8: H QK + H PV = 2H
Full V4 HPG=8 cache4: 8 QK + 4 QK(recompute) + 8 PV + 8 AV = 28
Pre0 HPG=8 cache8: 8 QK + 8 PV = 16
```

短测配置:

```text
B=16 T=4096 BM=16 N=32 D=128
CUDA_VISIBLE_DEVICES=2 /home/lishengping/miniconda3/bin/python
```

结果:

```text
W=112 FA2w=2218 us
G=1 HPG=32 V4=6311 us Pre0=4310 us Pre/fw=1.94 Pre/V4=0.68
G=2 HPG=16 V4=6473 us Pre0=4267 us Pre/fw=1.92 Pre/V4=0.66
G=4 HPG=8  V4=6022 us Pre0=3642 us Pre/fw=1.64 Pre/V4=0.60
G=8 HPG=4  V4=5300 us Pre0=3483 us Pre/fw=1.57 Pre/V4=0.66

W=240 FA2w=3257 us
G=1 HPG=32 V4=13203 us Pre0=9752  us Pre/fw=2.99 Pre/V4=0.74
G=2 HPG=16 V4=12941 us Pre0=9599  us Pre/fw=2.95 Pre/V4=0.74
G=4 HPG=8  V4=13129 us Pre0=9718  us Pre/fw=2.98 Pre/V4=0.74
G=8 HPG=4  V4=12720 us Pre0=10414 us Pre/fw=3.20 Pre/V4=0.82
```

结论: 去掉 post DC 后收益比去掉 pre DC 更稳定。`W=112, HPG=8` 的 `Pre/V4=0.60` 接近理论 dot ratio；`W=240` 仍然要走 generic two-pass 且 KL=256，收益稳定在 `0.74x` 左右。

### 结果
- 相比full dc，overhead减少15% ~ 30%
B  BM    W   G  HPG |       V0       V1       V3       V4    Post0    Post1     FA2w |  V4/fw  P0/fw  P1/fw  P1/P0  P1/V4
---------------------------------------------------------------------------------------------------------------------------
  8  16  112   1   32 |    3769u    3701u    3289u    3281u    2328u    2246u    1100u |   2.98x   2.12x   2.04x   0.96x   0.68x
  8  16  112   2   16 |    3772u    3590u    3271u    3249u    2279u    2127u    1100u |   2.95x   2.07x   1.93x   0.93x   0.65x
  8  16  112   4    8 |    3680u    3423u    3184u    3021u    2228u    2106u    1100u |   2.75x   2.03x   1.91x   0.95x   0.70x
  8  16  112   8    4 |    3409u    3087u    2988u    2655u    2227u    2208u    1100u |   2.41x   2.03x   2.01x   0.99x   0.83x
  8  16  240   1   32 |   11365u   10045u    6613u    6630u    5070u    5056u    1621u |   4.09x   3.13x   3.12x   1.00x   0.76x
  8  16  240   2   16 |   11308u    9453u    6475u    6490u    5010u    4999u    1621u |   4.00x   3.09x   3.08x   1.00x   0.77x
  8  16  240   4    8 |   11344u    9554u    6561u    6554u    5053u    5064u    1621u |   4.04x   3.12x   3.12x   1.00x   0.77x
  8  16  240   8    4 |   11952u   10875u    6359u    6360u    5268u    5271u    1621u |   3.92x   3.25x   3.25x   1.00x   0.83x
 16  16  112   1   32 |    7388u    7286u    6456u    6435u    4597u    4494u    2422u |   2.66x   1.90x   1.86x   0.98x   0.70x
 16  16  112   2   16 |    7529u    7315u    6520u    6491u    4619u    4281u    2422u |   2.68x   1.91x   1.77x   0.93x   0.66x
 16  16  112   4    8 |    7340u    6847u    6365u    6095u    4504u    4232u    2422u |   2.52x   1.86x   1.75x   0.94x   0.69x
 16  16  112   8    4 |    6781u    6205u    6016u    5382u    4511u    4496u    2422u |   2.22x   1.86x   1.86x   1.00x   0.84x
 16  16  240   1   32 |   22982u   20891u   13316u   13314u   10073u   10048u    3359u |   3.96x   3.00x   2.99x   1.00x   0.75x
 16  16  240   2   16 |   22676u   18746u   13022u   13047u    9968u    9858u    3359u |   3.88x   2.97x   2.93x   0.99x   0.76x
 16  16  240   4    8 |   23021u   18835u   13124u   13062u   10049u   10051u    3359u |   3.89x   2.99x   2.99x   1.00x   0.77x
 16  16  240   8    4 |   24208u   21855u   12676u   12710u   10502u   10508u    3359u |   3.78x   3.13x   3.13x   1.00x   0.83x
 32  16  112   1   32 |   14933u   14699u   12844u   12917u    9243u    8988u    4958u |   2.61x   1.86x   1.81x   0.97x   0.70x
 32  16  112   2   16 |   15339u   14473u   13005u   13126u    9230u    8560u    4958u |   2.65x   1.86x   1.73x   0.93x   0.65x
 32  16  112   4    8 |   14851u   13786u   12686u   12242u    9031u    8418u    4958u |   2.47x   1.82x   1.70x   0.93x   0.69x
 32  16  112   8    4 |   13804u   12566u   12159u   10741u    9010u    8998u    4958u |   2.17x   1.82x   1.81x   1.00x   0.84x
 32  16  240   1   32 |   46432u   42139u   26601u   26529u   20223u   20134u    6689u |   3.97x   3.02x   3.01x   1.00x   0.76x
 32  16  240   2   16 |   45236u   37572u   26121u   26031u   19852u   19567u    6689u |   3.89x   2.97x   2.93x   0.99x   0.75x
 32  16  240   4    8 |   45923u   37810u   26324u   26315u   20055u   19989u    6689u |   3.93x   3.00x   2.99x   1.00x   0.76x
 32  16  240   8    4 |   48392u   43526u   25505u   25494u   20983u   21010u    6689u |   3.81x   3.14x   3.14x   1.00x   0.82x
 64  16  112   1   32 |   30165u   29586u   25911u   25823u   18813u   18332u    9957u |   2.59x   1.89x   1.84x   0.97x   0.71x
 64  16  112   2   16 |   30595u   29220u   26319u   26201u   18897u   17312u    9957u |   2.63x   1.90x   1.74x   0.92x   0.66x
 64  16  112   4    8 |   29817u   27946u   25821u   24795u   18540u   17121u    9957u |   2.49x   1.86x   1.72x   0.92x   0.69x
 64  16  112   8    4 |   27667u   25573u   24483u   21779u   18368u   18314u    9957u |   2.19x   1.84x   1.84x   1.00x   0.84x
 64  16  240   1   32 |   93731u   85154u   53529u   53516u   40520u   40583u   13480u |   3.97x   3.01x   3.01x   1.00x   0.76x
 64  16  240   2   16 |   91071u   75619u   52488u   52491u   40056u   39522u   13480u |   3.89x   2.97x   2.93x   0.99x   0.75x
 64  16  240   4    8 |   92541u   76115u   53221u   53220u   40281u   40165u   13480u |   3.95x   2.99x   2.98x   1.00x   0.75x
 64  16  240   8    4 |   97572u   88698u   51697u   51696u   42044u   42064u   13480u |   3.83x   3.12x   3.12x   1.00x   0.81x
