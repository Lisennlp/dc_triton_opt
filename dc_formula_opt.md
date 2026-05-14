# Attention 修正结构的矩阵融合优化分析

## 1. 原始算法

设：

- `q, k, v` 的 shape 为：

\[
(B, N, T, D)
\]

其中：

- \(B\)：batch size
- \(N\)：head 数
- \(T\)：sequence length
- \(D\)：head dim

Attention logits：

\[
logits = QK^T
\]

因此：

\[
logits \in \mathbb{R}^{B\times N\times T\times T}
\]

算法如下：

```python
logits = q @ k.transpose(-1, -2)

pre_logits = logits @ w1 @ w2
dd_logits  = logits @ w3

logits = logits + pre_logits + dd_logits

probs = softmax(logits)

post_probs = probs @ w4 @ w5
dd_probs   = probs @ w6

probs = probs + post_probs + dd_probs

out = probs @ v
```

其中：

\[
w_1,w_2,w_3,w_4,w_5,w_6
\]

均作用于 attention map 的最后一个维度。

---

# 2. logits 阶段融合

原始形式：

\[
L' = L + LW_1W_2 + LW_3
\]

利用矩阵分配律：

\[
(A+B)C = AC + BC
\]

可以提取公共项 \(L\)：

\[
L'
=
L(I + W_1W_2 + W_3)
\]

定义：

\[
W_L = I + W_1W_2 + W_3
\]

则：

\[
L' = LW_L
\]

因此：

```python
Wl = I + w1 @ w2 + w3

logits = (q @ k.transpose(-1, -2)) @ Wl
```

---

# 3. probs 阶段融合

同理：

\[
P' = P + PW_4W_5 + PW_6
\]

提取公共项：

\[
P'
=
P(I + W_4W_5 + W_6)
\]

定义：

\[
W_P = I + W_4W_5 + W_6
\]

则：

\[
P' = PW_P
\]

实现：

```python
Wp = I + w4 @ w5 + w6

probs = probs @ Wp
```

---

# 4. 吸收到 Value（最关键优化）

原始：

\[
out = (PW_P)V
\]

利用矩阵结合律：

\[
(AB)C = A(BC)
\]

可得：

\[
out = P(W_PV)
\]

定义：

\[
V' = W_PV
\]

则：

\[
out = PV'
\]

因此：

```python
Vp = Wp @ v

out = probs @ Vp
```

这样：

- 不再需要：

\[
probs @ Wp
\]

- 避免一次 attention-size matmul
- 减少显存读写
- 减少 kernel launch
- 非常适合 FlashAttention/Triton

---

# 5. 吸收到 Key（进一步优化）

当前 logits：

\[
L = (QK^T)W_L
\]

利用结合律：

\[
(QK^T)W_L
=
Q(K^TW_L)
\]

又因为：

\[
K^TW_L = (W_L^TK)^T
\]

定义：

\[
K' = W_L^TK
\]

则：

\[
L = QK'^T
\]

实现：

```python
Kp = Wl.transpose(-1, -2) @ k

logits = q @ Kp.transpose(-1, -2)
```

---

# 6. 最终优化结构

最终 attention：

```python
Wl = I + w1 @ w2 + w3
Wp = I + w4 @ w5 + w6

Kp = Wl.transpose(-1, -2) @ k
Vp = Wp @ v

logits = q @ Kp.transpose(-1, -2)

probs = softmax(logits)

out = probs @ Vp
```

---

# 7. 复杂度分析

## 原始结构

额外包含：

- 两次 logits attention-map matmul
- 两次 probs attention-map matmul

复杂度：

\[
O(T^3)
\]

并且需要：

- 大量 attention matrix 中间缓存
- 多次 kernel launch

---

## 优化后

额外计算转移为：

- Key 投影
- Value 投影

复杂度：

\[
O(T^2D)
\]

或者：

\[
O(TD^2)
\]

当：

\[
T \gg D
\]

时收益非常明显。

---

# 8. 工程收益

优化后：

- 减少 attention map 的 materialize
- 减少 HBM 读写
- 减少 kernel launch
- 更适合 Triton kernel fusion
- 更适合 FlashAttention
- 更适合 long-context inference

---

# 9. 本质分析

原结构本质上是在：

- 对 attention map 做动态右乘修正
- 对 probs 做动态后处理

最终可以等价转化为：

- Key 空间变换
- Value 空间变换

因此本质上接近：

- Talking Heads Attention
- Dynamic Projection
- HyperAttention
- Low-rank Attention Modulation
- Linearized Attention Reparameterization

---

# 10. 最终推荐实现

推荐最终实现：

```python
# fuse logits branch
Wl = I + w1 @ w2 + w3

# fuse probs branch
Wp = I + w4 @ w5 + w6

# absorb into K/V
Kp = Wl.transpose(-1, -2) @ k
Vp = Wp @ v

# attention
logits = q @ Kp.transpose(-1, -2)

probs = softmax(logits)

out = probs @ Vp
```

这是：

- 数学等价
- 计算更高效
- 显存更友好
- 更适合 fused attention kernel

的最终形式。
