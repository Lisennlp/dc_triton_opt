# FA3 / Hopper CUDA DC Kernel Plan

This note records the CUDA direction after the V4H/V4HCM Triton experiments.

## Why the direct FA3 scoremod patch is not enough

FA3 forward is organized around one attention head per tile. In the SM90 mainloop,
`tSrS` is the QK score fragment for one head, then FA3 applies optional softcap,
masking, online softmax, and `P @ V`.

DC needs cross-head state inside each group:

```text
s_acc = sum_i pre_w1_i * qk_i
score_h = (1 + pre_dd_h) * qk_h + pre_w2_h * s_acc

a_acc = sum_i post_w1_i * probs_i
out_h = (((1 + post_dd_h) * probs_h + post_w2_h * a_acc) @ V_h)
```

For `G=8, HPG=4`, one output head cannot be computed from its own `tSrS` alone.
The kernel must see the four head-local QK/prob fragments for the same
`(batch, m_block, group, n_block)` before softmax and before final `P @ V`.

Relevant FA3 insertion points:

- `flash-attention/hopper/flash.h`
  - `Flash_fwd_params` would need DC weight pointers and strides.
- `flash-attention/hopper/flash_api.cpp`
  - add a new DC op rather than changing the public FA3 `fwd` path.
- `flash-attention/hopper/flash_fwd_launch_template.h`
  - instantiate a DC-specific mainloop for `sm90, D=128`.
- `flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`
  - current `scoremod_premask_fn(tSrS)` only has one head's score fragment.
  - DC requires a forked mainloop that owns the four head fragments.
- `flash-attention/hopper/epilogue_fwd.hpp`
  - reusable only after the DC mainloop has produced final per-head `tOrO`.

Conclusion: do not patch the generic FA3 scoremod lambda. It cannot express DC
without adding inter-head communication that the standard FA3 kernel does not
have.

## Target shapes

Keep the first CUDA path narrow and measurable:

```text
D = 128
N = 32
G = 8
HPG = 4
T = 4096
forward only
causal local window
no dropout
no varlen
no paged KV
no softcap
```

Priority configs:

```text
BM=32, W=96,  KL=128   # current best H100 small-window target
BM=32, W=224, KL=256   # wide-window target
```

`BM=16,W=112/240` should remain benchmark rows, but the H100 CUDA work should
start with `BM=32` because all Triton measurements show better tensor-core
utilization there.

## FA3 lessons that must carry over

The CUDA/Hopper path should copy the useful FA3-vs-FA2 ideas, not just rewrite
the Triton algorithm in CUDA.

FA3 changes that matter for DC:

- **WGMMA instead of warp-level mma.sync/WMMA**
  - Use Hopper warp-group matrix multiply for QK and final mixed-probability V
    dot.
  - Avoid the sm80-style “one warp owns one small tile” design used by the old
    `dc_fused_cuda` experiment.
- **TMA global-to-shared copies**
  - Q/K/V tile movement should be asynchronous and descriptor-driven.
  - K/V loads should be pipelined across stages, not scalar/vector copied by
    every worker thread.
- **Warp-specialized producer/consumer pipeline**
  - Keep a producer warp group feeding Q/K/V smem while consumer warp groups run
    WGMMA and softmax.
  - The current Triton DC path serializes too much of this work, so FA3 gets a
    much better H100 baseline.
- **Overlap QK, softmax, and PV**
  - FA3 overlaps the next QK tile load / QK compute with current softmax and
    `P @ V` work.
  - DC adds two barriers (`s_acc` before softmax, `a_acc` before final V dot),
    but the CUDA design should still pipeline K/V TMA and WGMMA around those
    barriers.
- **Use FA3-style tile scheduling**
  - Prefer FA3's `(m_block, head, batch)` scheduling and local-window
    `n_block_min/n_block_max` logic.
  - For DC, replace the head dimension in the scheduler with a group dimension
    when using one CTA per HPG group, or with a CTA cluster when using DSM.
- **Keep the epilogue simple**
  - FA3's output epilogue is valuable only after the DC mainloop has already
    produced final per-head `tOrO`.
  - Do not store intermediate direct PV output to HBM and reload it; V4HCM's
    mixed-probability identity must remain the final-output form.
- **Avoid FP8 as the first target**
  - FA3's FP8 path is important for Hopper peak throughput, but DC currently
    needs fp16/fp32 cross-head reductions and a correctness baseline first.
  - FP8 can be a later experiment after the fp16/bf16 HPG=4 path is stable.

Practical implication:

```text
Bad CUDA port:
  scalar/thread-loop QK + shared-memory softmax + scalar PV

Useful Hopper port:
  TMA K/V staging + WGMMA QK/PV + warp-specialized pipeline
  + DC cross-head reductions inserted at the two unavoidable group barriers
```

## Candidate implementations

### A. Fork FA3 into a 4-head group mainloop

One CTA computes one `(batch, m_block, group)` and owns all four heads.

Pros:
- Direct cross-head reductions stay in registers/shared memory.
- No global intermediate buffers.
- Epilogue can write the same output layout as FA3.

Cons:
- Register pressure is high: four `qk/probs` fragments plus `s_acc/a_acc`.
- `KL=256` is especially risky because live state doubles.
- Requires substantial changes to FA3's mainloop layout and TMA descriptors.

Expected first target:

```text
BM=32, KL=128, HPG=4
qk0..qk3 in fp32 accumulator fragments
s_acc/a_acc in fp32 or fp16 fragments depending pressure
cached probs in fp16
mixed probs before the single final V dot
```

### B. Hopper cluster/DSM design

Use a CTA cluster of four CTAs. Each CTA owns one head in the group and uses a
FA3-like single-head WGMMA/TMA pipeline. The four CTAs exchange compact fp16
QK/prob tiles through distributed shared memory and synchronize at the two DC
barriers.

Per local tile:

```text
CTA h computes qk_h
store fp16 qk_h to cluster-visible smem
cluster.sync
each CTA reads qk_0..qk_3, builds s_acc and score_h
softmax -> probs_h
store fp16 probs_h to cluster-visible smem
cluster.sync
each CTA reads probs_0..probs_3, builds a_acc
mixed_h = (1 + post_dd_h) * probs_h + post_w2_h * a_acc
mixed_h @ V_h -> out_h
```

Pros:
- Keeps the FA3 mental model: one CTA still owns one head.
- Avoids carrying four heads of WGMMA accumulator state in one CTA.
- Cross-head exchange uses Hopper DSM instead of HBM.

Cons:
- Requires CUDA cooperative cluster launch attributes.
- DSM reads add latency and can reduce occupancy.
- More intrusive than the Triton path and needs H100-only validation.

This is the more promising implementation path if the one-CTA/four-head fork
spills too much.

## First coding milestone

Create a separate experimental extension instead of modifying the benchmark
baseline in-place:

```text
dc_triton_opt/dc_hopper_cuda/
  setup.py
  dc_hopper_api.cpp
  dc_hopper_kernel.cu
  __init__.py
```

Expose one narrow function:

```python
dc_hopper_cuda.forward_hpg4_bm32(
    q, k, v,
    pre_w1, pre_w2, pre_dd,
    post_w1, post_w2, post_dd,
    scaling,
    window,
)
```

Shape checks should reject everything except the target configs above.
`bench_onekernel_h100.py` should only add the CUDA column after the extension
returns correct output on the debug benchmark.

## Correctness gate

Before timing on H100:

```text
B in {1, 2}
T in {128, 256, 512}
BM=32
W in {96, 224}
G=8
HPG=4
compare against dc_attention_torch or existing V4HCM/V4HCM256
max error target: same order as Triton V4HCM fp16 path
```

Only after this gate should the kernel be inserted into
`bench_onekernel_h100.py`.

## Performance gate

The first useful H100 target is:

```text
BM=32,W=96:
  <= 2.2x FA3 for B >= 16

BM=32,W=224:
  first target <= 3.0x FA3
  later target ~= A800 overhead band, around 2.2x FA2/FA3-relative
```

If the forked FA3 mainloop cannot pass the small-window target without heavy
spilling, switch to the cluster/DSM design.
