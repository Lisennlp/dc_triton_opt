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
BM=32, W=224, KL=256   # primary wide-window target, best wide-window speed so far
BM=16, W=240, KL=256   # secondary target if maximizing W matters more than speed
```

`BM=32,W=96,KL=128` is no longer an active CUDA target because the window is too
small for the intended use case. It remains useful only as a diagnostic if the
wide-window path needs a smaller live-state sanity check.

`BM=16,W=240` should remain as the larger-W fallback, but the first performance
implementation should start with `BM=32,W=224` because all Triton measurements
show better tensor-core utilization there.

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
BM=32, KL=256, HPG=4
qk0..qk3 in fp32 accumulator fragments
s_acc/a_acc in fp32 or fp16 fragments depending pressure
cached probs in fp16
mixed probs before the single final V dot
```

If the `BM=32,KL=256` one-CTA/four-head design spills heavily, try either
`BM=16,W=240,KL=256` or the cluster/DSM design below.

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

Expose one narrow wide-window function:

```python
dc_hopper_cuda.forward_hpg4_wide_ref(
    q, k, v,
    pre_w1, pre_w2, pre_dd,
    post_w1, post_w2, post_dd,
    scaling,
    window,
    chunk_size,
)
```

for `BM=32,W=224` and `BM=16,W=240`.

Shape checks should reject everything except the target configs above.
`bench_onekernel_h100.py` should only add the CUDA column after the extension
returns correct output on the debug benchmark.

## Correctness gate

Before timing on H100:

```text
B in {1, 2}
T in {128, 256, 512}
BM=32
W in {224}
or BM=16,W=240
G=8
HPG=4
compare against dc_attention_torch or existing V4HCM/V4HCM256
max error target: same order as Triton V4HCM fp16 path
```

Only after this gate should the kernel be inserted into
`bench_onekernel_h100.py`.

Initial scalar scaffold result from H100 before dropping the small-window target:

```text
B=1,T=128,W=96 : max=3.22e-2, mean=5.52e-4
B=1,T=256,W=224: max=3.52e-2, mean=5.53e-4
B=1,T=512,W=96 : max=3.32e-2, mean=5.41e-4
B=1,T=512,W=224: max=9.38e-2, mean=5.55e-4
B=2,T=128,W=96 : max=4.74e-2, mean=5.64e-4
```

Interpretation:
- Mean error is stable around `5.5e-4`, so group/head/window indexing is likely
  correct.
- The larger max error on `T=512,W=224` is consistent with fp16 cache and
  different reduction/softmax order sensitivity.
- Use `mean <= 1e-3` and `max <= 1.2e-1` as the current scaffold pass gate.
- New correctness runs should focus on `BM=32,W=224` and `BM=16,W=240`.

Wide-window scaffold result after retargeting to `KL=256`:

```text
B=1,T=256,BM=32,W=224: max=5.86e-2, mean=5.57e-4 PASS
B=1,T=512,BM=32,W=224: max=4.81e-2, mean=5.61e-4 PASS
B=1,T=256,BM=16,W=240: max=3.13e-2, mean=5.58e-4 PASS
B=1,T=512,BM=16,W=240: max=5.86e-2, mean=5.55e-4 PASS
B=2,T=256,BM=32,W=224: max=6.25e-2, mean=5.63e-4 PASS
B=2,T=512,BM=32,W=224: max=4.69e-2, mean=5.53e-4 PASS
B=2,T=256,BM=16,W=240: max=6.05e-2, mean=5.54e-4 PASS
B=2,T=512,BM=16,W=240: max=4.69e-2, mean=5.61e-4 PASS
```

Conclusion:
- The `KL=256` API, head/group indexing, and wide-window masks are validated.
- Next implementation work should not change this reference path; add a separate
  optimized entry and compare against `forward_hpg4_wide_ref`.

First tensor-core experiment:
- Added `forward_hpg4_wide_opt`.
- Current scope is only `BM=16,W=240,KL=256`.
- It uses CUDA WMMA for QK and final `mixed @ V`, while leaving DC-pre,
  softmax, and `a_acc` as scalar/shared-memory code.
- This is a stepping stone to judge the one-CTA HPG=4 dataflow; it is not the
  final FA3-style WGMMA/TMA pipeline.
- Validate with `test_dc_hopper_cuda.py --opt`, then benchmark with
  `bench_dc_hopper_cuda.py`.

H100 benchmark result:

```text
B= 8:  V4HCM256  2219us, CUDAopt  32117us, opt/V4 14.48x
B=16:  V4HCM256  4403us, CUDAopt  64405us, opt/V4 14.63x
B=32:  V4HCM256  8781us, CUDAopt 129554us, opt/V4 14.75x
B=64:  V4HCM256 17545us, CUDAopt 278217us, opt/V4 15.86x
```

Conclusion:
- The naive one-CTA WMMA/scalar hybrid is structurally wrong for performance.
- Moving only QK/final V dot to WMMA is not enough; scalar softmax and
  shared-memory staging dominate, and there is no FA3-style TMA/producer-
  consumer overlap.
- Do not spend more time micro-optimizing this WMMA path.
- Keep it only as a diagnostic negative result; the next implementation must
  fork the FA3 Hopper pipeline or use Hopper cluster/DSM.

Cluster/DSM smoke test:
- Added `cluster_dsm_smoke(num_clusters)`.
- It launches clusters with `clusterDim.x = 4`.
- Each CTA writes its rank value into shared memory, synchronizes the cluster,
  then reads all four CTA shared-memory values through DSM.
- Expected output is `[num_clusters, 4]` filled with `10.0`.
- This validates the PyTorch extension launch path for Hopper cluster/DSM before
  implementing DC cross-head exchange.

Short Triton probe before the CUDA rewrite:
- Added `TritonDCOneKernelMixedProbs256Narrow` (`V4HCM256N`) to reduce
  `KL=256` live-state pressure by keeping `s_acc/a_acc` in fp16.
- It keeps the `V4HCM256` mixed-output identity and only changes accumulator
  precision.
- Local A800 sanity vs `V4HCM256`:

```text
BM=16,W=240: max=4.69e-2, mean=3.50e-4
BM=32,W=224: max=3.13e-2, mean=3.47e-4
```

- Short A800 latency showed a small gain on `BM=16,W=240` and near tie on
  `BM=32,W=224`; H100 should decide whether this remains in the table.
- H100 retest showed a regression: about `+2.3%` for `BM=16,W=240` and `+3.0%`
  for `BM=32,W=224`. Keep it in the A800 benchmark only; remove it from the
  H100 main table.

Cluster/DSM diagnostic forward:
- Added `forward_hpg4_wide_cluster`.
- Scope: `G=8,HPG=4,KL=256`, with `BM=16,W=240` and `BM=32,W=224`.
- Launches `clusterDim.x=4`; each CTA rank owns one head in the HPG group.
- DSM exchange points:
  1. after per-head QK tile construction, each CTA reads the four QK tiles to
     build its DC-pre score and softmax;
  2. after per-head probability construction, each CTA reads the four probs
     tiles to build post `a_acc` for final mixed AV.
- Current inner loops are scalar. This is a correctness/structure probe, not
  the final performance path. If it passes, replace the scalar QK/PV sections
  with a FA3/CuTe WGMMA/TMA mainloop.

## Performance gate

The first useful H100 target is:

```text
BM=32,W=224:
  first target <= 3.0x FA3
  later target ~= A800 overhead band, around 2.2x FA2/FA3-relative

BM=16,W=240:
  secondary target; useful if W=240 is required, but expected to be slower than
  BM=32,W=224 on H100 because the M tile is smaller.
```

If the forked FA3 mainloop cannot pass the small-window target without heavy
spilling, switch to the cluster/DSM design.
