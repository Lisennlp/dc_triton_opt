# DC Hopper CUDA Experimental Extension

This directory is the CUDA landing zone for the post-V4H Triton work.

The initial implementation is intentionally a narrow correctness scaffold:

- `D=128`
- `N=32`
- `G=8`
- `HPG=4`
- `BM=32`
- `W in {96, 224}`
- fp16 tensors
- contiguous `[B, T, N, D]` Q/K/V and `[B, T, N]` DC weights

`forward_hpg4_bm32_ref` is a scalar CUDA reference kernel, not the performance
target. It exists to lock down the C++/Python extension API and correctness
tests before replacing the inner loops with a Hopper WGMMA/TMA or cluster/DSM
mainloop.

Build on an H100 machine:

```bash
cd dc_triton_opt/dc_hopper_cuda
/home/lishengping/miniconda3/bin/python setup.py build_ext --inplace
```

Do not add this extension to `bench_onekernel_h100.py` until the WGMMA/TMA
version passes the correctness gate in `../fa3_dc_hopper_plan.md`.
