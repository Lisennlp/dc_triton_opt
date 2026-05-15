# DC Hopper CUDA Experimental Extension

This directory is the CUDA landing zone for the post-V4H Triton work.

The initial implementation is intentionally a narrow correctness scaffold:

- `D=128`
- `N=32`
- `G=8`
- `HPG=4`
- `KL=256`
- `BM=32,W=224`
- `BM=16,W=240`
- fp16 tensors
- contiguous `[B, T, N, D]` Q/K/V and `[B, T, N]` DC weights

`forward_hpg4_wide_ref` is a scalar CUDA reference kernel, not the performance
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

Run correctness:

```bash
cd ..
CUDA_VISIBLE_DEVICES=1 /home/lishengping/miniconda3/bin/python test_dc_hopper_cuda.py
```

Run the first tensor-core experiment:

```bash
cd ..
CUDA_VISIBLE_DEVICES=1 /home/lishengping/miniconda3/bin/python test_dc_hopper_cuda.py --opt
```

If `--opt` passes, run the isolated microbenchmark:

```bash
cd ..
CUDA_VISIBLE_DEVICES=1 /home/lishengping/miniconda3/bin/python bench_dc_hopper_cuda.py
```
