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

Run the cluster/DSM diagnostic forward:

```bash
cd ..
CUDA_VISIBLE_DEVICES=1 /home/lishengping/miniconda3/bin/python test_dc_hopper_cuda.py --cluster
```

Run the isolated microbenchmark:

```bash
cd ..
CUDA_VISIBLE_DEVICES=1 /home/lishengping/miniconda3/bin/python bench_dc_hopper_cuda.py
```

Current status: the WMMA/scalar hybrid is a negative diagnostic result. It is
much slower than V4HCM256 on H100 and should not be optimized further.
`forward_hpg4_wide_cluster` is the current cluster/DSM structure probe: four
CTAs in one cluster own the four HPG heads and exchange QK/probability tiles
through DSM. It still uses scalar inner loops, so it is a stepping stone toward
WGMMA/TMA rather than the final performance kernel.

Run the Hopper cluster/DSM smoke test:

```bash
cd ..
CUDA_VISIBLE_DEVICES=1 /home/lishengping/miniconda3/bin/python test_dc_hopper_cluster.py
```

Expected output is an `[8, 4]` tensor filled with `10.0`. This only validates
cluster launch, synchronization, and distributed shared-memory reads; it is not
a DC performance kernel.
