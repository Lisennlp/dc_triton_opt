"""Hopper cluster/DSM smoke test for dc_hopper_cuda.

Build first:
  cd dc_hopper_cuda
  /home/lishengping/miniconda3/bin/python setup.py build_ext --inplace

Run on H100:
  CUDA_VISIBLE_DEVICES=1 /home/lishengping/miniconda3/bin/python test_dc_hopper_cluster.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "dc_hopper_cuda"))

import dc_hopper_cuda  # noqa: E402


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        raise RuntimeError(f"this test needs sm90+; current device is sm{major}{minor}")

    out = dc_hopper_cuda.cluster_dsm_smoke(8)
    torch.cuda.synchronize()
    expected = torch.full_like(out, 10.0)
    diff = (out - expected).abs()
    print(out)
    print(f"max={diff.max().item():.4e} mean={diff.mean().item():.4e}")
    if diff.max().item() != 0.0:
        raise RuntimeError("cluster DSM smoke test failed")


if __name__ == "__main__":
    main()
