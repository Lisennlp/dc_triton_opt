"""Experimental Hopper CUDA DC entry points."""

from __future__ import annotations

import importlib


def _ext():
    return importlib.import_module("_dc_hopper_cuda")


def forward_hpg4_bm32_ref(
    q,
    k,
    v,
    pre_w1,
    pre_w2,
    pre_dd,
    post_w1,
    post_w2,
    post_dd,
    scaling,
    window,
):
    """Scalar CUDA reference path for the fixed HPG=4/BM=32 Hopper target."""
    return _ext().forward_hpg4_bm32_ref(
        q,
        k,
        v,
        pre_w1,
        pre_w2,
        pre_dd,
        post_w1,
        post_w2,
        post_dd,
        float(scaling),
        int(window),
    )

