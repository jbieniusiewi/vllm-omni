# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Cosmos3 fused CUDA ops must stay bit-identical to the eager ops they replace."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.cosmos3.fused_ops import (
    head_rms_norm,
    try_fused_rotary_pos_emb,
    try_fused_silu_mul,
)
from vllm_omni.diffusion.models.cosmos3.norm import RMSNorm
from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import _rotate_half

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.nvidia_only,
    pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip is not None, reason="requires CUDA"),
]

DTYPES = [torch.bfloat16, torch.float16]


def _mixed_magnitudes(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Rows whose squares span many binades, so any other summation order changes the sum."""
    scale = torch.exp2(torch.randint(-10, 10, shape, device="cuda").float())
    if dtype == torch.float16:
        scale = scale * 1e-2
    return (torch.randn(shape, device="cuda") * scale).to(dtype)


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", [16, 37, 3634])
@pytest.mark.parametrize("mixed", [False, True])
@torch.no_grad()
def test_rms_norm_matches_native(dtype: torch.dtype, rows: int, mixed: bool) -> None:
    torch.manual_seed(0)
    for hidden in (4096, 1000, 128):
        norm = RMSNorm(hidden, eps=1e-6).cuda().to(dtype)
        norm.weight.copy_(torch.randn(hidden) * 0.5 + 1)
        x = (
            _mixed_magnitudes((1, rows, hidden), dtype)
            if mixed
            else (torch.randn(1, rows, hidden) * 3).to("cuda", dtype)
        )
        _assert_bitwise_equal(norm.forward_cuda(x), norm.forward_native(x))


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mixed", [False, True])
@torch.no_grad()
def test_head_rms_norm_matches_f_rms_norm(dtype: torch.dtype, mixed: bool) -> None:
    torch.manual_seed(0)
    weight = (torch.randn(128, device="cuda") * 0.3 + 1).to(dtype)
    for shape in ((1, 3634, 32, 128), (2, 77, 8, 128), (1, 1, 1, 128)):
        x = _mixed_magnitudes(shape, dtype) if mixed else (torch.randn(shape, device="cuda") * 3).to(dtype)
        for eps in (1e-6, 1e-5):
            _assert_bitwise_equal(head_rms_norm(x, weight, eps), F.rms_norm(x, (128,), weight, eps=eps))


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.no_grad()
def test_rotary_pos_emb_matches_eager(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    for batch, seq_len, freqs_batch in ((1, 3634, 1), (2, 100, 1), (2, 100, 2)):
        q = (torch.randn(batch, seq_len, 32, 128, device="cuda") * 3).to(dtype)
        k = (torch.randn(batch, seq_len, 8, 128, device="cuda") * 3).to(dtype)
        angles = torch.randn(freqs_batch, seq_len, 1, 128, device="cuda") * 100
        cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
        fused = try_fused_rotary_pos_emb(q, k, cos, sin)
        assert fused is not None
        _assert_bitwise_equal(fused[0], (q * cos) + (_rotate_half(q) * sin))
        _assert_bitwise_equal(fused[1], (k * cos) + (_rotate_half(k) * sin))


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.no_grad()
def test_silu_mul_matches_eager(dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    # Every finite value of the 16-bit dtype, then random products.
    gate = torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16).view(dtype).cuda()
    gate = gate[torch.isfinite(gate)]
    fused = try_fused_silu_mul(gate, torch.ones_like(gate))
    assert fused is not None
    _assert_bitwise_equal(fused, F.silu(gate))
    gate = (torch.randn(3, 4097, device="cuda") * 4).to(dtype)
    up = (torch.randn(3, 4097, device="cuda") * 4).to(dtype)
    fused = try_fused_silu_mul(gate, up)
    assert fused is not None
    _assert_bitwise_equal(fused, F.silu(gate) * up)
