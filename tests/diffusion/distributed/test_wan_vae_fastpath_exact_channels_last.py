# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU bit-exactness tests for the lossless Wan VAE decoder's channels-last activations."""

from __future__ import annotations

import pytest
import torch
from diffusers.models.autoencoders import AutoencoderKLWan, autoencoder_kl_wan

from tests.diffusion.distributed.wan_vae_fastpath_helpers import (  # noqa: F401
    original_wan_rms_norm,
    unpatched_wan_rms_norm,
)
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import install_wan_vae_fastpath
from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import triton_rms_norm_cl_exact as cle
from vllm_omni.diffusion.layers.vae_rms_norm import rms_norm_fastpath

pytestmark = [
    pytest.mark.usefixtures("unpatched_wan_rms_norm"),
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.diffusion,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

# Cosmos3 480p decoder stages: every ATen split (16, 11, 6, 3 and 1 CTAs per output) occurs.
DECODER_SHAPES = [
    (1, 1024, 1, 30, 52),
    (1, 1024, 4, 60, 104),
    (1, 1024, 2, 120, 208),
    (1, 1024, 4, 120, 208),
    (1, 512, 4, 120, 208),
    (1, 256, 2, 240, 416),
]


def _mixed_magnitudes(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    # Wide per-element magnitudes make any other summation order visible in the fp32 denominators.
    x = torch.randn(shape, device="cuda") * torch.rand(shape, device="cuda").pow(4) * 10
    return x.to(dtype)


@torch.no_grad()
@pytest.mark.parametrize("shape", DECODER_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("silu", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_channels_last_norm_matches_lossless_channels_first(
    shape: tuple[int, ...], dtype: torch.dtype, silu: bool, with_bias: bool
) -> None:
    torch.manual_seed(0)
    norm = autoencoder_kl_wan.WanRMS_norm(shape[1], images=False).to(device="cuda", dtype=dtype)
    norm.gamma.normal_()
    x = _mixed_magnitudes(shape, dtype)
    bias = torch.randn(shape[1], device="cuda").to(dtype) if with_bias else None

    expected = rms_norm_fastpath(norm, x, silu=silu, bias=bias)
    actual = cle.rms_norm_channels_last_exact(
        x.contiguous(memory_format=torch.channels_last_3d), norm.gamma, norm.scale, silu=silu, bias=bias
    )

    assert actual is not None
    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(actual.contiguous().view(torch.int16), expected.view(torch.int16))


@torch.no_grad()
def test_channels_last_norm_declines_channels_first_input() -> None:
    norm = autoencoder_kl_wan.WanRMS_norm(64, images=False).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, 64, 2, 8, 8, device="cuda", dtype=torch.bfloat16)
    assert cle.rms_norm_channels_last_exact(x, norm.gamma, norm.scale) is None


def test_reduce_config_follows_aten_split() -> None:
    device = torch.device("cuda")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    # 1560 positions x 1024 channels: one warp row of 32 x 4 outputs per CTA, split over 4 warps
    # and, while the grid is small, over min(ceil(target / grid), 1024 / 4 / 16) CTAs.
    block_height, ctas = cle.aten_norm_reduce_config(1024, 1560, device)
    assert block_height == 4
    assert ctas == max(min(-(-sms * 16 // 13), 16), 1)
    # Few values per thread: no global split.
    assert cle.aten_norm_reduce_config(256, 4 * 240 * 416, device) == (4, 1)


@torch.no_grad()
def test_wan21_decoder_keeps_channels_first_weights() -> None:
    vae = AutoencoderKLWan(base_dim=8, z_dim=4, dim_mult=[1, 2], num_res_blocks=1, temperal_downsample=[False, True])
    vae = vae.eval().to(device="cuda", dtype=torch.bfloat16)
    report = install_wan_vae_fastpath(vae, level="lossless")
    assert report.installed and not report.channels_last
    assert all(m.weight.is_contiguous() for m in vae.decoder.modules() if isinstance(m, torch.nn.Conv3d))
