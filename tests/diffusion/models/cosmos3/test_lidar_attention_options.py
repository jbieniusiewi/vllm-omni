# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The LiDAR neighborhood-attention launch options keep the reference numerics."""

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3 import lidar_vae

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def test_cpu_keeps_reference_options():
    assert lidar_vae._kernel_options(64, torch.device("cpu")) == lidar_vae._KERNEL_OPTIONS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention kernels need CUDA")
@pytest.mark.parametrize("head_dim", [8, 16, 32, 64, 128])
def test_kernel_options_keep_block_sizes(head_dim):
    options = lidar_vae._kernel_options(head_dim, torch.empty(0, device="cuda").device)
    for name in ("BLOCK_M", "BLOCK_N", "USE_TMA", "FORCE_USE_FLEX_ATTENTION", "FLOAT32_PRECISION"):
        assert options[name] == lidar_vae._KERNEL_OPTIONS[name]
    assert options["num_warps"] in (4, 8)
    assert options["num_stages"] in (1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention kernels need CUDA")
@pytest.mark.parametrize(
    ("heads", "height", "width", "head_dim", "dilation"),
    [(4, 32, 120, 32, (1, 1)), (8, 16, 112, 64, (1, 1)), (4, 16, 96, 64, (1, 2)), (2, 16, 72, 128, (1, 1))],
)
def test_tuned_options_are_bitwise_identical(heads, height, width, head_dim, dilation):
    from torch.nn.attention.flex_attention import flex_attention

    generator = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(2, heads, height * width, head_dim, device="cuda", generator=generator) for _ in range(3))
    mask = lidar_vae._get_block_mask(q.device, height, width, (5, 15), dilation)

    def attend(options):
        def run(q, k, v):
            return flex_attention(q, k, v, block_mask=mask, scale=1.0, kernel_options=options)

        return torch.compile(run, fullgraph=True, dynamic=False)(q, k, v)

    reference = attend(dict(lidar_vae._KERNEL_OPTIONS))
    torch.compiler.reset()
    assert torch.equal(attend(lidar_vae._kernel_options(head_dim, q.device)), reference)
