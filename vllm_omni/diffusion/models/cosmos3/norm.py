# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch
from torch.distributed.tensor import DTensor
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:
    from triton.language.extra import libdevice

from vllm_omni.diffusion.layers.norm import RMSNorm as _VllmRMSNorm

#: Elements per program of the elementwise kernels below.
_BLOCK_SIZE = 2048
#: Rows per program of the single-pass 4096-wide kernel.
_ROWS_4096 = 1

if HAS_TRITON:

    @triton.jit
    def _square_f32_kernel(x_ptr, out_ptr, numel, block: tl.constexpr):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + offsets, x * x, mask=mask)

    @triton.jit
    def _rms_norm_apply_kernel(
        x_ptr, rstd_ptr, weight_ptr, out_ptr, numel, hidden_size: tl.constexpr, block: tl.constexpr
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < numel
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + offsets // hidden_size, mask=mask, other=0.0)
        weight = tl.load(weight_ptr + offsets % hidden_size, mask=mask, other=0.0).to(tl.float32)
        # Same FP32 products and single final rounding as the native path.
        out = weight * (x * rstd)
        tl.store(out_ptr + offsets, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rms_norm_4096_kernel(x_ptr, weight_ptr, out_ptr, rows, eps, block_rows: tl.constexpr):
        # ATen's mean over 4096 contiguous FP32 values (>= 16 rows) runs one warp
        # per row: lane t keeps four accumulators of columns 128 j + 4 t + i,
        # summed over j in order, combines them as ((a0 + a1) + a2) + a3, then a
        # shuffle-down tree adds lanes t + 16, t + 8, ... Squares of 16-bit
        # inputs are exact in FP32, so the same order gives the same variance.
        row = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)[:, None]
        columns = tl.arange(0, 128)[None, :]
        mask = row < rows
        base = row * 4096 + columns
        acc = tl.zeros((block_rows, 128), dtype=tl.float32)
        for chunk in range(32):
            x = tl.load(x_ptr + base + chunk * 128, mask=mask, other=0.0).to(tl.float32)
            acc += x * x
        even, odd = tl.split(tl.reshape(acc, (block_rows, 32, 2, 2)))
        acc0, acc2 = tl.split(even)
        acc1, acc3 = tl.split(odd)
        total = ((acc0 + acc1) + acc2) + acc3
        for width in tl.static_range(4, -1, -1):
            total = tl.sum(tl.reshape(total, (block_rows, 2, 1 << width)), axis=1)
        rstd = tl.reshape(libdevice.rsqrt(total * (1.0 / 4096) + eps), (block_rows, 1))
        for chunk in range(32):
            x = tl.load(x_ptr + base + chunk * 128, mask=mask, other=0.0).to(tl.float32)
            weight = tl.load(weight_ptr + columns + chunk * 128).to(tl.float32)
            out = weight * (x * rstd)
            tl.store(out_ptr + base + chunk * 128, out.to(out_ptr.dtype.element_ty), mask=mask)


def _fused_rms_norm_native(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor | None:
    """Bit-exact ``forward_native`` with its elementwise steps fused; ``None`` outside the contract.

    The reduction order decides the variance rounding. For 16-bit rows of 4096
    (at least 16 rows, where ATen reduces each row in one warp) one kernel
    reproduces that order. Otherwise the variance keeps ATen's ``mean`` over the
    FP32 squares, while the up-cast/square and the ``weight * (x * rsqrt)``
    epilogue with the final cast each run as one kernel instead of
    materializing every FP32 intermediate.
    """
    if not (
        HAS_TRITON
        and torch.version.hip is None
        and not torch.compiler.is_compiling()
        and not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))
        and not isinstance(weight, DTensor)
        and x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.ndim >= 1
        and 0 < x.numel() < 2**31
        and x.is_contiguous()
        and weight.device == x.device
        and weight.is_contiguous()
        and weight.shape == x.shape[-1:]
    ):
        return None
    numel = x.numel()
    rows = numel // x.shape[-1]
    # FP32 squares round (and may contract into FMAs), so FP32 inputs keep ATen's mean.
    if x.shape[-1] == 4096 and rows >= 16 and x.dtype != torch.float32:
        out = torch.empty_like(x)
        _rms_norm_4096_kernel[(triton.cdiv(rows, _ROWS_4096),)](x, weight, out, rows, eps, block_rows=_ROWS_4096)
        return out
    grid = (triton.cdiv(numel, _BLOCK_SIZE),)
    squares = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    _square_f32_kernel[grid](x, squares, numel, block=_BLOCK_SIZE)
    rstd = torch.rsqrt(squares.mean(-1, keepdim=True) + eps)
    del squares
    out = torch.empty_like(x)
    _rms_norm_apply_kernel[grid](x, rstd, weight, out, numel, hidden_size=x.shape[-1], block=_BLOCK_SIZE)
    return out


class RMSNorm(_VllmRMSNorm):
    """Cosmos3 RMSNorm that uses the FP32 native implementation."""

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        out = _fused_rms_norm_native(x, self.weight, self.variance_epsilon)
        return self.forward_native(x) if out is None else out

    def forward_hip(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)

    def forward_npu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)
