# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bit-exact fused ops (q/k RMSNorm, RoPE, SwiGLU) for the Cosmos3 eager transformer path."""

from __future__ import annotations

from functools import cache

import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:
    from triton.language.extra import libdevice

if HAS_TRITON:

    @triton.jit
    def _round_to(value, dtype: tl.constexpr):
        # Opaque to the compiler: a plain down/up-cast pair can be folded away,
        # dropping the rounding the eager op performs between mul and add.
        if dtype == tl.bfloat16:
            return tl.inline_asm_elementwise(
                "{ .reg .b16 r, z; cvt.rn.bf16.f32 r, $1; mov.b16 z, 0; mov.b32 $0, {z, r}; }",
                "=r,r",
                [value],
                dtype=tl.float32,
                is_pure=True,
                pack=1,
            )
        return tl.inline_asm_elementwise(
            "{ .reg .b16 r; cvt.rn.f16.f32 r, $1; cvt.f32.f16 $0, r; }",
            "=r,r",
            [value],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _rope_rotate_half_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        seq_len,
        freqs_batch_stride,
        row_size: tl.constexpr,
        head_dim: tl.constexpr,
        block: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, block)
        mask = columns < row_size
        dim = columns % head_dim
        half = head_dim // 2
        first = dim < half
        base = x_ptr + row.to(tl.int64) * row_size
        x = tl.load(base + columns, mask=mask, other=0.0)
        partner = tl.load(base + tl.where(first, columns + half, columns - half), mask=mask, other=0.0)
        freqs = (row // seq_len) * freqs_batch_stride + (row % seq_len) * head_dim + dim
        cos = tl.load(cos_ptr + freqs, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + freqs, mask=mask, other=0.0)
        # Round after each eager op: x * cos, rotate_half(x) * sin, then their sum.
        x_cos = _round_to(x.to(tl.float32) * cos.to(tl.float32), x.dtype)
        rotated = tl.where(first, -partner.to(tl.float32), partner.to(tl.float32))
        rotated_sin = _round_to(rotated * sin.to(tl.float32), x.dtype)
        out = x_cos + rotated_sin
        tl.store(out_ptr + row.to(tl.int64) * row_size + columns, out.to(x.dtype), mask=mask)

    @triton.jit
    def _silu_mul_kernel(gate_ptr, up_ptr, out_ptr, numel, block: tl.constexpr):
        offsets = tl.program_id(0).to(tl.int64) * block + tl.arange(0, block)
        mask = offsets < numel
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0)
        # F.silu rounds x / (1 + exp(-x)) (IEEE division, accurate expf) to the
        # input dtype before the product with ``up`` rounds again.
        gate_f32 = gate.to(tl.float32)
        silu = _round_to(tl.math.div_rn(gate_f32, 1.0 + libdevice.exp(-gate_f32)), gate.dtype)
        tl.store(out_ptr + offsets, (silu * up.to(tl.float32)).to(gate.dtype), mask=mask)

    @triton.jit
    def _head_rms_norm_kernel(x_ptr, weight_ptr, out_ptr, rows, eps, block_rows: tl.constexpr):
        # ATen's vectorized RMS kernel for 128 columns runs one warp per row: lane
        # t accumulates the squares of columns 4t..4t+3 in order (exact in FP32
        # for 16-bit inputs), then a shuffle-down tree adds lanes t and t + 16,
        # t + 8, ... The same order here gives a bit-identical rstd.
        row = tl.program_id(0).to(tl.int64) * block_rows + tl.arange(0, block_rows)[:, None]
        columns = tl.arange(0, 128)[None, :]
        mask = row < rows
        x = tl.load(x_ptr + row * 128 + columns, mask=mask, other=0.0).to(tl.float32)
        # [rows, lane, 2, 2] -> the lane's columns 4t, 4t + 1, 4t + 2, 4t + 3.
        even, odd = tl.split(tl.reshape(x, (block_rows, 32, 2, 2)))
        x0, x2 = tl.split(even)
        x1, x3 = tl.split(odd)
        squares = x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3
        for width in tl.static_range(4, -1, -1):
            squares = tl.sum(tl.reshape(squares, (block_rows, 2, 1 << width)), axis=1)
        rstd = tl.reshape(libdevice.rsqrt(squares / 128.0 + eps), (block_rows, 1))
        weight = tl.load(weight_ptr + columns).to(tl.float32)
        out = weight * (rstd * x)
        tl.store(out_ptr + row * 128 + columns, out.to(out_ptr.dtype.element_ty), mask=mask)


@cache
def _supports_bf16_cvt(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) >= (8, 0)


def _rope_contract_ok(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> bool:
    if x.ndim != 4 or cos.ndim != 4 or cos.shape != sin.shape or cos.stride() != sin.stride():
        return False
    batch, seq_len, heads, head_dim = x.shape
    return (
        x.dtype in (torch.bfloat16, torch.float16)
        and cos.dtype == sin.dtype == x.dtype
        and x.is_cuda
        and cos.device == sin.device == x.device
        and x.is_contiguous()
        and cos.is_contiguous()
        and head_dim % 2 == 0
        and cos.shape[0] in (1, batch)
        and cos.shape[1:] == (seq_len, 1, head_dim)
        and 0 < x.numel() < 2**31
        and (x.dtype != torch.bfloat16 or _supports_bf16_cvt(x.device.index))
    )


def try_fused_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(q * cos) + (rotate_half(q) * sin)`` (and for ``k``) in one kernel each; ``None`` outside the contract.

    Every intermediate is rounded to the input dtype exactly where the eager
    expression rounds it, so the result is bit-identical to the unfused ops.
    """
    if not (
        HAS_TRITON
        and torch.version.hip is None
        and not torch.compiler.is_compiling()
        and not (torch.is_grad_enabled() and (q.requires_grad or k.requires_grad))
        and _rope_contract_ok(q, cos, sin)
        and _rope_contract_ok(k, cos, sin)
    ):
        return None
    outputs = []
    for x in (q, k):
        batch, seq_len, heads, head_dim = x.shape
        out = torch.empty_like(x)
        row_size = heads * head_dim
        _rope_rotate_half_kernel[(batch * seq_len,)](
            x,
            cos,
            sin,
            out,
            seq_len,
            0 if cos.shape[0] == 1 else seq_len * head_dim,
            row_size=row_size,
            head_dim=head_dim,
            block=triton.next_power_of_2(row_size),
        )
        outputs.append(out)
    return outputs[0], outputs[1]


#: Elements per program of :func:`try_fused_silu_mul`.
_SILU_MUL_BLOCK = 2048


def try_fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor | None:
    """``F.silu(gate) * up`` in one kernel; ``None`` outside the contract.

    Bit-identical to the two eager ops: the SiLU result is rounded to the input
    dtype before the product, as the eager intermediate tensor is.
    """
    if not (
        HAS_TRITON
        and torch.version.hip is None
        and not torch.compiler.is_compiling()
        and not (torch.is_grad_enabled() and (gate.requires_grad or up.requires_grad))
        and gate.dtype in (torch.bfloat16, torch.float16)
        and up.dtype == gate.dtype
        and gate.shape == up.shape
        and gate.is_cuda
        and up.device == gate.device
        and gate.is_contiguous()
        and up.is_contiguous()
        and gate.numel() > 0
        and (gate.dtype != torch.bfloat16 or _supports_bf16_cvt(gate.device.index))
    ):
        return None
    out = torch.empty_like(gate)
    numel = gate.numel()
    _silu_mul_kernel[(triton.cdiv(numel, _SILU_MUL_BLOCK),)](gate, up, out, numel, block=_SILU_MUL_BLOCK)
    return out


#: Head rows per program of :func:`head_rms_norm`.
_HEAD_RMS_NORM_ROWS = 16


def head_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``F.rms_norm(x, (128,), weight, eps)`` over 128-wide heads, fused on CUDA when bit-identical.

    Reproduces the reduction order of ATen's vectorized RMS kernel for 128
    columns (which that kernel uses for aligned 16-bit inputs), so the result
    equals ``F.rms_norm`` bit for bit; everything else calls ``F.rms_norm``.
    """
    if not (
        HAS_TRITON
        and torch.version.hip is None
        and not torch.compiler.is_compiling()
        and not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))
        and x.dtype in (torch.bfloat16, torch.float16)
        and weight.dtype == x.dtype
        and x.is_cuda
        and weight.device == x.device
        and x.ndim >= 1
        and x.shape[-1] == 128
        and weight.shape == (128,)
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and x.numel() > 0
    ):
        return F.rms_norm(x, (x.shape[-1],), weight, eps=eps)
    out = torch.empty_like(x)
    rows = x.numel() // 128
    _head_rms_norm_kernel[(triton.cdiv(rows, _HEAD_RMS_NORM_ROWS),)](
        x, weight, out, rows, eps, block_rows=_HEAD_RMS_NORM_ROWS
    )
    return out
