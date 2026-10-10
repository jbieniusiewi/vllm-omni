# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused gather/merge/scatter kernels for the maskless Cosmos3 merge.

The merge recurrence itself lives in
:func:`.multiview_maskless_attention.merge_step`; this module only changes how
it is *executed* on CUDA.  A pass whose queries are a permutation of the packed
GEN rows previously ran three kernels per chunk -- gather the FP32 accumulator
rows, evaluate the recurrence, scatter them back -- so every merge moved the
accumulator tile across HBM three times, and the scatter leg
(``Tensor.index_copy_``) is an unvectorized elementwise kernel that reaches a
fraction of the bandwidth the vectorized gather does.

The kernels here keep the accumulator in place and do the indirection in
registers: one pass over ``acc`` per merge instead of three.  The arithmetic is
emitted in the same order, with the same Triton primitives, as the Inductor
graph compiled from ``merge_step``, so results are bit-identical; the eager
recurrence remains the reference for CPU and the oracles.
"""

from __future__ import annotations

import torch

try:  # Triton ships with the CUDA builds the maskless backend needs.
    import triton
    import triton.language as tl
    from torch._inductor.runtime import triton_helpers
    from torch._inductor.runtime.triton_helpers import libdevice

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - CPU-only installs fall back to eager.
    HAVE_TRITON = False


#: Rows per program.  The inner tile is one head's ``head_dim`` row, so this
#: only trades occupancy against per-program index arithmetic.
_BLOCK_ROWS = 8


if HAVE_TRITON:

    @triton.jit
    def _merge_scatter_kernel(
        acc_ptr,
        acc_lse_ptr,
        out_ptr,
        lse_ptr,
        index_ptr,
        rows,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Fold one branch into ``acc``/``acc_lse`` at the rows ``index`` selects.

        Mirrors ``merge_step`` term by term.  Safe in place: the planner proves
        a pass never attends a query row twice, so the gathered rows are
        distinct and each program owns the rows it writes.
        """
        row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        head = tl.program_id(1)
        row_mask = row < rows
        target = tl.load(index_ptr + row, row_mask, other=0)

        lse_offsets = target * heads + head
        acc_lse = tl.load(acc_lse_ptr + lse_offsets, row_mask, other=0.0)
        lse = tl.load(lse_ptr + row * heads + head, row_mask, other=0.0)

        # acc = acc - sigmoid(lse - acc_lse) * (acc - out.float())
        weight = tl.sigmoid(lse - acc_lse)
        lane = tl.arange(0, head_dim)
        base = head * head_dim + lane[None, :]
        acc_offsets = target[:, None] * (heads * head_dim) + base
        out_offsets = row[:, None] * (heads * head_dim) + base
        tile_mask = row_mask[:, None]
        acc = tl.load(acc_ptr + acc_offsets, tile_mask, other=0.0)
        out = tl.load(out_ptr + out_offsets, tile_mask, other=0.0).to(tl.float32)
        tl.store(acc_ptr + acc_offsets, acc - weight[:, None] * (acc - out), tile_mask)

        # acc_lse = acc_lse - logsigmoid(acc_lse - lse)
        delta = acc_lse - lse
        log_sigmoid = triton_helpers.minimum(tl.full([1], 0.0, tl.float32), delta) - libdevice.log1p(
            libdevice.exp(-tl.abs(delta))
        )
        tl.store(acc_lse_ptr + lse_offsets, acc_lse - log_sigmoid, row_mask)

    @triton.jit
    def _seed_scatter_kernel(
        acc_ptr,
        acc_lse_ptr,
        out_ptr,
        lse_ptr,
        index_ptr,
        rows,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Seed ``acc``/``acc_lse`` from the first pass at the rows ``index`` selects."""
        row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        head = tl.program_id(1)
        row_mask = row < rows
        target = tl.load(index_ptr + row, row_mask, other=0)

        lse = tl.load(lse_ptr + row * heads + head, row_mask, other=0.0)
        tl.store(acc_lse_ptr + target * heads + head, lse, row_mask)

        lane = tl.arange(0, head_dim)
        base = head * head_dim + lane[None, :]
        tile_mask = row_mask[:, None]
        out = tl.load(out_ptr + row[:, None] * (heads * head_dim) + base, tile_mask, other=0.0)
        tl.store(
            acc_ptr + target[:, None] * (heads * head_dim) + base,
            out.to(tl.float32),
            tile_mask,
        )


def can_fuse(acc: torch.Tensor, out: torch.Tensor, index: torch.Tensor) -> bool:
    """Whether the fused kernels apply to this pass.

    Requires CUDA with Triton and the contiguous ``[rows, heads, head_dim]``
    layout the pass kernels return; anything else keeps the eager path.
    """
    return (
        HAVE_TRITON
        and acc.is_cuda
        and acc.dtype == torch.float32
        and acc.is_contiguous()
        and out.is_contiguous()
        and index.is_contiguous()
        and index.dtype == torch.int64
        and out.shape[1:] == acc.shape[1:]
    )


def _launch(kernel, acc, acc_lse, out, lse, index) -> None:
    rows, heads, head_dim = out.shape
    kernel[(triton.cdiv(rows, _BLOCK_ROWS), heads)](
        acc,
        acc_lse,
        out,
        lse,
        index,
        rows,
        heads=heads,
        head_dim=head_dim,
        BLOCK_ROWS=_BLOCK_ROWS,
    )


def seed_scatter(
    acc: torch.Tensor, acc_lse: torch.Tensor, out: torch.Tensor, lse: torch.Tensor, index: torch.Tensor
) -> None:
    """``acc[index] = out.float(); acc_lse[index] = lse`` in one pass."""
    _launch(_seed_scatter_kernel, acc, acc_lse, out, lse, index)


def merge_scatter(
    acc: torch.Tensor, acc_lse: torch.Tensor, out: torch.Tensor, lse: torch.Tensor, index: torch.Tensor
) -> None:
    """Apply ``merge_step`` to the ``index`` rows of ``acc``/``acc_lse`` in place."""
    _launch(_merge_scatter_kernel, acc, acc_lse, out, lse, index)
