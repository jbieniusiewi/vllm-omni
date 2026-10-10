# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fused scatter-merge for the maskless Cosmos3 multiview LSE accumulator.

The streaming recurrence in :mod:`.multiview_maskless_attention` folds each
attention pass into an FP32 ``[tokens, heads, head_dim]`` accumulator.  Written
as tensor ops it costs three full traversals of that accumulator per chunk --
``index_select`` gathers the rows, a pointwise kernel applies the recurrence and
``index_copy_`` scatters them back -- and the accumulator is the largest tensor
in the denoise step, so those round trips dominate the merge.

This module runs the same arithmetic as one Triton kernel that addresses the
accumulator in place: each program loads the rows it owns, folds the pass in and
stores them back, so the accumulator is read once and written once.  The
expressions mirror :func:`~.multiview_maskless_attention.merge_step` operation
for operation (including ``log1p``/``exp``/``abs`` on the logsigmoid path) so
results are bit-identical to the tensor implementation, which remains the
reference on CPU and wherever Triton is unavailable.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

#: Query rows per Triton program.  The merge is memory bound, so this only has
#: to keep enough rows in flight to saturate the accumulator's bandwidth.
_ROWS_PER_PROGRAM = 16

_kernel = None
_unavailable = False
#: Stands in for ``index`` when the pass covers every row: ``IDENTITY`` is a
#: constexpr, so the kernel never dereferences this, but Triton still needs an
#: argument of pointer type to specialize on.
_UNUSED_INDEX: torch.Tensor | None = None


def _build_kernel():
    """Compile the fused merge once; ``None`` when Triton cannot provide it."""
    global _kernel, _unavailable
    if _kernel is not None or _unavailable:
        return _kernel
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice

        @triton.jit
        def _fused_merge(
            acc_ptr,
            acc_lse_ptr,
            out_ptr,
            lse_ptr,
            index_ptr,
            num_slots,
            heads,
            head_dim,
            IDENTITY: tl.constexpr,
            SEED: tl.constexpr,
            BLOCK_SLOTS: tl.constexpr,
            BLOCK_DIM: tl.constexpr,
        ):
            # One slot is a (row, head) pair: the recurrence shares its weight
            # across head_dim, which one program loads as a single tile.
            slot = tl.program_id(0) * BLOCK_SLOTS + tl.arange(0, BLOCK_SLOTS)
            slot_mask = slot < num_slots
            row = slot // heads
            head = slot - row * heads
            destination = row if IDENTITY else tl.load(index_ptr + row, slot_mask, other=0)

            dim = tl.arange(0, BLOCK_DIM)
            tile_mask = slot_mask[:, None] & (dim < head_dim)[None, :]
            acc_offset = (destination * heads + head)[:, None] * head_dim + dim[None, :]
            lse_offset = destination * heads + head

            out = tl.load(out_ptr + slot[:, None] * head_dim + dim[None, :], tile_mask).to(tl.float32)
            lse = tl.load(lse_ptr + slot, slot_mask)
            if SEED:
                # The first pass covers every row and seeds the accumulators.
                tl.store(acc_ptr + acc_offset, out, tile_mask)
                tl.store(acc_lse_ptr + lse_offset, lse, slot_mask)
            else:
                acc_lse = tl.load(acc_lse_ptr + lse_offset, slot_mask)
                weight = tl.sigmoid(lse - acc_lse)
                acc = tl.load(acc_ptr + acc_offset, tile_mask)
                tl.store(acc_ptr + acc_offset, acc - weight[:, None] * (acc - out), tile_mask)
                # logsigmoid(x) as min(0, x) - log1p(exp(-|x|)), matching the
                # tensor path's lowering so the FP32 result is bit-identical.
                shift = acc_lse - lse
                log_sigmoid = tl.minimum(0.0, shift) - libdevice.log1p(libdevice.exp(-libdevice.abs(shift)))
                tl.store(acc_lse_ptr + lse_offset, acc_lse - log_sigmoid, slot_mask)

        _kernel = (triton, _fused_merge)
    except Exception as error:  # pragma: no cover - depends on the Triton build
        _unavailable = True
        logger.info("Cosmos3 maskless: fused merge unavailable (%s); using the tensor merge.", error)
    return _kernel


def fused_merge_available(device: torch.device) -> bool:
    """Whether :func:`fused_merge` can serve ``device``."""
    return device.type == "cuda" and _build_kernel() is not None


def fused_merge(
    acc: torch.Tensor,
    acc_lse: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    index: torch.Tensor | None,
    *,
    seed: bool,
) -> None:
    """Fold ``out``/``lse`` into ``acc``/``acc_lse`` in place.

    ``index`` maps pass rows to accumulator rows, or is ``None`` when the pass
    covers every row in order.  ``seed`` writes the accumulators instead of
    folding into them, for the first pass.
    """
    built = _build_kernel()
    if built is None:
        raise RuntimeError("Cosmos3 maskless fused merge requires Triton.")
    triton, kernel = built
    rows, heads, head_dim = out.shape
    identity = index is None
    if identity:
        global _UNUSED_INDEX
        if _UNUSED_INDEX is None or _UNUSED_INDEX.device != out.device:
            _UNUSED_INDEX = torch.zeros(1, dtype=torch.int64, device=out.device)
    if not identity and index.numel() != rows:
        raise ValueError(f"Fused merge index covers {index.numel()} rows for {rows} pass rows.")
    num_slots = rows * heads
    kernel[(triton.cdiv(num_slots, _ROWS_PER_PROGRAM),)](
        acc,
        acc_lse,
        out,
        lse,
        _UNUSED_INDEX if identity else index,
        num_slots,
        heads,
        head_dim,
        IDENTITY=identity,
        SEED=seed,
        BLOCK_SLOTS=_ROWS_PER_PROGRAM,
        BLOCK_DIM=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
