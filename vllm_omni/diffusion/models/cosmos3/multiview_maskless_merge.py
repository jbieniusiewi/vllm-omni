# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-kernel log-sum-exp merge for the maskless Cosmos3 multiview backend.

The sequential merge walks one pass at a time, so the FP32 accumulator over
every GEN row (``[tokens, heads, head_dim]``) is gathered out of DRAM and
scattered back for each of the five-or-so passes.  At the released 11-view
geometry that accumulator is ~7 GiB and the loop touches it a dozen times,
which costs more memory traffic than the merge arithmetic itself.

The kernel here keeps one row's accumulator in registers and folds every pass
into it before storing the result once, in the query dtype.  Each pass
contributes through an inverse permutation (``acc`` row -> that pass's row),
derived from the plan's forward ``q_index`` and cached with it, plus a coverage
flag for the passes that do not span every row.  Uncovered rows merge an LSE of
``-inf``, which is the identity of the recurrence and reproduces exactly what
skipping the row does in the sequential path.

A program owns a ``[heads_per_program, head_dim]`` tile rather than a single
head, so each gather reads a contiguous span of the pass output row instead of
one head's slice of it.  The whole tile shares the row's inverse lookup, which
is what makes the wider tile cheaper: measured ~2.2x over one head per program
at every head count the model presents.

The arithmetic deliberately mirrors ``merge_step`` operation for operation --
``sigmoid``, ``minimum``/``log1p``/``exp``/``abs`` for ``logsigmoid`` -- in the
same order and all in FP32, so the merged result is bit-identical to the
sequential path and the output md5 does not move.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

#: Warps per program.  One was the fastest of the tried configurations at every
#: tile width: the merge is bound by the gather, not by arithmetic throughput.
_MERGE_NUM_WARPS = 1

#: Head dims the kernel accepts.  ``tl.arange`` needs a power-of-two extent and
#: a whole head must fit in one program's registers.
_MAX_MERGE_HEAD_DIM = 256

#: Largest ``heads_per_program * head_dim`` tile a program may hold.  Tiles at
#: this size kept every measured head count at its fastest; past it the extra
#: registers cost more than the wider contiguous gather wins.
_MAX_MERGE_TILE = 4096


def _heads_per_program(heads: int, head_dim: int) -> int:
    """Largest head block that tiles ``heads`` exactly and fits the budget.

    ``tl.arange`` needs a power-of-two extent and the grid divides ``heads`` by
    the block, so only power-of-two divisors of ``heads`` are admissible. A head
    count with an odd factor (an unusual TP split) simply gets a smaller block.
    """
    block = 1
    while block * 2 <= heads and heads % (block * 2) == 0 and block * 2 * head_dim <= _MAX_MERGE_TILE:
        block *= 2
    return block

#: Bytes of device memory left free when deciding whether the fused merge fits.
#: The fused path holds every pass output at once instead of one at a time, and
#: trades the FP32 accumulator away; keep a margin so a geometry that would run
#: the allocator dry falls back instead of failing the request.
_MERGE_FREE_MEMORY_MARGIN = 2 << 30

_TRITON_MERGE: Any = None
_TRITON_UNAVAILABLE = False


def _load_triton_merge() -> Any:
    """Compile the merge kernel once, or report Triton as unusable."""
    global _TRITON_MERGE, _TRITON_UNAVAILABLE
    if _TRITON_MERGE is not None or _TRITON_UNAVAILABLE:
        return _TRITON_MERGE
    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice
    except Exception as exc:  # noqa: BLE001 - any import failure means fall back
        logger.debug("Cosmos3 maskless fused merge needs Triton: %s", exc)
        _TRITON_UNAVAILABLE = True
        return None

    @triton.jit
    def _merge_passes_kernel(
        out_ptr,
        pass_outputs,
        pass_lses,
        inverse_indices,
        coverage_flags,
        num_rows,
        NUM_PASSES: tl.constexpr,
        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HEAD_BLOCK: tl.constexpr,
    ):
        blocks_per_row: tl.constexpr = HEADS // HEAD_BLOCK
        program = tl.program_id(0)
        row = program // blocks_per_row
        block = program % blocks_per_row
        if row >= num_rows:
            return
        head = block * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
        lane = tl.arange(0, HEAD_DIM)
        # Contiguous within the row: heads are the outer axis of [heads, head_dim].
        tile = head[:, None] * HEAD_DIM + lane[None, :]
        accumulator = tl.zeros([HEAD_BLOCK, HEAD_DIM], dtype=tl.float32)
        accumulator_lse = tl.zeros([HEAD_BLOCK, 1], dtype=tl.float32)
        for index in tl.static_range(NUM_PASSES):
            # One lookup per row serves the whole tile.
            source = tl.load(inverse_indices[index] + row)
            covered = tl.load(coverage_flags[index] + row)
            values = tl.load(pass_outputs[index] + source * (HEADS * HEAD_DIM) + tile)
            lse = tl.load(pass_lses[index] + source * HEADS + head)[:, None]
            if index == 0:
                # The first pass spans every row and seeds the accumulators.
                accumulator = values.to(tl.float32)
                accumulator_lse = lse
            else:
                # -inf is the recurrence identity: sigmoid(-inf) is 0, so the
                # accumulator and its LSE both stay exactly as they were.
                lse = tl.where(covered != 0, lse, float("-inf"))
                weight = tl.sigmoid(lse - accumulator_lse)
                accumulator = accumulator - weight * (accumulator - values.to(tl.float32))
                delta = accumulator_lse - lse
                log_sigmoid = tl.minimum(0.0, delta) - libdevice.log1p(libdevice.exp(-libdevice.abs(delta)))
                accumulator_lse = accumulator_lse - log_sigmoid
        tl.store(out_ptr + row * (HEADS * HEAD_DIM) + tile, accumulator.to(out_ptr.dtype.element_ty))

    _TRITON_MERGE = _merge_passes_kernel
    return _TRITON_MERGE


def _inverse_permutation(q_index: torch.Tensor, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each accumulator row to its row in one pass, plus a coverage flag.

    Rows the pass does not attend keep source ``0`` -- always a valid row, so
    the gather stays in bounds -- and are masked by the coverage flag instead.
    """
    inverse = torch.zeros(num_rows, dtype=torch.int32, device=q_index.device)
    coverage = torch.zeros(num_rows, dtype=torch.int8, device=q_index.device)
    rows = torch.arange(q_index.numel(), dtype=torch.int32, device=q_index.device)
    inverse.index_copy_(0, q_index, rows)
    coverage.index_fill_(0, q_index, 1)
    return inverse, coverage


#: Inverse permutations keyed by the plan's forward index tensors. A plan is
#: request-local and its tensors outlive every layer of a forward, so caching
#: here rebuilds the inverses once per request rather than once per layer.
#:
#: Each entry keeps a strong reference to the ``q_index`` it was built from.
#: That is what makes the address safe as a key: while the entry lives the
#: storage cannot be freed, so no later tensor can land on the same address and
#: collide with a stale inverse.
_INVERSE_CACHE: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
#: Entries are a few index vectors each. The bound is well above the handful of
#: passes one request plans and only guards against unbounded growth if a caller
#: never resets between requests.
_INVERSE_CACHE_LIMIT = 64


def inverse_permutation_for(q_index: torch.Tensor, num_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = (q_index.data_ptr(), num_rows)
    cached = _INVERSE_CACHE.get(key)
    if cached is None:
        if len(_INVERSE_CACHE) >= _INVERSE_CACHE_LIMIT:
            _INVERSE_CACHE.clear()
        inverse, coverage = _inverse_permutation(q_index, num_rows)
        cached = (q_index, inverse, coverage)
        _INVERSE_CACHE[key] = cached
    return cached[1], cached[2]


def reset_inverse_cache() -> None:
    """Release the cached inverses; call when the owning plans are dropped."""
    _INVERSE_CACHE.clear()


def fused_merge_supported(
    *,
    device: torch.device,
    heads: int,
    head_dim: int,
    num_passes: int,
    planned_tokens: int,
    pass_rows: list[int],
    dtype: torch.dtype,
) -> bool:
    """Whether the fused merge can run this geometry on this device."""
    if device.type != "cuda" or num_passes < 2:
        return False
    if head_dim > _MAX_MERGE_HEAD_DIM or head_dim & (head_dim - 1):
        return False
    if _load_triton_merge() is None:
        return False
    # The fused path holds every pass output at once and drops the FP32
    # accumulator; the sequential path holds the accumulator plus one pass.
    # Only take the fused path when the difference clearly fits.
    element = torch.finfo(dtype).bits // 8
    # One output row is heads*head_dim elements; its LSE is one float per head.
    pass_bytes = [rows * heads * (head_dim * element + 4) for rows in pass_rows]
    # Per pass, the inverse map is one int32 and the coverage flag one byte per
    # accumulator row -- shared across heads, so far below the pass outputs.
    fused = sum(pass_bytes) + num_passes * planned_tokens * 5
    sequential = planned_tokens * heads * (head_dim * 4 + 4) + max(pass_bytes)
    extra = fused - sequential
    if extra <= 0:
        return True
    free, _ = torch.cuda.mem_get_info(device)
    return free > extra + _MERGE_FREE_MEMORY_MARGIN


def fused_merge(
    result: torch.Tensor,
    outputs: list[torch.Tensor],
    lse_tensors: list[torch.Tensor],
    q_indices: list[torch.Tensor],
    *,
    planned_tokens: int,
    heads: int,
    head_dim: int,
) -> None:
    """Merge every pass into ``result`` with one kernel launch.

    ``result`` is ``[tokens, heads, head_dim]``; ``outputs[i]`` holds the rows
    ``q_indices[i]`` of pass ``i`` in that pass's own row order.
    """
    kernel = _load_triton_merge()
    if kernel is None:
        raise RuntimeError("Cosmos3 maskless fused merge requires Triton.")
    inverses, coverages = [], []
    for q_index in q_indices:
        inverse, coverage = inverse_permutation_for(q_index, planned_tokens)
        inverses.append(inverse)
        coverages.append(coverage)
    head_block = _heads_per_program(heads, head_dim)
    kernel[(planned_tokens * (heads // head_block),)](
        result,
        tuple(outputs),
        tuple(lse_tensors),
        tuple(inverses),
        tuple(coverages),
        planned_tokens,
        NUM_PASSES=len(outputs),
        HEADS=heads,
        HEAD_DIM=head_dim,
        HEAD_BLOCK=head_block,
        num_warps=_MERGE_NUM_WARPS,
    )
