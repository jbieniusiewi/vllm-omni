# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Maskless Cosmos3 multiview attention: unmasked varlen passes merged by log-sum-exp.

Planning is host-side (:mod:`.multiview_maskless_plan`).  This module owns the
pass kernels, the FP32 merge and the custom op that keeps both opaque to
Dynamo so the GEN layers compile around a single call.

Pass-kernel contract (``PASS_KERNELS``): packed ``[total, H, D]`` queries and
``[total, H_kv, D]`` keys/values with ``H_kv | H``, int32 cumulative sequence
offsets, no causal masking, no dropout, no zero-length segment; returns the
attention output in the query dtype and the per-row log-sum-exp as natural-log
FP32 ``[total_q, H]``.  Only vLLM's bundled FlashAttention is registered for
production; a pure-torch kernel serves CPU tests.
"""

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache
from typing import Any

import torch
import torch.nn.functional as F

from .multiview_flex_attention import MultiviewAttentionContext
from .multiview_maskless_plan import (
    META_GEN_TOKENS,
    META_IDENTITY_K,
    META_IDENTITY_Q,
    META_KEYS_FROM_UND,
    META_MAX_SEQLEN_K,
    META_MAX_SEQLEN_Q,
    TENSORS_PER_PASS,
    MultiviewMasklessPlan,
    build_multiview_maskless_plan,
)

logger = logging.getLogger(__name__)

#: Rows merged per step: bounds the FP32 temporaries to a few hundred MiB.
MERGE_CHUNK_SIZE = 8192
_INT32_LIMIT = 2**31

PassKernel = Callable[..., tuple[torch.Tensor, torch.Tensor]]
PASS_KERNELS: dict[str, Callable[[], PassKernel]] = {}


def register_pass_kernel(name: str, factory: Callable[[], PassKernel]) -> None:
    PASS_KERNELS[name] = factory


@cache
def get_pass_kernel(name: str) -> PassKernel:
    factory = PASS_KERNELS.get(name)
    if factory is None:
        raise ValueError(f"Unknown Cosmos3 maskless pass kernel {name!r}; expected one of {sorted(PASS_KERNELS)}.")
    return factory()


def validate_indexing(lengths: list[int], heads: int, head_dim: int) -> None:
    """Check before allocating indices or converting cumulative offsets to int32."""
    if heads <= 0 or head_dim <= 0 or any(length < 0 for length in lengths):
        raise ValueError("Maskless attention requires nonnegative lengths and positive head geometry.")
    if sum(lengths) >= _INT32_LIMIT or any(length * heads * head_dim >= _INT32_LIMIT for length in lengths):
        raise ValueError(
            "Maskless attention exceeds int32 indexing (length × local heads × head_dim or cumulative offsets). "
            "Reduce frames/resolution/views or increase supported TP/Ulysses head partitioning."
        )


def normalize_varlen_lse(lse: torch.Tensor, tokens: int, query_heads: int) -> torch.Tensor:
    """vLLM FA2/FA3/FA4 varlen return LSE as ``[heads, total_q]``; merge wants ``[total_q, heads]`` FP32."""
    if lse.ndim != 2 or lse.shape != (query_heads, tokens):
        raise ValueError(f"Expected FlashAttention LSE [heads,tokens]={query_heads, tokens}, got {tuple(lse.shape)}.")
    return lse.transpose(0, 1).float().contiguous()


def _make_vllm_flash_attn_kernel(fa_version: int) -> PassKernel:
    def kernel(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm_omni.diffusion.attention.backends.utils.fa import vllm_flash_attn_varlen_with_lse

        out, lse = vllm_flash_attn_varlen_with_lse(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=False,
            fa_version=fa_version,
        )
        return out, normalize_varlen_lse(lse, q.shape[0], q.shape[1])

    return kernel


def torch_reference_pass_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pass-kernel contract as per-segment dense softmax, for any device and dtype."""
    compute_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
    ratio = q.shape[1] // k.shape[1]
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[0], q.shape[1], dtype=torch.float32, device=q.device)
    bounds = zip(
        cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist(), cu_seqlens_k[:-1].tolist(), cu_seqlens_k[1:].tolist()
    )
    for qa, qb, ka, kb in bounds:
        keys = k[ka:kb].to(compute_dtype).repeat_interleave(ratio, 1)
        values = v[ka:kb].to(compute_dtype).repeat_interleave(ratio, 1)
        scores = torch.einsum("qhd,khd->hqk", q[qa:qb].to(compute_dtype), keys) * scale
        out[qa:qb] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), values).to(q.dtype)
        lse[qa:qb] = scores.logsumexp(-1).transpose(0, 1).float()
    return out, lse


for _version in (2, 3, 4):
    register_pass_kernel(f"fa{_version}", lambda version=_version: _make_vllm_flash_attn_kernel(version))
register_pass_kernel("torch", lambda: torch_reference_pass_kernel)


@cache
def resolve_maskless_kernel() -> str:
    """Pin the worker's vLLM-bundled FlashAttention once and log the choice."""
    from vllm_omni.diffusion.attention.backends.utils.fa import resolve_vllm_flash_attn_version

    fa_version = resolve_vllm_flash_attn_version()
    logger.info(
        "Cosmos3 maskless: GPU=%s FA=%s merge=streaming-fp32 PyTorch=%s CUDA=%s",
        torch.cuda.get_device_name() if torch.cuda.is_available() else "none",
        fa_version,
        torch.__version__,
        torch.version.cuda,
    )
    return f"fa{fa_version}"


def merge_step(
    acc: torch.Tensor, acc_lse: torch.Tensor, out: torch.Tensor, lse: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold one branch into the FP32 accumulator.

    This is the sequential sigmoid/logsigmoid recurrence of NATTEN 0.21.6's
    ``merge_attentions``: ``acc`` becomes the softmax-weighted average over
    branches seen so far and ``acc_lse`` their ``logaddexp``.  LSE is natural
    log; both accumulators stay FP32 until the final cast.
    """
    weight = torch.sigmoid(lse.float() - acc_lse).unsqueeze(-1)
    acc = acc - weight * (acc - out.float())
    acc_lse = acc_lse - F.logsigmoid(acc_lse - lse.float())
    return acc, acc_lse


def _merge_attention_outputs(outputs: list[torch.Tensor], lse_tensors: list[torch.Tensor]) -> torch.Tensor:
    """All-branches reference merge with ``[..., tokens, heads]`` FP32 LSE (tests and oracles)."""
    if not outputs or len(outputs) != len(lse_tensors):
        raise ValueError("Maskless merge requires matching nonempty output and LSE lists.")
    acc, acc_lse = outputs[0].float(), lse_tensors[0].float()
    for out, lse in zip(outputs[1:], lse_tensors[1:], strict=True):
        acc, acc_lse = merge_step(acc, acc_lse, out, lse)
    return acc.to(outputs[0].dtype)


def merge_into(acc: torch.Tensor, acc_lse: torch.Tensor, out: torch.Tensor, lse: torch.Tensor) -> None:
    """``merge_step`` folded into the accumulators in place.

    Same arithmetic in the same order as :func:`merge_step`, so the result is
    bit-identical.  Writing through the accumulators lets one fused elementwise
    kernel do the whole fold: the functional form has to materialize its two
    results and copy them back, which reads and writes the FP32 accumulator
    twice over.  Both deltas are read off the incoming ``acc_lse`` before
    either store, so the two in-place updates cannot observe each other.
    """
    lse_f = lse.float()
    weight = torch.sigmoid(lse_f - acc_lse).unsqueeze(-1)
    delta_lse = F.logsigmoid(acc_lse - lse_f)
    acc.sub_(weight * (acc - out.float()))
    acc_lse.sub_(delta_lse)


def merge_into_result(
    result: torch.Tensor, acc: torch.Tensor, acc_lse: torch.Tensor, out: torch.Tensor, lse: torch.Tensor
) -> None:
    """Fold the final branch straight into the output, in the caller's dtype.

    The last pass is the only one whose updated LSE nobody reads, and whose
    accumulator is immediately cast to the output.  Writing the cast value
    directly retires both the accumulator store and the separate output copy;
    the rounding is the same single FP32->output conversion either way.
    """
    weight = torch.sigmoid(lse.float() - acc_lse).unsqueeze(-1)
    result.copy_(acc - weight * (acc - out.float()))


def merge_into_gathered(
    acc: torch.Tensor,
    acc_lse: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    inverse: torch.Tensor,
) -> None:
    """``merge_into`` for a pass that permutes the rows it covers.

    ``inverse`` is the inverse of the pass's ``q_index``, so row ``g`` of the
    accumulator folds in row ``inverse[g]`` of the pass -- the same pairing
    ``q_index`` expresses, read from the other side.  Gathering the pass output
    instead of the accumulator keeps the accumulator traffic sequential and
    leaves only one indexed read, over the smaller branch tensors.
    """
    out_rows = out.index_select(0, inverse)
    lse_rows = lse.index_select(0, inverse)
    weight = torch.sigmoid(lse_rows - acc_lse).unsqueeze(-1)
    delta_lse = F.logsigmoid(acc_lse - lse_rows)
    acc.sub_(weight * (acc - out_rows.float()))
    acc_lse.sub_(delta_lse)


def merge_into_result_gathered(
    result: torch.Tensor,
    acc: torch.Tensor,
    acc_lse: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    inverse: torch.Tensor,
) -> None:
    """``merge_into_result`` for a permuting final pass; see :func:`merge_into_gathered`."""
    out_rows = out.index_select(0, inverse)
    lse_rows = lse.index_select(0, inverse)
    weight = torch.sigmoid(lse_rows - acc_lse).unsqueeze(-1)
    result.copy_(acc - weight * (acc - out_rows.float()))


#: In-place merge kernels, compiled once per device type on first use.
_MERGE_KERNELS: tuple[Callable[..., None], ...] = (
    merge_into,
    merge_into_result,
    merge_into_gathered,
    merge_into_result_gathered,
)
_compiled_merge_kernels: tuple[Callable[..., None], ...] | None = None


def _merge_kernels_for(device: torch.device) -> tuple[Callable[..., None], ...]:
    # Elementwise over the accumulator rows: one dynamic-shape graph per kernel
    # serves every geometry.  CPU (tests) runs eager.
    if device.type != "cuda":
        return _MERGE_KERNELS
    global _compiled_merge_kernels
    if _compiled_merge_kernels is None:
        _compiled_merge_kernels = tuple(
            torch.compile(kernel, fullgraph=True, dynamic=True) for kernel in _MERGE_KERNELS
        )
    return _compiled_merge_kernels


def _inverse_permutation(index: torch.Tensor) -> torch.Tensor:
    """Inverse of a full-coverage ``q_index``, cached on the plan tensor.

    A pass whose row count equals the GEN length touches every row exactly
    once -- the planner rejects repeats -- so ``q_index`` is a permutation and
    has an inverse.  It is part of the request-local plan, so derive it once
    and keep it alongside; the plan tensors outlive every denoise step.
    """
    inverse = getattr(index, "_cosmos3_inverse_permutation", None)
    if inverse is None:
        inverse = torch.empty_like(index)
        inverse.scatter_(0, index, torch.arange(index.numel(), device=index.device, dtype=index.dtype))
        try:
            index._cosmos3_inverse_permutation = inverse
        except (AttributeError, RuntimeError):
            # Cannot cache on this tensor (e.g. a traced subclass); recompute.
            pass
    return inverse


def _maskless_attention_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
    plan: list[torch.Tensor],
    num_passes: int,
    kernel_name: str,
) -> torch.Tensor:
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("Maskless multiview attention requires B == 1.")
    if num_passes <= 0 or len(plan) != num_passes * TENSORS_PER_PASS:
        raise ValueError(f"Maskless plan carries {len(plan)} tensors for {num_passes} passes.")
    first_meta = plan[TENSORS_PER_PASS - 1]
    planned_tokens = int(first_meta[META_GEN_TOKENS])
    if q.shape[1] != planned_tokens or plan[0].numel() != planned_tokens:
        raise ValueError(
            "Cosmos3 maskless packed GEN length does not match the request plan: "
            f"attention={q.shape[1]}, plan={planned_tokens}."
        )
    if k.shape != v.shape or k_und.shape != v_und.shape or k.shape[:2] != q.shape[:2]:
        raise ValueError("Maskless Q/K/V geometry mismatch.")
    if k_und.shape[0] != 1 or k_und.shape[2:] != k.shape[2:] or q.shape[-1] != k.shape[-1]:
        raise ValueError("Maskless GEN/UND head geometry mismatch.")
    if q.shape[2] % k.shape[2]:
        raise ValueError("Maskless attention requires an integral GQA ratio.")
    heads, head_dim = q.shape[2], q.shape[3]
    kernel = get_pass_kernel(kernel_name)
    merge_in, merge_final, merge_in_gathered, merge_final_gathered = _merge_kernels_for(q.device)
    # Preserve the caller's tensor mode: HSDP/offload use no_grad() and need
    # outputs with version counters.  Only the passes and the accumulators are
    # inference-mode temporaries.
    result = torch.empty_like(q)
    with torch.inference_mode():
        acc: torch.Tensor | None = None
        acc_lse: torch.Tensor | None = None
        folded = False
        for index in range(num_passes):
            q_index, k_index, cu_seqlens_q, cu_seqlens_k, meta = plan[
                index * TENSORS_PER_PASS : (index + 1) * TENSORS_PER_PASS
            ]
            meta_values = meta.tolist()
            max_seqlen_q, max_seqlen_k = int(meta_values[META_MAX_SEQLEN_Q]), int(meta_values[META_MAX_SEQLEN_K])
            keys_from_und = bool(meta_values[META_KEYS_FROM_UND])
            identity_q, identity_k = bool(meta_values[META_IDENTITY_Q]), bool(meta_values[META_IDENTITY_K])
            # Actual heads here are already partitioned by TP and Ulysses.
            validate_indexing([max_seqlen_q], heads, head_dim)
            validate_indexing([max_seqlen_k], k.shape[2], head_dim)
            keys, values = (k_und[0], v_und[0]) if keys_from_und else (k[0], v[0])
            queries = q[0] if identity_q else q[0].index_select(0, q_index)
            if not identity_k:
                keys, values = keys.index_select(0, k_index), values.index_select(0, k_index)
            out, lse = kernel(
                queries,
                keys,
                values,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
            )
            del queries, keys, values
            rows = q_index.numel()
            if out.shape != (rows, heads, head_dim) or lse.shape != (rows, heads):
                raise ValueError(
                    f"Maskless pass kernel returned output {tuple(out.shape)} and LSE {tuple(lse.shape)} "
                    f"for {rows} rows × {heads} heads × {head_dim}."
                )
            # The last pass can fold straight into the result: nothing reads the
            # accumulators afterwards, so its stores and the output copy merge.
            # Only a pass covering every row may do so -- a partial one would
            # leave the rows it does not touch unwritten.
            is_last_pass = index == num_passes - 1 and rows == planned_tokens
            if acc is None:
                # The same-view pass covers every row and seeds the accumulators.
                acc = torch.empty((planned_tokens, heads, head_dim), dtype=torch.float32, device=q.device)
                acc_lse = torch.empty((planned_tokens, heads), dtype=torch.float32, device=q.device)
                for start in range(0, rows, MERGE_CHUNK_SIZE):
                    chunk = slice(start, start + MERGE_CHUNK_SIZE)
                    if identity_q:
                        acc[chunk].copy_(out[chunk])
                        acc_lse[chunk].copy_(lse[chunk])
                    else:
                        acc.index_copy_(0, q_index[chunk], out[chunk].float())
                        acc_lse.index_copy_(0, q_index[chunk], lse[chunk].float())
            elif identity_q:
                folded = is_last_pass
                for start in range(0, rows, MERGE_CHUNK_SIZE):
                    chunk = slice(start, start + MERGE_CHUNK_SIZE)
                    if folded:
                        merge_final(result[0, chunk], acc[chunk], acc_lse[chunk], out[chunk], lse[chunk])
                    else:
                        merge_in(acc[chunk], acc_lse[chunk], out[chunk], lse[chunk])
            else:
                # A pass covering every row permutes them, so walk the
                # accumulator in order and gather the branch rows instead.
                # CPU keeps the pass-order traversal: its vectorized sigmoid is
                # not position-invariant, so reordering the rows would shift the
                # last ULP away from the reference the tests compare against.
                reorders = rows == planned_tokens and q.device.type == "cuda"
                inverse = _inverse_permutation(q_index) if reorders else None
                if inverse is None:
                    for start in range(0, rows, MERGE_CHUNK_SIZE):
                        chunk = slice(start, start + MERGE_CHUNK_SIZE)
                        gather = q_index[chunk]
                        rows_acc = acc.index_select(0, gather)
                        rows_lse = acc_lse.index_select(0, gather)
                        merge_in(rows_acc, rows_lse, out[chunk], lse[chunk])
                        acc.index_copy_(0, gather, rows_acc)
                        acc_lse.index_copy_(0, gather, rows_lse)
                else:
                    folded = is_last_pass
                    for start in range(0, planned_tokens, MERGE_CHUNK_SIZE):
                        chunk = slice(start, start + MERGE_CHUNK_SIZE)
                        if folded:
                            merge_final_gathered(
                                result[0, chunk], acc[chunk], acc_lse[chunk], out, lse, inverse[chunk]
                            )
                        else:
                            merge_in_gathered(acc[chunk], acc_lse[chunk], out, lse, inverse[chunk])
            del out, lse
        assert acc is not None
        if not folded:
            # No final fold happened (a single pass, a last pass covering only
            # some rows, or the CPU scatter path), so cast the accumulator.
            for start in range(0, planned_tokens, MERGE_CHUNK_SIZE):
                chunk = slice(start, start + MERGE_CHUNK_SIZE)
                result[0, chunk].copy_(acc[chunk])
    return result


# Dynamo sees one opaque call per layer; the guard preserves registration
# across module re-imports in tests.
if not hasattr(torch.ops.vllm_omni, "cosmos3_maskless_attention"):

    @torch.library.custom_op("vllm_omni::cosmos3_maskless_attention", mutates_args=())
    def _cosmos3_maskless_attention_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        plan: list[torch.Tensor],
        num_passes: int,
        kernel: str,
    ) -> torch.Tensor:
        return _maskless_attention_impl(q, k, v, k_und, v_und, plan, num_passes, kernel)

    @_cosmos3_maskless_attention_op.register_fake
    def _(q, k, v, k_und, v_und, plan, num_passes, kernel):
        return torch.empty_like(q)


_cosmos3_maskless_attention_op = torch.ops.vllm_omni.cosmos3_maskless_attention


@dataclass(frozen=True, eq=False)
class MasklessRuntime:
    """Flattened plan plus the worker's pinned kernel, carried on the attention context."""

    plan: list[torch.Tensor]
    num_passes: int
    kernel: str


def get_maskless_plan(
    context: MultiviewAttentionContext, *, num_und_tokens: int, device: torch.device | str
) -> MultiviewMasklessPlan:
    """Build or retrieve the request-local plan for one CFG text length."""
    device = torch.device(device)
    key = ("maskless", context.layout, num_und_tokens, device)
    plan = context.mask_cache.get(key)
    if plan is None:
        plan = build_multiview_maskless_plan(context.layout, num_und_tokens, device)
        context.mask_cache[key] = plan
    return plan


def prepare_maskless_context(
    context: MultiviewAttentionContext,
    *,
    num_und_tokens: int,
    device: torch.device | str,
    kernel: str | None = None,
) -> MultiviewAttentionContext:
    """Attach the plan and kernel before the compiled GEN layers run."""
    plan = get_maskless_plan(context, num_und_tokens=num_und_tokens, device=device)
    runtime = MasklessRuntime(plan.flatten(), len(plan.passes), kernel or resolve_maskless_kernel())
    return replace(context, maskless=runtime)


def maskless_multiview_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_und: torch.Tensor,
    v_und: torch.Tensor,
    context: MultiviewAttentionContext,
) -> torch.Tensor:
    runtime: Any = context.maskless
    if runtime is None:
        raise RuntimeError(
            "Cosmos3 maskless attention needs a prepared plan; call prepare_maskless_context before the GEN layers."
        )
    return _cosmos3_maskless_attention_op(q, k, v, k_und, v_und, runtime.plan, runtime.num_passes, runtime.kernel)
