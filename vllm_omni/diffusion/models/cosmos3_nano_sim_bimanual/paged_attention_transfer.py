# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transfer-only paged joint attention for sequential control/latent forwards.

The runner still owns pages and session lifetime. This adapter stages the Transfer
model's immutable text KV once and keeps model-owned metadata addresses stable
for regional CUDA graphs. It does not change generic AR-Diffusion dispatch or
allocation. A cache must be used exclusively by this adapter while it is bound;
forwards submit reads and subsequent updates on the same CUDA stream.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ARDiffusionPagedForwardContext,
    ARDiffusionPagedLayerInputs,
    _paged_write_attn_impl,
)


class CosmosSimTransferPagedLayerInputs(NamedTuple):
    """Mark layer inputs whose immutable text is already in paged scratch."""

    attention: ARDiffusionPagedLayerInputs


class CosmosSimTransferPagedAttention:
    """Model-owned staging state, shared across requests using the same pool."""

    def __init__(self, cache: Any):
        self._cache = weakref.ref(cache)
        self._text: dict[str, tuple] = {}
        self._storage: dict[tuple, torch.Tensor] = {}
        self._views: dict[tuple, torch.Tensor] = {}

    def is_bound_to(self, cache: Any) -> bool:
        return self._cache() is cache

    def wrap(self, contexts, *, session_id: str, text_kv, text_length: int):
        base = contexts[0].forward_ctx
        if not self.is_bound_to(base.kv_cache):
            raise ValueError("Sim-Transfer paged adapter belongs to a different KV pool")
        forward = _TransferForward(self, base, session_id, text_kv, text_length)
        return [_TransferLayer(forward, layer.layer_idx) for layer in contexts]

    def _activate_text(self, base, session_id, text_kv, length):
        cache = base.kv_cache
        if not 0 < length <= cache.max_scratch_tokens_per_branch:
            raise ValueError("Sim-Transfer text length exceeds the declared auxiliary scratch capacity")
        if len(text_kv) != cache.num_layers:
            raise ValueError("Sim-Transfer text KV layer count differs from the paged pool")
        blocks = cache.scratch_block_ids(
            base.kv_branch, cache.max_scratch_frames_per_branch, (length + cache.block_size - 1) // cache.block_size
        )
        old = self._text.get(base.kv_branch)
        tensors = tuple(t for pair in text_kv for t in pair)
        if old is not None and old[:2] == (session_id, length):
            if all(ref() is tensor for ref, tensor in zip(old[2], tensors)):
                return blocks
        # Do not preserve a successful activation if a replacement copy fails.
        self._text.pop(base.kv_branch, None)
        start = blocks[0] * cache.block_size
        for layer, (key, value) in enumerate(text_kv):
            expected_tail = (cache.num_kv_heads, cache.head_size)
            if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
                raise ValueError("Sim-Transfer text KV must have matching [1, tokens, heads, dim] shapes")
            if key.shape[1] < length or tuple(key.shape[2:]) != expected_tail:
                raise ValueError("Sim-Transfer text KV shape does not fit the paged pool")
            cache.key_cache(layer).flatten(0, 1)[start : start + length].copy_(key[0, :length])
            cache.value_cache(layer).flatten(0, 1)[start : start + length].copy_(value[0, :length])
        # Weak references prevent this adapter from retaining an obsolete prompt
        # allocation after reset/close. Sim-Transfer text tensors are immutable per request.
        self._text[base.kv_branch] = (session_id, length, tuple(weakref.ref(t) for t in tensors))
        return blocks

    def _copy_metadata(self, branch, name, host, device):
        base_key = (branch, name, host.dtype, torch.device(device))
        key = (*base_key, tuple(host.shape))
        view = self._views.get(key)
        if view is None:
            size = host.numel()
            storage = self._storage.get(base_key)
            if storage is None or storage.numel() < size:
                capacity = 1 << max(0, (max(1, size) - 1).bit_length())
                storage = torch.empty(capacity, dtype=host.dtype, device=device)
                self._storage[base_key] = storage
            view = storage[:size].view(host.shape)
            torch._dynamo.mark_static_address(view)
            self._views[key] = view
        if host.numel():
            if view.is_cuda:
                host = host.pin_memory()
            view.copy_(host, non_blocking=True)
        return view


@dataclass
class _TransferForward:
    owner: CosmosSimTransferPagedAttention
    base: ARDiffusionPagedForwardContext
    session_id: str
    text_kv: list[tuple[torch.Tensor, torch.Tensor]]
    text_length: int
    _prepared: bool = False

    @property
    def seq_len(self):
        return self.base.seq_len

    def prepare(self, *, device, action_len, query_len):
        if self._prepared:
            return
        base = self.base
        if action_len != self.text_length:
            raise ValueError("Sim-Transfer text length differs from the activated prompt")
        if not base.commit_current or base.frame_causal:
            if base.num_current_video_blocks > base.kv_cache.max_scratch_frames_per_branch:
                raise ValueError("Sim-Transfer current latent span exceeds the declared scratch frame capacity")
        base.action_scratch_block_ids = self.owner._activate_text(base, self.session_id, self.text_kv, action_len)
        # Reuse the generic builder's existing preallocated-auxiliary contract.
        # No per-layer text scatter is needed: its pages are already populated.
        base.action_slot_mapping = torch.empty(0, dtype=torch.long)
        base._action_len = action_len
        base.prepare(device=torch.device("cpu"), action_len=action_len, query_len=query_len)
        names = ("current_video_slot_mapping", "action_slot_mapping", "block_table", "query_start_loc", "seq_lens")
        try:
            converted = {
                name: self.owner._copy_metadata(base.kv_branch, name, getattr(base, name), device) for name in names
            }
        except Exception:
            base._prepared = False
            raise
        for name, tensor in converted.items():
            setattr(base, name, tensor)
        self._prepared = True


@dataclass
class _TransferLayer:
    forward_ctx: _TransferForward
    layer_idx: int

    def to_layer_inputs(self):
        return CosmosSimTransferPagedLayerInputs(self.forward_ctx.base.layer_inputs(self.layer_idx))


if not hasattr(torch.ops.vllm_omni, "cosmos_sim_transfer_paged_write_attn"):

    def _paged_write_attn_op(
        query: torch.Tensor,
        k_curr: torch.Tensor,
        v_curr: torch.Tensor,
        k_act: torch.Tensor | None,
        v_act: torch.Tensor | None,
        key_pool: torch.Tensor,
        value_pool: torch.Tensor,
        block_size: int,
        video_slots: torch.Tensor,
        action_slots: torch.Tensor,
        block_table: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        max_query_len: int,
        max_seq_len: int,
        softmax_scale: float,
        stage_key: torch.Tensor | None,
        stage_value: torch.Tensor | None,
        reuse_history: bool,
        stage_first_block: int,
        framewise_attention: bool,
    ) -> torch.Tensor:
        return _paged_write_attn_impl(
            query,
            k_curr,
            v_curr,
            k_act,
            v_act,
            key_pool,
            value_pool,
            block_size,
            video_slots,
            action_slots,
            block_table,
            query_start_loc,
            seq_lens,
            max_query_len,
            max_seq_len,
            softmax_scale,
            stage_key,
            stage_value,
            reuse_history,
            stage_first_block,
            framewise_attention,
        )

    # Register the schema/backend directly. custom_op's Python backend wrapper
    # repeatedly inspects the schema and checks output aliasing; at one call per
    # layer this dominates small AR forwards. The mutation contract and fake
    # implementation still make the operation opaque to fullgraph compilation.
    # The public define/impl helpers retain their libraries across module reloads.
    torch.library.define(
        "vllm_omni::cosmos_sim_transfer_paged_write_attn",
        torch.library.infer_schema(
            _paged_write_attn_op,
            mutates_args=("key_pool", "value_pool", "stage_key", "stage_value"),
        ),
    )
    torch.library.impl(
        "vllm_omni::cosmos_sim_transfer_paged_write_attn", "CompositeExplicitAutograd", _paged_write_attn_op
    )

    @torch.library.register_fake("vllm_omni::cosmos_sim_transfer_paged_write_attn")
    def _(
        query,
        k_curr,
        v_curr,
        k_act,
        v_act,
        key_pool,
        value_pool,
        block_size,
        video_slots,
        action_slots,
        block_table,
        query_start_loc,
        seq_lens,
        max_query_len,
        max_seq_len,
        softmax_scale,
        stage_key=None,
        stage_value=None,
        reuse_history=False,
        stage_first_block=0,
        framewise_attention=False,
    ):
        return torch.empty_like(query)


def paged_write_attn(
    inputs: ARDiffusionPagedLayerInputs | CosmosSimTransferPagedLayerInputs,
    query,
    k_curr,
    v_curr,
    k_act,
    v_act,
    softmax_scale: float,
    *,
    framewise_attention: bool = False,
) -> torch.Tensor:
    """Model-facing entry: routes through the custom op (traceable in fullgraph)."""
    if isinstance(inputs, CosmosSimTransferPagedLayerInputs):
        inputs = inputs.attention
        k_act = v_act = None
    return torch.ops.vllm_omni.cosmos_sim_transfer_paged_write_attn(
        query,
        k_curr,
        v_curr,
        k_act,
        v_act,
        inputs.key_pool,
        inputs.value_pool,
        inputs.block_size,
        inputs.video_slots,
        inputs.action_slots,
        inputs.block_table,
        inputs.query_start_loc,
        inputs.seq_lens,
        inputs.max_query_len,
        inputs.max_seq_len,
        softmax_scale,
        inputs.stage_key,
        inputs.stage_value,
        inputs.reuse_history,
        inputs.stage_first_block,
        framewise_attention,
    )
