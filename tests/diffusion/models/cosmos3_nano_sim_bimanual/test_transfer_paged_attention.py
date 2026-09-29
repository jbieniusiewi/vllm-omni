# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-owned text and metadata reuse must preserve paged attention semantics."""

import gc
import weakref

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.paged_attention_transfer import (
    CosmosSimTransferPagedAttention,
    paged_write_attn,
)
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
BLOCK, HEADS, DIM = 16, 2, 64


@pytest.fixture
def setup():
    cache = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=BLOCK, window_chunks=8),
        num_layers=1,
        num_kv_heads=HEADS,
        head_size=DIM,
        dtype=torch.float32,
        block_size=BLOCK,
        max_model_len=4096,
        available_bytes=1 << 22,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=1,
        frames_per_block=4,
        max_scratch_frames_per_branch=4,
        max_scratch_tokens_per_branch=21,
        device=torch.device("cpu"),
    )
    state = ARDiffusionKVState(cache, "s1", {"main": cache.begin_request("s1")}, num_layers=1)
    owner = CosmosSimTransferPagedAttention(cache)
    yield cache, state, owner
    state.close()


def prepare(state, owner, text, *, frames=1, commit=False, causal=False, session="s1"):
    contexts = state.get_kv_caches(
        "main",
        seq_len=frames * BLOCK,
        commit_current=commit,
        extra_visible_tokens=BLOCK if causal else frames * BLOCK,
        frame_causal=causal,
    )
    layer = owner.wrap(contexts, session_id=session, text_kv=text, text_length=21)[0]
    layer.forward_ctx.prepare(device=torch.device("cpu"), action_len=21, query_len=frames * BLOCK)
    return layer.to_layer_inputs()


@pytest.mark.parametrize("frames", [1, 2, 3, 4])
@pytest.mark.parametrize("commit", [False, True])
def test_joint_attention_matches_dense_and_preserves_commit(setup, frames, commit):
    cache, state, owner = setup
    text = [(torch.randn(1, 21, HEADS, DIM), torch.randn(1, 21, HEADS, DIM))]
    inputs = prepare(state, owner, text, frames=frames, commit=commit)
    q, k, v = (torch.randn(frames * BLOCK, HEADS, DIM) for _ in range(3))
    actual = paged_write_attn(inputs, q, k, v, None, None, DIM**-0.5)
    key, value = torch.cat([k, text[0][0][0]]), torch.cat([v, text[0][1][0]])
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
    ).transpose(0, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    state.commit_paged_context("main")
    assert state.adapter("main").completed_chunks == (frames if commit else 0)
    assert inputs.attention.action_slots.numel() == 0


def test_text_staged_once_and_replaced_without_retaining_old_prompt(setup, monkeypatch):
    cache, state, owner = setup
    text = [(torch.randn(1, 21, HEADS, DIM), torch.randn(1, 21, HEADS, DIM))]
    calls = []
    original = torch.Tensor.copy_

    def copy(dst, src, *args, **kwargs):
        if src.shape == (21, HEADS, DIM):
            calls.append(src.data_ptr())
        return original(dst, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    first = prepare(state, owner, text)
    state.commit_paged_context("main")
    again = prepare(state, owner, text, frames=4)
    state.commit_paged_context("main")
    third = prepare(state, owner, text)
    state.commit_paged_context("main")
    assert len(calls) == 2
    assert first.attention.block_table is third.attention.block_table
    assert first.attention.video_slots is third.attention.video_slots
    assert again.attention.video_slots.numel() == 4 * BLOCK
    old = weakref.ref(text[0][0])
    text = [(torch.full((1, 21, HEADS, DIM), 7.0), torch.full((1, 21, HEADS, DIM), 9.0))]
    gc.collect()
    assert old() is None
    prepare(state, owner, text)
    state.commit_paged_context("main")
    assert len(calls) == 4
    blocks = cache.scratch_block_ids("main", 4, 2)
    torch.testing.assert_close(
        cache.key_cache(0).flatten(0, 1)[blocks[0] * BLOCK : blocks[0] * BLOCK + 21], text[0][0][0]
    )
    prepare(state, owner, text, session="s2")
    state.commit_paged_context("main")
    prepare(state, owner, text)
    state.commit_paged_context("main")
    assert len(calls) == 8


@pytest.mark.parametrize("causal", [False, True])
def test_text_pages_are_protected_from_oversized_latent_scratch(setup, causal):
    _, state, owner = setup
    text = [(torch.zeros(1, 21, HEADS, DIM), torch.zeros(1, 21, HEADS, DIM))]
    with pytest.raises(ValueError, match="scratch frame capacity"):
        prepare(state, owner, text, frames=5, commit=causal, causal=causal)


def test_metadata_views_keep_addresses_with_bounded_storage(setup):
    _, _, owner = setup
    views = {}
    for size in (5, 31, 9, 5, 31):
        host = torch.arange(size, dtype=torch.int32) + size
        view = owner._copy_metadata("main", "test", host, torch.device("cpu"))
        if size in views:
            assert view is views[size]
        views[size] = view
        torch.testing.assert_close(view, host)
    storages = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in views.values()}
    assert sum(storages.values()) < 2 * 32 * 4


def test_sim_operator_mutation_and_fake_contract(setup):
    _, state, owner = setup
    text = [(torch.randn(1, 21, HEADS, DIM), torch.randn(1, 21, HEADS, DIM))]
    i = prepare(state, owner, text).attention
    q, k, v = (torch.randn(BLOCK, HEADS, DIM) for _ in range(3))
    result = torch.library.opcheck(
        torch.ops.vllm_omni.cosmos_sim_transfer_paged_write_attn.default,
        (
            q,
            k,
            v,
            None,
            None,
            i.key_pool,
            i.value_pool,
            i.block_size,
            i.video_slots,
            i.action_slots,
            i.block_table,
            i.query_start_loc,
            i.seq_lens,
            i.max_query_len,
            i.max_seq_len,
            DIM**-0.5,
            None,
            None,
            False,
            0,
            False,
        ),
    )
    assert all(value == "SUCCESS" for value in result.values())


def test_only_transfer_activates_model_specific_paged_adapter(setup):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.pipeline_cosmos3_nano_sim_bimanual import (
        Cosmos3NanoSimBimanualPipeline,
    )
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.pipeline_cosmos3_nano_sim_transfer import (
        Cosmos3NanoSimTransferPipeline,
    )

    _, state, _ = setup
    pipeline = SimpleNamespace(_ar_diffusion_kv_state=state)
    text = [(torch.zeros(1, 21, HEADS, DIM), torch.zeros(1, 21, HEADS, DIM))]
    contexts = state.get_kv_caches("main", seq_len=BLOCK, commit_current=False)
    assert not hasattr(Cosmos3NanoSimBimanualPipeline, "_prepare_paged_attention")
    assert not hasattr(pipeline, "_transfer_paged_attention")
    adapted = Cosmos3NanoSimTransferPipeline._prepare_paged_attention(
        pipeline, contexts, text_kv=text, real_text_kv_len=21
    )
    assert adapted[0].forward_ctx.base is contexts[0].forward_ctx
    assert pipeline._transfer_paged_attention.is_bound_to(state.kv_cache)


@torch.no_grad()
def test_transfer_attention_specialization_matches_inherited_dense_math(setup):
    from torch import nn

    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_bimanual import (
        Cosmos3NanoSimBimanualGenDecoderLayer,
        Cosmos3NanoSimBimanualJointAttention,
    )
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_transfer import (
        Cosmos3NanoSimTransferGenDecoderLayer,
        Cosmos3NanoSimTransferJointAttention,
    )

    assert Cosmos3NanoSimTransferGenDecoderLayer.forward is Cosmos3NanoSimBimanualGenDecoderLayer.forward

    class DenseAttention(nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), enable_gqa=True
            ).transpose(1, 2)

    attention = Cosmos3NanoSimTransferJointAttention.__new__(Cosmos3NanoSimTransferJointAttention)
    nn.Module.__init__(attention)
    attention.num_heads_local, attention.num_kv_heads_local, attention.head_dim = 4, HEADS, DIM
    attention.qk_norm = True
    attention.to_q, attention.to_k, attention.to_v = [nn.Linear(DIM, n * DIM, bias=False) for n in (4, HEADS, HEADS)]
    attention.to_out = nn.Linear(4 * DIM, DIM, bias=False)
    attention.norm_q, attention.norm_k = nn.RMSNorm(DIM), nn.RMSNorm(DIM)
    attention.norm_q.variance_epsilon = attention.norm_k.variance_epsilon = 1e-6
    attention.attn = DenseAttention()
    _, state, owner = setup
    text = [(torch.randn(1, 21, HEADS, DIM), torch.randn(1, 21, HEADS, DIM))]
    hidden = torch.randn(1, BLOCK, DIM)
    args = dict(
        text_k=text[0][0],
        text_v=text[0][1],
        real_text_kv_len=21,
        freqs_cos=torch.ones(1, BLOCK, 1, DIM),
        freqs_sin=torch.zeros(1, BLOCK, 1, DIM),
        num_frames=1,
        tokens_per_frame=BLOCK,
        action_tokens_per_frame=0,
    )
    expected = Cosmos3NanoSimBimanualJointAttention.forward(attention, hidden, **args)
    dense = attention(hidden, **args)
    for a, b in zip(dense, expected):
        assert torch.equal(a, b)
    paged = attention(hidden, paged_context=prepare(state, owner, text), **args)
    for a, b in zip(paged, expected):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
