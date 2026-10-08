# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare the single architecture to an independent dense capture-time reference."""

from __future__ import annotations

import math

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3.multiview_flex_attention import (
    MaskItem,
    MultiviewAttentionContext,
    MultiviewLayout,
    PaddedAttentionGeometry,
    _pack_padded_bshd,
    _semantic_groups,
    build_multiview_block_sparsity,
    build_multiview_flex_metadata,
    get_multiview_attention_plan,
    multiview_pair_predicate,
    padded_multiview_flex_attention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _layout(window: float = 0.4, backend: str = "triton") -> MultiviewLayout:
    return MultiviewLayout(
        items=(
            MaskItem((10, 1, 2), 2, is_control=True, seconds_per_frame=0.2),
            MaskItem((10, 1, 2), 2, seconds_per_frame=0.2),
            MaskItem((11, 1, 1), 1, view_offset=2, is_control=True, is_lidar=True, seconds_per_frame=0.1),
            MaskItem((11, 1, 1), 1, view_offset=2, is_lidar=True, seconds_per_frame=0.1),
        ),
        cross_view_past_window_seconds=window,
        caption_lengths=(2, 3),
        max_und_tokens=64,
        backend=backend,
    )


def _dense_reference(layout: MultiviewLayout, geometry: PaddedAttentionGeometry) -> torch.Tensor:
    # Construct records from sensor geometry, without consulting production token metadata.
    # A record is (real, control, text, sensor, capture_time); None marks padding.
    keys: list[tuple[bool, bool, bool, int, float] | None] = []
    for view, length in enumerate(layout.caption_lengths):
        keys.extend([(True, False, True, view, 0.0)] * length)
    keys.extend([None] * (geometry.padded_und_len - len(keys)))
    queries: list[tuple[bool, bool, bool, int, float] | None] = []
    for item in layout.items:
        for view in range(item.num_views):
            for frame in range(item.token_shape[0] // item.num_views):
                queries.extend(
                    [
                        (
                            True,
                            item.is_control,
                            False,
                            -2 if item.is_lidar else item.view_offset + view,
                            frame * item.seconds_per_frame,
                        )
                    ]
                    * (item.token_shape[1] * item.token_shape[2])
                )
    queries.extend([None] * (geometry.padded_q_len - len(queries)))
    keys.extend(queries)
    expected = torch.zeros(len(queries), len(keys), dtype=torch.bool)
    for qi, q in enumerate(queries):
        for ki, k in enumerate(keys):
            if q is None or k is None:
                allowed = q is None and k is None
            elif k[2]:
                allowed = q[3] == -2 or q[3] == k[3]
            elif q[3] == k[3]:
                allowed = True  # All same-sensor target and control edges span the clip.
            else:
                allowed = not q[1] and not k[1] and -1e-4 <= q[4] - k[4] <= layout.cross_view_past_window_seconds + 1e-4
            expected[qi, ki] = allowed
    return expected


@pytest.mark.cpu
@pytest.mark.parametrize("window", [0.0, 0.4, 0.40001])
def test_attention_pairs_match_independent_dense_reference(window: float) -> None:
    layout = _layout(window)
    geometry = PaddedAttentionGeometry(layout.gen_tokens, 64, 5, 64)
    metadata = build_multiview_flex_metadata(layout, geometry, "cpu")
    actual = multiview_pair_predicate(metadata, torch.arange(64)[:, None], torch.arange(128)[None, :])
    expected = _dense_reference(layout, geometry)
    torch.testing.assert_close(actual, expected)
    assert metadata.timestamp.dtype == torch.float32
    sparsity = build_multiview_block_sparsity(metadata)
    mask = sparsity.to_block_mask()
    torch.testing.assert_close(
        mask.mask_mod(torch.tensor(0), torch.tensor(0), torch.arange(64)[:, None], torch.arange(128)[None, :]), expected
    )
    # Both cross-view boundaries are inclusive; future keys are excluded, while
    # same-view future keys and same-view controls remain unrestricted.
    if window == 0.4:
        query = geometry.padded_und_len + 20 + 4  # camera 0 target, t=0.4
        other_target = geometry.padded_und_len + 20 + 10  # camera 1 target, t=0
        assert actual[query - 64, other_target]
        assert actual[query - 64, other_target + 4]  # t=0.4 upper boundary
        assert not actual[query - 64, other_target + 6]  # t=0.6 future cross-view
        assert actual[query - 64, query + 4]  # t=0.8 same-view future
        assert actual[query - 64, geometry.padded_und_len + 8]  # same-view future control
        assert not actual[: layout.gen_tokens, 5:64].any()  # text padding
        assert not actual[: layout.gen_tokens, 64 + layout.gen_tokens :].any()  # GEN padding


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("camera_rate", "lidar_rate", "lidar_frame", "visible"),
    [
        (0.20004, 0.1, 0, True),  # oldest key lies 8e-5 beyond the nominal past bound
        (0.20006, 0.1, 0, False),
        (0.2, 0.10002, 4, True),  # key lies 8e-5 beyond the nominal current-time bound
        (0.2, 0.10003, 4, False),
    ],
)
def test_capture_time_tolerance_at_both_boundaries(
    camera_rate: float, lidar_rate: float, lidar_frame: int, visible: bool
) -> None:
    layout = MultiviewLayout(
        items=(
            MaskItem((3, 1, 1), 1, seconds_per_frame=camera_rate),
            MaskItem((5, 1, 1), 1, view_offset=1, is_lidar=True, seconds_per_frame=lidar_rate),
        ),
        cross_view_past_window_seconds=0.4,
        caption_lengths=(2,),
        max_und_tokens=64,
    )
    metadata = build_multiview_flex_metadata(layout, PaddedAttentionGeometry(8, 64, 2, 64), "cpu")
    visible_pair = multiview_pair_predicate(metadata, torch.tensor(2), torch.tensor(64 + 3 + lidar_frame))
    assert bool(visible_pair) is visible


@pytest.mark.cpu
@pytest.mark.parametrize("window", [None, True, -0.1, float("inf"), float("nan")])
def test_layout_rejects_invalid_window(window: float) -> None:
    with pytest.raises(ValueError, match="finite.*non-negative"):
        _layout(window)


@pytest.mark.cpu
@pytest.mark.parametrize("backend", ["maskless", "unknown"])
def test_layout_rejects_removed_backends(backend: str) -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        _layout(backend=backend)


def _numerical_comparison(device: str, backend: str, dtype: torch.dtype, *, compiled: bool = False) -> None:
    torch.manual_seed(37)
    layout = _layout(backend=backend)
    q_block, kv_block = layout.block_sizes
    q_len = math.ceil(layout.gen_tokens / q_block) * q_block
    und_len = math.ceil(layout.max_und_tokens / kv_block) * kv_block
    geometry = PaddedAttentionGeometry(layout.gen_tokens, q_len, 5, und_len)
    q = torch.randn(1, layout.gen_tokens, 4, 128, device=device, dtype=dtype)
    k = torch.randn(1, layout.gen_tokens, 2, 128, device=device, dtype=dtype)
    v = torch.randn_like(k)
    ku = torch.randn(1, 5, 2, 128, device=device, dtype=dtype)
    vu = torch.randn_like(ku)
    keys, values = torch.cat([ku, k], dim=1), torch.cat([vu, v], dim=1)
    dense_mask = _dense_reference(layout, geometry)[: layout.gen_tokens]
    dense_mask = torch.cat([dense_mask[:, :5], dense_mask[:, und_len : und_len + layout.gen_tokens]], dim=1).to(device)
    scores = q.float().transpose(1, 2) @ keys.float().repeat_interleave(2, dim=2).transpose(1, 2).transpose(-1, -2)
    scores = scores / math.sqrt(128)
    scores.masked_fill_(~dense_mask, -float("inf"))
    expected = (scores.softmax(-1) @ values.float().repeat_interleave(2, dim=2).transpose(1, 2)).transpose(1, 2)
    context = MultiviewAttentionContext(layout, {})
    if compiled:
        from vllm_omni.diffusion.models.cosmos3.multiview_fa4 import multiview_fa4_attention

        plan, padded = get_multiview_attention_plan(
            context, real_und_len=5, real_q_len=layout.gen_tokens, device=q.device
        )
        attention = torch.compile(multiview_fa4_attention, fullgraph=True)
        actual = attention(
            _pack_padded_bshd((q, padded.padded_q_len)),
            _pack_padded_bshd((ku, padded.padded_und_len), (k, padded.padded_q_len)),
            _pack_padded_bshd((vu, padded.padded_und_len), (v, padded.padded_q_len)),
            plan,
        )[:, : layout.gen_tokens]
    else:
        actual = padded_multiview_flex_attention(q, k, v, ku, vu, context)
    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
        atol=0.01 if dtype == torch.bfloat16 else 1e-5,
    )


@pytest.mark.cpu
def test_triton_attention_numerically_matches_dense_on_cpu() -> None:
    _numerical_comparison("cpu", "triton", torch.float32)


@pytest.mark.gpu
@pytest.mark.parametrize(("backend", "compiled"), [("triton", False), ("fa4", False), ("fa4", True)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA numerical comparison")
def test_cuda_attention_numerically_matches_dense(backend: str, compiled: bool) -> None:
    # Triton's production entrypoint compiles the dynamic-shape Flex kernel.
    _numerical_comparison("cuda", backend, torch.bfloat16, compiled=compiled)


# -- FA4 vector mask uniform-run fast path -----------------------------------
#
# The SM100/SM110 vector callback answers a whole 32-key vector from one table
# bit when the vector's first and last key share a semantic run. These CPU tests
# pin the two invariants that fast path relies on: key run ids never decrease,
# and the packed result equals the exact per-key fallback (and the predicate) for
# every lane -- including short runs where neighbouring lanes fall in different
# runs and the fallback loop must run.


def _fa4_metadata_and_sparsity(layout: MultiviewLayout, real_und_len: int):
    q_block, kv_block = layout.block_sizes
    q_len = math.ceil(layout.gen_tokens / q_block) * q_block
    und_len = math.ceil(layout.max_und_tokens / kv_block) * kv_block
    geometry = PaddedAttentionGeometry(layout.gen_tokens, q_len, real_und_len, und_len)
    metadata = build_multiview_flex_metadata(layout, geometry, torch.device("cpu"))
    sparsity = build_multiview_block_sparsity(metadata, q_block_size=q_block, kv_block_size=kv_block)
    return metadata, sparsity


@pytest.mark.cpu
def test_fa4_key_run_ids_never_decrease() -> None:
    # One layout exercises captions, control, LiDAR, and padding at once; the run
    # ids over its key (and query) fields must be monotonically non-decreasing.
    _, sparsity = _fa4_metadata_and_sparsity(_layout(backend="fa4"), real_und_len=5)
    k_group_ids = sparsity.k_group_ids
    assert int(k_group_ids[0]) == 0
    assert bool((k_group_ids[1:] >= k_group_ids[:-1]).all()), "key run ids must never decrease"
    # The same invariant on the raw grouping helper, across every field tuple.
    metadata, _ = _fa4_metadata_and_sparsity(_layout(backend="fa4"), real_und_len=5)
    for vectors in (metadata.key_grouping_vectors(), metadata.query_grouping_vectors()):
        group_ids, _ = _semantic_groups(vectors)
        assert bool((group_ids[1:] >= group_ids[:-1]).all())
        assert int(group_ids.min()) == 0


class _U32(int):
    """Minimal unsigned 32-bit int so the CuTe callback runs on the CPU."""

    _M = 0xFFFFFFFF

    def __new__(cls, value: int) -> "_U32":
        return int.__new__(cls, int(value) & cls._M)

    def __add__(self, o): return _U32((int(self) + int(o)) & self._M)
    def __sub__(self, o): return _U32((int(self) - int(o)) & self._M)
    def __lshift__(self, o): return _U32((int(self) << int(o)) & self._M)
    def __rshift__(self, o): return _U32(int(self) >> int(o))
    def __and__(self, o): return _U32(int(self) & int(o))
    def __or__(self, o): return _U32(int(self) | int(o))
    def __floordiv__(self, o): return _U32(int(self) // int(o))
    def __mod__(self, o): return _U32(int(self) % int(o))
    def __eq__(self, o): return int(self) == int(o)
    def __hash__(self): return int.__hash__(self)


class _Vec:
    def __init__(self, data) -> None:
        self.data = list(data)

    def __getitem__(self, i):
        return self.data[i]

    @property
    def shape(self):
        return self


class _Res:
    def __init__(self, n: int) -> None:
        self.data = [0] * n

    def __getitem__(self, i):
        return self.data[i]

    def __setitem__(self, i, v) -> None:
        self.data[i] = v

    def load(self):
        return self.data[0] if len(self.data) == 1 else list(self.data)


def _cute_cpu_stubs():
    import types

    cutlass = types.SimpleNamespace(
        Uint32=_U32,
        Int32=lambda v: int(v),
        Boolean=lambda v: bool(int(v)),
        const_expr=lambda v: v,
        range_constexpr=lambda n: range(n),
    )
    cute = types.SimpleNamespace(
        jit=lambda fn: fn,
        size=lambda s: len(s.data) if hasattr(s, "data") else int(s),
        make_rmem_tensor=lambda shape, dtype=None: _Res(len(shape.data) if hasattr(shape, "data") else int(shape)),
    )
    fa_utils = types.SimpleNamespace(shr_u32=lambda w, s: _U32(int(w) >> int(s)))
    return cutlass, cute, fa_utils


@pytest.mark.cpu
def test_fa4_vector_mask_matches_predicate_and_fallback_on_cpu() -> None:
    from vllm_omni.diffusion.models.cosmos3.multiview_fa4 import _build_mask_mod

    # The toy layout has runs far shorter than 32, so most 32-key vectors span a
    # run boundary and take the per-key fallback; the padding tail is one long
    # run that takes the uniform-run fast path. Both must agree with the exact
    # predicate, lane for lane.
    layout = _layout(backend="fa4")
    metadata, sparsity = _fa4_metadata_and_sparsity(layout, real_und_len=5)
    q_word_base = sparsity.q_word_base.tolist()
    k_group_ids = sparsity.k_group_ids.tolist()
    allowed_words = sparsity.allowed_words.tolist()
    aux = (q_word_base, k_group_ids, allowed_words)

    cutlass, cute, fa_utils = _cute_cpu_stubs()
    vector_cb = _build_mask_mod(cutlass, cute, fa_utils, vec_size=32)
    scalar_cb = _build_mask_mod(cutlass, cute, fa_utils, vec_size=1)

    kv_len = len(k_group_ids)
    q_len = metadata.q_len
    assert kv_len % 32 == 0
    saw_fast_path = False
    saw_fallback = False
    # Exhaustive over query rows is O(q_len * kv_len); sample rows that cover the
    # UND prefix, each item boundary, and padding, which is where runs change.
    q_rows = sorted(set(range(0, q_len, 7)) | {0, q_len - 1})
    for q in q_rows:
        reference = multiview_pair_predicate(
            metadata, torch.tensor(q), torch.arange(kv_len)
        ).tolist()
        for start in range(0, kv_len, 32):
            n_idx = _Vec(list(range(start, start + 32)))
            m_idx = _Vec([q])
            packed = int(vector_cb(0, 0, m_idx, n_idx, None, aux))
            if int(k_group_ids[start]) == int(k_group_ids[start + 31]):
                saw_fast_path = True
            else:
                saw_fallback = True
            scalar = scalar_cb(0, 0, m_idx, n_idx, None, aux)
            for lane in range(32):
                bit = (packed >> lane) & 1
                assert bool(bit) == bool(scalar[lane]), (q, start, lane)
                assert bool(bit) == bool(reference[start + lane]), (q, start, lane)
    assert saw_fast_path, "fast path (uniform run) was never exercised"
    assert saw_fallback, "fallback (mixed run) was never exercised"
