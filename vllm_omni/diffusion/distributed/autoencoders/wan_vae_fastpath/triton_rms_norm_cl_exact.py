# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# ruff: noqa: N803
"""Bit-exact channels-last RMSNorm(+SiLU) for the lossless Wan VAE decoder.

The lossless decoder normalizes channels-first activations: ATen's
``torch.linalg.vector_norm(x, dim=1, dtype=torch.float32)`` produces the fp32
denominators and :mod:`vllm_omni.diffusion.layers.vae_norm_epilogue` applies the
epilogue with ATen's rounding points. Every convolution around those norms then
pays two cuDNN layout transposes (NCDHW -> NDHWC input, NDHWC -> NCDHW output).

This kernel lets the decoder keep its activations channels-last while
producing the same bits. The channels of one position are contiguous, but the
fp32 sum of squares must still be accumulated in the order ATen's reduction
kernel uses for the channels-first tensor (``Reduce.cuh``, output-vectorized
config): block-height ``Y`` threads per output, ``G`` CTAs per output, lane
``j = y + Y * g`` sums channels ``j + k * Y * G`` into accumulator ``k % 4``
with FMAs, then ``((a0 + a1) + a2) + a3``, a shared-memory tree over ``y``
(``v[y] += v[y + Y / 2]``, ...), and, for ``G > 1``, a global fold where lane
``y`` adds the CTA partials ``g = y, y + Y, ...`` in order before the same tree.
:func:`aten_norm_reduce_config` reproduces ATen's ``setReduceConfig`` for this
case, and every (channels, positions, device) configuration is compared
bitwise against ATen once before it is used (:func:`rms_norm_channels_last_exact`
declines when it does not match). The epilogue is the lossless one: IEEE
``div_rn``, a rounding to the activation dtype after the divide, the scale, the
gain and the ``+ 0.0``, and the optionally fused SiLU (only for dtypes whose
fused SiLU was verified exact).
"""

from __future__ import annotations

import torch
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, tldevice, triton

logger = init_logger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_MAX_CHANNELS = 1024
_SUPPORTED_BLOCK_HEIGHTS = (1, 2, 4, 8)
# The in-order global fold handles up to this many CTAs per block-height lane.
_MAX_CTAS_PER_OUTPUT = 8
# Elements per program in each reduction load and in each epilogue tile.
_REDUCE_TILE = 1024
_EPILOGUE_TILE = 1024
_NUM_WARPS = 8
# ATen's Reduce.cuh constants for the output-vectorized reduction config.
_MAX_REDUCE_THREADS = 512
_MIN_VALUES_PER_THREAD = 16
_MAX_VALUES_PER_THREAD = 256
# {(channels, positions, dtype, has_bias, device): the kernel's denominators match ATen's}.
_VERDICTS: dict[tuple, bool] = {}
_MAX_VERDICTS = 512


def _last_pow2(n: int) -> int:
    return 1 << (n.bit_length() - 1)


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def aten_norm_reduce_config(channels: int, positions: int, device: torch.device) -> tuple[int, int]:
    """``(block_height, ctas_per_output)`` of ATen's dim-1 norm reduction over a contiguous ``(N, C, S)`` tensor.

    Mirrors ``setReduceConfig`` for a reduction that is not on the fastest
    striding dimension (output-vectorized, ``vt0 = 4``), assuming the 16-byte
    aligned fresh activations the decoder produces.
    """
    vec = 4
    while positions % vec:
        vec //= 2
    max_threads = _MAX_REDUCE_THREADS // vec
    dim0 = positions // vec
    dim0_pow2 = _last_pow2(dim0) if dim0 < max_threads else max_threads
    dim1_pow2 = _last_pow2(channels) if channels < max_threads else max_threads
    block_width = min(dim0_pow2, 32)
    block_height = min(dim1_pow2, max_threads // block_width)
    block_width = min(dim0_pow2, max_threads // block_height)
    step_output = block_width
    if channels >= block_height * _MIN_VALUES_PER_THREAD or channels >= _MAX_VALUES_PER_THREAD:
        split = block_height
    else:
        split = 1
        step_output *= block_height
    values_per_thread = _cdiv(channels, split)
    grid = _cdiv(dim0, step_output)
    props = torch.cuda.get_device_properties(device)
    target_grid = props.multi_processor_count * (props.max_threads_per_multi_processor // (block_width * block_height))
    ctas = 1
    if split > 1 and values_per_thread >= _MAX_VALUES_PER_THREAD and grid <= target_grid:
        ctas = max(
            min(_cdiv(target_grid, grid), _cdiv(values_per_thread, _MIN_VALUES_PER_THREAD)),
            _cdiv(values_per_thread, _MAX_VALUES_PER_THREAD),
        )
    return split, ctas


if HAS_TRITON:

    @triton.jit
    def _tree(t, N: tl.constexpr, BLOCK_P: tl.constexpr, LEAD: tl.constexpr):
        """``[BLOCK_P, LEAD, N] -> [BLOCK_P, LEAD]`` pairing ``v[y]`` with ``v[y + n / 2]`` per level."""
        if N >= 8:
            a, b = tl.split(tl.permute(tl.reshape(t, [BLOCK_P, LEAD, 2, N // 2]), (0, 1, 3, 2)))
            t = a + b
        if N >= 4:
            a, b = tl.split(tl.permute(tl.reshape(t, [BLOCK_P, LEAD, 2, 2]), (0, 1, 3, 2)))
            t = a + b
        if N >= 2:
            a, b = tl.split(tl.reshape(t, [BLOCK_P, LEAD, 2]))
            t = a + b
        else:
            t = tl.reshape(t, [BLOCK_P, LEAD])
        return t

    @triton.jit
    def _load_rows(X, BIAS, row, row_mask, c, c_mask, C: tl.constexpr, HAS_BIAS: tl.constexpr):
        x = tl.load(X + row[:, None] * C + c[None, :], mask=row_mask[:, None] & c_mask[None, :], other=0.0)
        v = x.to(tl.float32)
        if HAS_BIAS:
            # The producing convolution's deferred bias, rounded like ATen's ``add_``.
            bias = tl.load(BIAS + c, mask=c_mask, other=0.0).to(tl.float32)
            v = (v + bias[None, :]).to(x.dtype).to(tl.float32)
        return v

    @triton.jit
    def _fold(p, M: tl.constexpr, BLOCK_P: tl.constexpr, Y: tl.constexpr):
        """``[BLOCK_P, M, Y] -> [BLOCK_P, Y]`` summing over ``m`` in order, like ATen's global fold."""
        if M == 1:
            out = tl.reshape(p, [BLOCK_P, Y])
        elif M == 2:
            p0, p1 = tl.split(tl.permute(p, (0, 2, 1)))
            out = p0 + p1
        elif M == 4:
            # m = 2a + b
            even, odd = tl.split(tl.permute(tl.reshape(p, [BLOCK_P, 2, 2, Y]), (0, 3, 1, 2)))
            p0, p2 = tl.split(even)
            p1, p3 = tl.split(odd)
            out = ((p0 + p1) + p2) + p3
        else:
            # m = 4a + 2b + c
            c0, c1 = tl.split(tl.permute(tl.reshape(p, [BLOCK_P, 2, 2, 2, Y]), (0, 4, 1, 2, 3)))
            c0b0, c0b1 = tl.split(c0)
            c1b0, c1b1 = tl.split(c1)
            p0, p4 = tl.split(c0b0)
            p2, p6 = tl.split(c0b1)
            p1, p5 = tl.split(c1b0)
            p3, p7 = tl.split(c1b1)
            out = ((((((p0 + p1) + p2) + p3) + p4) + p5) + p6) + p7
        return out

    @triton.jit
    def _rms_norm_cl_exact_kernel(
        X,
        GAMMA,
        BIAS,
        OUT,
        DENOM,
        rows,
        scale,
        write_denom,
        C: tl.constexpr,
        Y: tl.constexpr,
        GP: tl.constexpr,
        STEP: tl.constexpr,
        K: tl.constexpr,
        SILU: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
        row_mask = row < rows
        # Lane j = g * Y + y of CTA g, thread row y; lanes past STEP (= Y * G) stay zero.
        LANES: tl.constexpr = Y * GP
        lanes = tl.arange(0, LANES)
        lane_mask = lanes < STEP
        a0 = tl.zeros([BLOCK_P, LANES], dtype=tl.float32)
        a1 = tl.zeros([BLOCK_P, LANES], dtype=tl.float32)
        a2 = tl.zeros([BLOCK_P, LANES], dtype=tl.float32)
        a3 = tl.zeros([BLOCK_P, LANES], dtype=tl.float32)
        # Channel k * STEP + j goes to accumulator k % 4, in increasing k. One load
        # per group covers k = 4 * group + 2a + b as a [BLOCK_P, 2, 2, LANES] tile.
        pair = tl.arange(0, 2)
        k_offsets = (pair[:, None, None] * 2 + pair[None, :, None]) * STEP + lanes[None, None, :]
        dtype = X.dtype.element_ty
        if K // 4 > 0:
            for group in range(K // 4):
                c = group * 4 * STEP + k_offsets
                c_mask = lane_mask[None, None, :] & (c < C)
                v = tl.load(
                    X + row[:, None, None, None] * C + c[None, :, :, :],
                    mask=row_mask[:, None, None, None] & c_mask[None, :, :, :],
                    other=0.0,
                ).to(tl.float32)
                if HAS_BIAS:
                    bias = tl.load(BIAS + c, mask=c_mask, other=0.0).to(tl.float32)
                    v = (v + bias[None, :, :, :]).to(dtype).to(tl.float32)
                even, odd = tl.split(tl.permute(v, (0, 3, 1, 2)))
                v0, v2 = tl.split(even)
                v1, v3 = tl.split(odd)
                a0 = tl.fma(v0, v0, a0)
                a1 = tl.fma(v1, v1, a1)
                a2 = tl.fma(v2, v2, a2)
                a3 = tl.fma(v3, v3, a3)
        if K % 4 > 0:
            for r in tl.static_range(K % 4):
                c = ((K // 4) * 4 + r) * STEP + lanes
                v = _load_rows(X, BIAS, row, row_mask, c, lane_mask & (c < C), C, HAS_BIAS)
                if r == 0:
                    a0 = tl.fma(v, v, a0)
                elif r == 1:
                    a1 = tl.fma(v, v, a1)
                else:
                    a2 = tl.fma(v, v, a2)
        lane_sums = ((a0 + a1) + a2) + a3
        # Block reduction per CTA, then the in-order global fold over CTAs g = y, y + Y, ...
        partial = _tree(tl.reshape(lane_sums, [BLOCK_P, GP, Y]), Y, BLOCK_P, GP)
        if GP > Y:
            partial = _fold(tl.reshape(partial, [BLOCK_P, GP // Y, Y]), GP // Y, BLOCK_P, Y)
            total = _tree(tl.reshape(partial, [BLOCK_P, 1, Y]), Y, BLOCK_P, 1)
        else:
            # Fewer CTAs than Y: the missing fold lanes are zeros, so a tree over GP equals ATen's over Y.
            total = _tree(tl.reshape(partial, [BLOCK_P, 1, GP]), GP, BLOCK_P, 1)
        total = tl.reshape(total, [BLOCK_P])
        denom = tl.maximum(tl.sqrt_rn(total), 1e-12)
        if write_denom:
            tl.store(DENOM + row, denom, mask=row_mask)

        dtype = X.dtype.element_ty
        for start in tl.static_range(0, C, BLOCK_E):
            c = start + tl.arange(0, BLOCK_E)
            c_mask = c < C
            x = _load_rows(X, BIAS, row, row_mask, c, c_mask, C, HAS_BIAS)
            gamma = tl.load(GAMMA + c, mask=c_mask, other=0.0).to(tl.float32)
            v = tl.math.div_rn(x, denom[:, None]).to(dtype).to(tl.float32)
            v = (v * scale).to(dtype).to(tl.float32)
            v = (v * gamma[None, :]).to(dtype).to(tl.float32)
            v = (v + 0.0).to(dtype)
            if SILU:
                v32 = v.to(tl.float32)
                v = tl.math.div_rn(v32, 1.0 + tldevice.exp(-v32)).to(dtype)
            tl.store(
                OUT + row[:, None] * C + c[None, :],
                v,
                mask=row_mask[:, None] & c_mask[None, :],
            )


def _rows_view(x: torch.Tensor) -> torch.Tensor | None:
    """``(positions, C)`` view of a channels-last 4D/5D tensor, or ``None`` if it is not one."""
    if x.dim() == 5:
        if not x.is_contiguous(memory_format=torch.channels_last_3d):
            return None
        rows = x.permute(0, 2, 3, 4, 1).reshape(-1, x.shape[1])
    elif x.dim() == 4:
        if not x.is_contiguous(memory_format=torch.channels_last):
            return None
        rows = x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    else:
        return None
    if rows.data_ptr() != x.data_ptr() or not rows.is_contiguous():
        return None
    return rows


def _launch(x_rows, gamma, bias, out_rows, denom, scale, config, *, silu):
    rows, channels = x_rows.shape
    block_height, ctas = config
    step = block_height * ctas
    ctas_pow2 = triton.next_power_of_2(ctas)
    k = _cdiv(channels, step)
    block_p = min(max(16, _REDUCE_TILE // (4 * block_height * ctas_pow2)), 256)
    with torch.get_device_module().device(x_rows.device):
        _rms_norm_cl_exact_kernel[(triton.cdiv(rows, block_p),)](
            x_rows,
            gamma,
            x_rows if bias is None else bias,
            out_rows,
            x_rows if denom is None else denom,
            rows,
            float(scale),
            denom is not None,
            C=channels,
            Y=block_height,
            GP=ctas_pow2,
            STEP=step,
            K=k,
            SILU=silu,
            HAS_BIAS=bias is not None,
            BLOCK_P=block_p,
            BLOCK_E=min(max(16, _EPILOGUE_TILE // block_p), triton.next_power_of_2(channels)),
            num_warps=_NUM_WARPS,
            # FMA contraction would swallow one of the low-precision roundings.
            enable_fp_fusion=False,
        )


def _matches_aten(x: torch.Tensor, x_rows, gamma, bias, out_rows, scale, config) -> bool:
    """Run the kernel once with denominators exported and compare them with ATen's channels-first reduction."""
    denom = torch.empty(x_rows.shape[0], dtype=torch.float32, device=x.device)
    _launch(x_rows, gamma, bias, out_rows, denom, scale, config, silu=False)
    reference_input = x.contiguous()
    if bias is not None:
        reference_input = reference_input + bias.view(1, -1, *([1] * (x.dim() - 2)))
    reference = torch.linalg.vector_norm(reference_input, dim=1, keepdim=True, dtype=torch.float32).clamp_min_(1e-12)
    reference = reference.movedim(1, -1).reshape(-1)
    return bool(torch.equal(denom.view(torch.int32), reference.view(torch.int32)))


def _exact_config(
    x: torch.Tensor,
    x_rows: torch.Tensor,
    gamma: torch.Tensor,
    bias: torch.Tensor | None,
    out_rows: torch.Tensor,
    scale: float,
) -> tuple[int, int] | None:
    """The reduction config for ``x``, once its denominators were verified against ATen; else ``None``."""
    rows, channels = x_rows.shape
    config = aten_norm_reduce_config(channels, rows, x.device)
    if config[0] not in _SUPPORTED_BLOCK_HEIGHTS or config[1] > _MAX_CTAS_PER_OUTPUT * config[0]:
        return None
    key = (channels, rows, x.dtype, bias is not None, x.device)
    verdict = _VERDICTS.get(key)
    if verdict is None:
        verdict = _matches_aten(x, x_rows, gamma, bias, out_rows, scale, config)
        _VERDICTS[key] = verdict
        if len(_VERDICTS) > _MAX_VERDICTS:
            del _VERDICTS[next(iter(_VERDICTS))]
        if not verdict:
            logger.info(
                "Channels-last RMSNorm does not reproduce ATen's reduction for %d channels x %d positions; "
                "using the channels-first path for this shape",
                channels,
                rows,
            )
    return config if verdict else None


def _supported(x: torch.Tensor, gamma: torch.Tensor, bias: torch.Tensor | None) -> bool:
    if not HAS_TRITON or not x.is_cuda or x.dtype not in _SUPPORTED_DTYPES or x.numel() == 0:
        return False
    channels = x.shape[1]
    if channels < 2 or channels > _MAX_CHANNELS:
        return False
    if gamma.numel() != channels or gamma.dtype is not x.dtype or gamma.device != x.device:
        return False
    return bias is None or (bias.dim() == 1 and bias.numel() == channels and bias.dtype is x.dtype)


def rms_norm_channels_last_exact(
    x: torch.Tensor,
    gamma: torch.Tensor,
    scale: float,
    *,
    silu: bool = False,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Lossless ``WanRMS_norm(x + bias)`` [then SiLU] of a channels-last ``x``, in its layout, or ``None``.

    Bit-identical to the channels-first lossless path. ``bias`` is the un-added
    per-channel bias of the convolution that produced ``x``. Callers must only
    request ``silu`` for dtypes whose fused SiLU epilogue was verified exact.
    """
    if not _supported(x, gamma, bias):
        return None
    x_rows = _rows_view(x)
    if x_rows is None:
        return None
    out = torch.empty_like(x)
    out_rows = _rows_view(out)
    if out_rows is None:
        return None
    gamma = gamma.reshape(-1).contiguous()
    bias = None if bias is None else bias.contiguous()
    config = _exact_config(x, x_rows, gamma, bias, out_rows, scale)
    if config is None:
        return None
    _launch(x_rows, gamma, bias, out_rows, None, scale, config, silu=silu)
    return out


def norm_act_cat_time_exact(
    x: torch.Tensor,
    gamma: torch.Tensor,
    scale: float,
    cache: torch.Tensor | None,
    pad_front: int,
    *,
    silu: bool,
    bias: torch.Tensor | None = None,
    keep_cache_frames: int = 2,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``[zeros | cache | rms_norm_channels_last_exact(x)]`` along time and its last frames, or ``None``.

    The channels-last counterpart of the lossless encoder's fused norm/assembly
    (:func:`..triton_norm_cache.norm_act_cat_time`): the normalized rows are
    written straight into the time-assembled convolution input, so no
    normalized temporary is materialized. Values are those of
    :func:`rms_norm_channels_last_exact`; only whole channels-last frames move
    otherwise. Requires a single-batch channels-last ``x`` and history.
    """
    if x.dim() != 5 or x.shape[0] != 1 or pad_front < 0 or keep_cache_frames < 1 or not _supported(x, gamma, bias):
        return None
    _, channels, frames, height, width = x.shape
    cache_frames = 0
    if cache is not None:
        if (
            cache.dim() != 5
            or cache.shape[:2] != x.shape[:2]
            or cache.shape[3:] != x.shape[3:]
            or cache.dtype != x.dtype
            or cache.device != x.device
            or not cache.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            return None
        cache_frames = cache.shape[2]
    if cache_frames > pad_front:
        return None
    x_rows = _rows_view(x)
    if x_rows is None:
        return None
    out_frames = pad_front + frames
    out = torch.empty(
        (1, channels, out_frames, height, width),
        dtype=x.dtype,
        device=x.device,
        memory_format=torch.channels_last_3d,
    )
    out_rows = _rows_view(out)
    if out_rows is None:
        return None
    # Frame-major rows: the normalized frames follow the front frames contiguously.
    norm_rows = out_rows[pad_front * height * width :]
    gamma = gamma.reshape(-1).contiguous()
    bias = None if bias is None else bias.contiguous()
    config = _exact_config(x, x_rows, gamma, bias, norm_rows, scale)
    if config is None:
        return None
    _launch(x_rows, gamma, bias, norm_rows, None, scale, config, silu=silu)
    zero_frames = pad_front - cache_frames
    if zero_frames:
        out[:, :, :zero_frames].zero_()
    if cache_frames:
        out[:, :, zero_frames:pad_front].copy_(cache)
    keep = out[:, :, out_frames - min(keep_cache_frames, out_frames) :].clone(memory_format=torch.channels_last_3d)
    return out, keep


__all__ = ["aten_norm_reduce_config", "norm_act_cat_time_exact", "rms_norm_channels_last_exact"]
