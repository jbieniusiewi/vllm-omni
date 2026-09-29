# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Action-free Cosmos3-Nano-Sim-Transfer transformer."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3 import Cosmos3GenDecoderLayer, _apply_rotary_pos_emb
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.conditioning_control import (
    build_shared_vision_mrope_position_ids,
    pack_pure_vision_tokens,
    unpack_pure_vision_tokens,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.paged_attention_transfer import (
    CosmosSimTransferPagedLayerInputs,
    paged_write_attn,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_bimanual import (
    Cosmos3NanoSimBimanualGenDecoderLayer,
    Cosmos3NanoSimBimanualJointAttention,
    Cosmos3NanoSimBimanualTransformer,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import ARDiffusionPagedLayerInputs


class Cosmos3NanoSimTransferJointAttention(Cosmos3NanoSimBimanualJointAttention):
    """Transfer paged specialization; dense attention stays inherited."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        text_k: torch.Tensor,
        text_v: torch.Tensor,
        real_text_kv_len: int,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        dense_history: tuple[torch.Tensor, torch.Tensor] | None = None,
        paged_context: ARDiffusionPagedLayerInputs | CosmosSimTransferPagedLayerInputs | None = None,
        num_frames: int,
        tokens_per_frame: int,
        action_tokens_per_frame: int | None = None,
        null_action_frame_indexes: tuple[int, ...] = (),
        clean_history_window: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(paged_context, CosmosSimTransferPagedLayerInputs):
            return super().forward(
                hidden_states,
                text_k=text_k,
                text_v=text_v,
                real_text_kv_len=real_text_kv_len,
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                dense_history=dense_history,
                paged_context=paged_context,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                action_tokens_per_frame=action_tokens_per_frame,
                null_action_frame_indexes=null_action_frame_indexes,
                clean_history_window=clean_history_window,
            )
        if hidden_states.shape[0] != 1:
            raise ValueError(
                f"Cosmos3-Nano-Sim-Transfer causal attention supports batch_size=1, got {hidden_states.shape[0]}"
            )
        if real_text_kv_len <= 0 or real_text_kv_len > text_k.shape[1]:
            raise ValueError(
                "Cosmos3-Nano-Sim-Transfer real text KV length must be in the stored range, "
                f"got real={real_text_kv_len}, stored={text_k.shape[1]}"
            )
        if text_k.shape != text_v.shape:
            raise ValueError(
                f"Cosmos3-Nano-Sim-Transfer text K/V shapes differ: {tuple(text_k.shape)} != {tuple(text_v.shape)}"
            )
        if dense_history is not None and paged_context is not None:
            raise ValueError(
                "Cosmos3-Nano-Sim-Transfer attention accepts either dense_history or paged_context, not both"
            )

        batch, seq_len, _ = hidden_states.shape
        q = self.to_q(hidden_states).view(batch, seq_len, self.num_heads_local, self.head_dim)
        k = self.to_k(hidden_states).view(batch, seq_len, self.num_kv_heads_local, self.head_dim)
        v = self.to_v(hidden_states).view(batch, seq_len, self.num_kv_heads_local, self.head_dim)
        if self.qk_norm:
            q = F.rms_norm(q, (self.head_dim,), self.norm_q.weight, eps=self.norm_q.variance_epsilon)
            k = F.rms_norm(k, (self.head_dim,), self.norm_k.weight, eps=self.norm_k.variance_epsilon)
        q, k = _apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)
        if action_tokens_per_frame not in (None, 0) or null_action_frame_indexes:
            raise ValueError("Cosmos3-Nano-Sim-Transfer paged attention does not accept action tokens")

        output = paged_write_attn(
            paged_context,
            q[0],
            k[0],
            v[0],
            None,
            None,
            self.head_dim**-0.5,
            framewise_attention=clean_history_window is not None,
        ).unsqueeze(0)
        return self.to_out(output.reshape(batch, seq_len, -1)), k, v


class Cosmos3NanoSimTransferGenDecoderLayer(Cosmos3NanoSimBimanualGenDecoderLayer):
    """Transfer joint attention with the same decoder math and weight names."""

    def __init__(
        self,
        *,
        layer_idx: int | None = None,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        quant_config=None,
        mlp_cls,
        qk_norm: bool = True,
        prefix: str = "",
    ) -> None:
        # Reuse the base decoder initialization and Bimanual's forward math,
        # installing Transfer attention without changing the upstream classes.
        Cosmos3GenDecoderLayer.__init__(
            self,
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            quant_config=quant_config,
            mlp_cls=mlp_cls,
            qk_norm=qk_norm,
            prefix=prefix,
        )
        self.cross_attention = Cosmos3NanoSimTransferJointAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            quant_config=quant_config,
            qk_norm=qk_norm,
            prefix=f"{prefix}.cross_attention",
        )


class Cosmos3NanoSimTransferTransformer(Cosmos3NanoSimBimanualTransformer):
    """Pure-vision Transfer variant with no action modules or weights."""

    _gen_layer_cls = Cosmos3NanoSimTransferGenDecoderLayer
    _repeated_blocks = ["Cosmos3NanoSimTransferGenDecoderLayer"]

    def _validate_conditioning_config(self) -> None:
        self.manifest.require_control_video_conditioning()
        if self.action_gen:
            raise ValueError("Cosmos3-Nano-Sim-Transfer checkpoints must set action_gen=False.")

    def _prepare_conditioning_tokens(
        self,
        hidden_states: torch.Tensor,
        *,
        num_frames: int,
        action_latents: torch.Tensor | None,
        action_domain_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if action_latents is not None or action_domain_ids is not None:
            raise ValueError("Cosmos3-Nano-Sim-Transfer does not accept action conditioning.")
        return hidden_states.new_empty(1, num_frames, 0, self.hidden_size)

    def _pack_tokens(self, conditioning_tokens: torch.Tensor, vision_tokens: torch.Tensor) -> torch.Tensor:
        if conditioning_tokens.ndim != 4 or conditioning_tokens.shape[2] != 0:
            raise ValueError("Cosmos3-Nano-Sim-Transfer conditioning token sequence must be empty.")
        return pack_pure_vision_tokens(vision_tokens)

    def _unpack_tokens(
        self,
        hidden: torch.Tensor,
        *,
        num_frames: int,
        conditioning_tokens_per_frame: int,
        vision_tokens_per_frame: int,
    ) -> torch.Tensor:
        if conditioning_tokens_per_frame != 0:
            raise ValueError("Cosmos3-Nano-Sim-Transfer cannot unpack action conditioning tokens.")
        return unpack_pure_vision_tokens(
            hidden,
            num_frames=num_frames,
            vision_tokens_per_frame=vision_tokens_per_frame,
        )

    def _build_position_ids(
        self,
        *,
        frame_start: int,
        num_frames: int,
        grid_h: int,
        grid_w: int,
        real_text_kv_len: int,
        fps: float,
        null_action_frame_indexes: tuple[int, ...],
    ) -> torch.Tensor:
        if null_action_frame_indexes:
            raise ValueError("Cosmos3-Nano-Sim-Transfer does not accept null action frame indexes.")
        return build_shared_vision_mrope_position_ids(
            frame_start=frame_start,
            num_frames=num_frames,
            grid_h=grid_h,
            grid_w=grid_w,
            text_temporal_offset=real_text_kv_len,
            temporal_modality_margin=self.temporal_modality_margin,
            fps=fps,
            base_fps=self.base_fps,
            temporal_compression_factor=self.manifest.temporal_compression_factor,
            enable_fps_modulation=self.enable_fps_modulation,
        )


__all__ = [
    "Cosmos3NanoSimTransferTransformer",
]
