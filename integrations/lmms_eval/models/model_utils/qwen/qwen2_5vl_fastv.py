# ------------------------------------------------------------------------
# Qwen2.5-VL backbone with FastV (token-level pruning). Mirrors
# `llava_next/llama_fastv.py`, adapted to Qwen2.5-VL's mrope-based LM.
#
# FastV (inplace):
#   * At decoder layer K, score each visual token by the last query row of
#     layer K-1's head-averaged attention and keep the top fast_v_keep_ratio.
#   * Slice `hidden_states` / `position_ids` at layer K, so layers >= K
#     cache a shorter KV.
#
# Channel pruning (ThinK / SparK / RotateK) uses the same kv_cluster and
# DynamicCache as `Qwen2_5_VLFlashAttention2` in `qwen2_5vl_visionzip.py`.
# Only prefill layer K-1 runs eager attention (FastV reads its weights);
# every other layer and all decode steps use FlashAttention-2.
#
# Differences vs. LLaVA-NeXT FastV:
#   * 3D mrope: `position_ids` has shape [3, B, S]. After the FastV slice we
#     index along the last dim and recompute (cos, sin) on the kept positions.
#   * Image-token bounds are read directly from `input_ids` (Qwen does not
#     expand placeholders post-merge, unlike LLaVA-NeXT anyres).
# ------------------------------------------------------------------------

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

from lmms_eval.models.model_utils.cache_utils import Cache
from lmms_eval.models.model_utils.kv_pruning_utils import init_visionzip
from lmms_eval.models.model_utils.qwen import qwen2_5vl_visionzip as _vz_mod
from lmms_eval.models.model_utils.qwen.qwen2_5vl_visionzip import (
    Qwen2_5_VLAttention,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLRotaryEmbedding,
    Qwen2MLP,
    Qwen2RMSNorm,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)
from rotatek.kernels.fused_decode import rotatek_decode_fused_triton


class Qwen2_5_VLFastVAttention(Qwen2_5_VLAttention):
    """Qwen2.5-VL attention with the channel-pruned KV cache; returns attention
    weights when asked (eager), which FastV needs from layer K-1."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = True,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        channel_method = self.config.channel_method
        channel_ratio = self.config.channel_ratio

        if q_len > 1:
            init_visionzip(self)

        query_states = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        attn_output = None  # set by the RotateK fused decode kernel
        if q_len > 1:  # prefill: attend over the full K/V, cache the pruned one
            if channel_ratio == 0.0:
                past_key_value.store_unified(key_states, value_states, self.layer_idx)
            else:
                update = {"think": self.kv_cluster.update_think,
                          "spark": self.kv_cluster.update_spark,
                          "rotatek": self.kv_cluster.update_rotatek}[channel_method]
                kv_pruned, kv_prompt, kv_text, mask, value_states_compress = update(
                    key_states, query_states, value_states, attention_mask,
                    num_key_value_groups=self.num_key_value_groups,
                )
                past_key_value.store_pruned(kv_pruned, kv_prompt, kv_text, mask, value_states_compress, self.layer_idx)
                if channel_method == "think":
                    past_key_value.think_mask.append(self.kv_cluster.current_think_mask)
                elif channel_method == "spark":
                    past_key_value.spark_mask.append(self.kv_cluster.current_spark_mask)
                    past_key_value.spark_pruned_mean.append(self.kv_cluster.current_spark_pruned_mean)
                else:
                    past_key_value.rotatek_rotations.append(self.kv_cluster.current_rotatek_R_partial)
                    past_key_value.rotatek_means.append(self.kv_cluster.current_rotatek_delta_mu)
            k_for_compute, v_for_compute = key_states, value_states

        elif channel_ratio == 0.0:  # decode, uncompressed
            k_for_compute, v_for_compute = past_key_value.update_unified(key_states, value_states, self.layer_idx)

        else:  # decode, compressed
            text_key_states, v_for_compute, key_pruned, key_prompt, mask = past_key_value.update(
                key_states, value_states, self.layer_idx
            )
            if channel_method == "rotatek":
                s_prompt, s_vision = key_prompt.shape[-2], key_pruned.shape[-2]
                q_squeezed = query_states.squeeze(2)  # [B, H_q, D]
                attn_output, _, _ = rotatek_decode_fused_triton(
                    q_full=q_squeezed,
                    R_partial=past_key_value.rotatek_rotations[self.layer_idx],
                    delta_mu=past_key_value.rotatek_means[self.layer_idx],
                    k_full=torch.cat([key_prompt, text_key_states], dim=-2),
                    v_full=torch.cat([v_for_compute[:, :, :s_prompt], v_for_compute[:, :, s_prompt + s_vision:]], dim=-2),
                    mask_full=torch.ones(bsz, s_prompt + text_key_states.shape[-2],
                                         device=q_squeezed.device, dtype=torch.uint8),
                    k_sparse=key_pruned,
                    v_sparse=v_for_compute[:, :, s_prompt:s_prompt + s_vision],
                    num_kv_groups=self.num_key_value_groups,
                )
                attn_output = attn_output.unsqueeze(1).reshape(bsz, q_len, -1)
            else:
                bsz_r, h_kv, seq_r = key_pruned.shape[:3]
                if channel_method == "think":
                    # pruned channels are zero
                    think_mask = past_key_value.think_mask[self.layer_idx]
                    recovered = torch.zeros(bsz_r, h_kv, seq_r, self.head_dim,
                                            dtype=key_pruned.dtype, device=key_pruned.device)
                    recovered[think_mask.unsqueeze(2).expand(-1, -1, seq_r, -1)] = key_pruned.reshape(-1)
                else:
                    # SparK: pruned channels take the token's pruned-channel mean
                    pruned_mean = past_key_value.spark_pruned_mean[self.layer_idx]
                    recovered = pruned_mean.expand(bsz_r, h_kv, seq_r, self.head_dim).contiguous()
                    recovered[past_key_value.spark_mask[self.layer_idx]] = key_pruned.reshape(-1)
                k_for_compute = torch.cat([key_prompt, recovered, text_key_states], dim=-2)

        attn_weights = None
        if attn_output is None:
            if not output_attentions:
                # FA2 takes a 2D padding mask (or None) and GQA natively
                attn_output = _flash_attention_forward(
                    query_states.transpose(1, 2),
                    k_for_compute.transpose(1, 2),
                    v_for_compute.transpose(1, 2),
                    attention_mask if (attention_mask is not None and attention_mask.dim() == 2) else None,
                    q_len,
                    is_causal=self.is_causal,
                )
                attn_output = attn_output.contiguous().reshape(bsz, q_len, -1)
            else:
                # eager, only on prefill layer K-1
                kv_seq_len = k_for_compute.shape[-2]
                k_rep = repeat_kv(k_for_compute, self.num_key_value_groups)
                v_rep = repeat_kv(v_for_compute, self.num_key_value_groups)
                attn_weights = torch.matmul(query_states, k_rep.transpose(2, 3)) / math.sqrt(self.head_dim)
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask[:, :, :, :kv_seq_len]
                if query_states.dtype == torch.float16:
                    attn_weights = torch.where(
                        torch.isinf(attn_weights),
                        torch.zeros_like(attn_weights),
                        attn_weights,
                    )
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
                attn_output = torch.matmul(attn_weights, v_rep)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class Qwen2_5_VLFastVDecoderLayer(nn.Module):
    """Qwen2_5_VLDecoderLayer with the FastV attention built directly (skipping
    the QWEN2_5_VL_ATTENTION_CLASSES lookup and its wasted allocation)."""

    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2_5_VLFastVAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class Qwen2_5_VLFastVModel(Qwen2_5_VLPreTrainedModel):
    """Qwen2.5-VL LM with the FastV slice at layer K.

    Per-forward state set by Qwen2_5_VLFastVForConditionalGeneration:
      fast_v_sys_length         -- start of the visual span in input_ids
      fast_v_image_token_length -- number of visual tokens
    Config: fast_v_agg_layer (K), fast_v_keep_ratio, channel_ratio, channel_method.
    """

    def __init__(self, config: Qwen2_5_VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2_5_VLFastVDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self._attn_implementation = "eager"
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2_5_VLRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.fast_v_sys_length: Optional[int] = None
        self.fast_v_image_token_length: Optional[int] = None

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        bsz, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        past_seen_tokens = past_key_values.get_seq_length()

        agg_layer = int(self.config.fast_v_agg_layer)
        do_fastv_prefill = (
            seq_len > 1
            and self.fast_v_sys_length is not None
            and self.fast_v_image_token_length is not None
            and self.fast_v_image_token_length > 0
        )
        if do_fastv_prefill:
            sys_length = int(self.fast_v_sys_length)
            image_token_length = int(self.fast_v_image_token_length)
            attention_rank = max(1, int(round(float(self.config.fast_v_keep_ratio) * image_token_length)))

        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (bsz, seq_len), inputs_embeds, past_seen_tokens
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        layer_attention_mask = causal_mask
        layer_position_ids = position_ids
        layer_position_embeddings = position_embeddings
        layer_cache_position = cache_position
        last_attn_weights = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if do_fastv_prefill and layer_idx == agg_layer:
                # As in FastV: head-mean -> last query row -> image columns -> top-k.
                avg_heads = last_attn_weights.mean(dim=1)[0]            # [Q, K]
                last_row = avg_heads[-1]                                # [K]
                img_scores = last_row[sys_length: sys_length + image_token_length]
                top_idx = img_scores.topk(attention_rank).indices + sys_length

                keep_indexs = torch.cat([
                    torch.arange(sys_length, device=device),
                    top_idx,
                    torch.arange(sys_length + image_token_length, seq_len, device=device),
                ]).sort().values

                hidden_states = hidden_states[:, keep_indexs, :]
                # mrope: slice the [3, B, S] tensor along the seq dim.
                layer_position_ids = position_ids[:, :, keep_indexs]
                layer_position_embeddings = self.rotary_emb(hidden_states, layer_position_ids)
                # layers >= K start with an empty cache: causal mask for the
                # shortened sequence with past=0
                new_seq_len = keep_indexs.shape[0]
                layer_attention_mask = _prepare_4d_causal_attention_mask(
                    None, (bsz, new_seq_len), hidden_states, 0
                )
                layer_cache_position = None

            # only prefill layer K-1 needs attention weights (eager)
            need_attn = do_fastv_prefill and layer_idx == agg_layer - 1
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_ids=layer_position_ids,
                past_key_value=past_key_values,
                output_attentions=need_attn,
                use_cache=use_cache,
                cache_position=layer_cache_position,
                position_embeddings=layer_position_embeddings,
            )
            hidden_states = layer_outputs[0]
            if need_attn:
                last_attn_weights = layer_outputs[1]

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )


class Qwen2_5_VLFastVForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """The VisionZip model class with the FastV LM swapped in.

    VisionZip's encoder-side pruning is disabled (dominant_ratio = 0); FastV
    prunes inside the LM instead. Each prefill stamps the visual span onto the
    LM (for FastV) and onto the config (for channel pruning) before
    delegating to the parent forward.
    """

    def __init__(self, config: Qwen2_5_VLConfig):
        config.dominant_ratio = 0.0
        config.contextual_ratio = 0.0
        config._attn_implementation = "eager"

        # Point the parent module's `Qwen2_5_VLModel` at the FastV LM while
        # super().__init__ runs, so the LM is allocated once, not twice.
        saved_model_cls = _vz_mod.Qwen2_5_VLModel
        _vz_mod.Qwen2_5_VLModel = Qwen2_5_VLFastVModel
        try:
            super().__init__(config)
        finally:
            _vz_mod.Qwen2_5_VLModel = saved_model_cls

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ):
        # reset per-sample state at every prefill so nothing leaks from the
        # previous sample
        if input_ids is not None and input_ids.shape[-1] > 1:
            self.model.fast_v_sys_length = None
            self.model.fast_v_image_token_length = None
            self.config.prompt_seqlen = 0
            self.config.query_seqlen = 0

        if input_ids is not None and (
            pixel_values is not None or pixel_values_videos is not None
        ):
            visual_token_mask = (
                (input_ids == self.config.image_token_id)
                | (input_ids == self.config.video_token_id)
            )[0]
            idxs = torch.nonzero(visual_token_mask, as_tuple=True)[0]
            if idxs.numel() > 0:
                first = int(idxs[0].item())
                last = int(idxs[-1].item())
                self.model.fast_v_sys_length = first
                self.model.fast_v_image_token_length = last - first + 1
                # channel pruning slices K into [prompt | vision | text] with these
                self.config.prompt_seqlen = first
                self.config.query_seqlen = input_ids.shape[-1] - last

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
        )
