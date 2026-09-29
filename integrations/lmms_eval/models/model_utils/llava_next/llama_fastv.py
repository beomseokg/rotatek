# ------------------------------------------------------------------------
# LLaMA backbone with FastV (token-level pruning) for LLaVA-NeXT, combined
# with the channel-pruned KV cache of `llama_visionzip.py`.
#
# FastV (inplace), as in the reference implementation:
#   * At decoder layer K, score each visual token by the last query row of
#     layer K-1's head-averaged attention and keep the top fast_v_keep_ratio.
#   * Slice `hidden_states` / `position_ids` at layer K, so layers >= K cache
#     a shorter KV.
#
# Differences from the reference FastV:
#   * Transformers 4.49 API (Cache object, position_embeddings kwarg).
#   * The visual span is computed per forward (LLaVA-NeXT anyres makes its
#     length image-dependent) and stamped on the model by
#     `LlavaNextFastVForConditionalGeneration`.
#   * Only prefill layer K-1 runs eager attention; every other layer and all
#     decode steps use FlashAttention-2. RotateK decode runs its fused kernel.
# ------------------------------------------------------------------------

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaPreTrainedModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    repeat_kv,
)

from lmms_eval.models.model_utils.cache_utils import Cache, DynamicCache
from lmms_eval.models.model_utils.kv_pruning_utils import init_visionzip
from lmms_eval.models.model_utils.llava_next.llama_visionzip import (
    LlamaVisionZipAttention,
    LlamaVisionZipDecoderLayer,
)
from rotatek.kernels.fused_decode import rotatek_decode_fused_triton


# Projections come from LlamaVisionZipAttention; the forward adds an eager
# path that returns attention weights, which FastV reads from layer K-1.
class LlamaFastVAttention(LlamaVisionZipAttention):
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
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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
                attn_output = attn_output.unsqueeze(1).reshape(bsz, q_len, self.hidden_size)
            else:
                bsz_r, h_kv, seq_r = key_pruned.shape[:3]
                if channel_method == "think":
                    # pruned channels are zero
                    think_mask = past_key_value.think_mask[self.layer_idx]  # [B, H_kv, D]
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
            if output_attentions:
                # eager, only on prefill layer K-1 (FastV reads the weights)
                kv_seq_len = k_for_compute.shape[-2]
                k_rep = repeat_kv(k_for_compute, self.num_key_value_groups)
                v_rep = repeat_kv(v_for_compute, self.num_key_value_groups)
                attn_weights = torch.matmul(query_states, k_rep.transpose(2, 3)) * self.scaling
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask[:, :, :, :kv_seq_len]
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
                    query_states.dtype
                )
                attn_output = torch.matmul(attn_weights, v_rep)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
            else:
                # FA2 takes a 2D padding mask (or None) and GQA natively. The
                # model builds a 4D causal mask for the eager layer; with batch
                # size 1 and no padding, None + is_causal=True is equivalent.
                attn_output = _flash_attention_forward(
                    query_states.transpose(1, 2),
                    k_for_compute.transpose(1, 2),
                    v_for_compute.transpose(1, 2),
                    attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None,
                    q_len,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    sliding_window=None,
                    is_causal=self.is_causal,
                    use_top_left_mask=self._flash_attn_uses_top_left_mask,
                )
                attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class LlamaFastVDecoderLayer(LlamaVisionZipDecoderLayer):
    """LlamaVisionZipDecoderLayer with `LlamaFastVAttention` swapped in."""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LlamaFastVAttention(config=config, layer_idx=layer_idx)


class LlamaFastVModel(LlamaPreTrainedModel):
    """LLaMA LM with the FastV slice at layer K.

    Per-forward state set by LlavaNextFastVForConditionalGeneration:
      fast_v_sys_length          -- start of the visual span in the merged sequence
      fast_v_pre_merge_text_len  -- input_ids length before the image merge
      fast_v_num_placeholders    -- number of <image> placeholders
    The visual span length is derived here:
      IMG = inputs_embeds.shape[1] - pre_merge_text_len + num_placeholders
    Config: fast_v_agg_layer (K), fast_v_keep_ratio, channel_ratio, channel_method.
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaFastVDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # set at prefill; kept across decode steps (the cache already holds
        # the visual span)
        self.fast_v_sys_length: Optional[int] = None
        self.fast_v_pre_merge_text_len: Optional[int] = None
        self.fast_v_num_placeholders: Optional[int] = None

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

        # HF `generate()` seeds its own (empty) cache; swap it for ours.
        if not isinstance(past_key_values, DynamicCache):
            past_key_values = DynamicCache()
        past_seen_tokens = past_key_values.get_seq_length()

        agg_layer = int(self.config.fast_v_agg_layer)
        do_fastv_prefill = seq_len > 1 and self.fast_v_sys_length is not None
        if do_fastv_prefill:
            sys_length = int(self.fast_v_sys_length)
            image_token_length = int(
                seq_len - self.fast_v_pre_merge_text_len + self.fast_v_num_placeholders
            )
            do_fastv_prefill = image_token_length > 0
            attention_rank = max(1, int(round(float(self.config.fast_v_keep_ratio) * image_token_length)))

        # The 4D mask is only used by the eager layer K-1; FA2 layers drop it
        # (see LlamaFastVAttention).
        causal_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (bsz, seq_len), inputs_embeds, past_seen_tokens
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        layer_attention_mask = causal_mask
        layer_position_ids = position_ids
        layer_position_embeddings = position_embeddings
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
                layer_position_ids = keep_indexs.unsqueeze(0)
                layer_position_embeddings = self.rotary_emb(hidden_states, layer_position_ids)
                # layers >= K start with an empty cache: causal mask for the
                # shortened sequence with past=0
                new_seq_len = keep_indexs.shape[0]
                layer_attention_mask = _prepare_4d_causal_attention_mask(
                    None, (bsz, new_seq_len), hidden_states, 0
                )

            # only prefill layer K-1 needs attention weights (eager)
            need_attn = do_fastv_prefill and layer_idx == agg_layer - 1
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_ids=layer_position_ids,
                past_key_value=past_key_values,
                output_attentions=need_attn,
                use_cache=use_cache,
                cache_position=cache_position if not (do_fastv_prefill and layer_idx >= agg_layer) else None,
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


class LlamaFastVForCausalLM(LlamaForCausalLM):
    """LlamaForCausalLM with the FastV LlamaModel. Checkpoint keys are
    identical to upstream LLaMA, so `from_pretrained` loads vanilla weights."""

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaFastVModel(config)
        self.post_init()
