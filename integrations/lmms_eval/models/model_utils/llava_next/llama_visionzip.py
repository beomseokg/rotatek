# ------------------------------------------------------------------------
# LLaMA language model for LLaVA-NeXT with the channel-pruned KV cache.
# Modified from transformers 4.49.0 LLaMA; mirrors the hooks in
# `lmms_eval.models.model_utils.qwen.qwen2_5vl_visionzip`.
#
# Prefill attends over the full K/V and caches the pruned visual Keys
# (ThinK / SparK / RotateK, see kv_pruning_utils). Decode either recovers
# full-width Keys for dense attention (ThinK / SparK) or runs the RotateK fused
# kernel on the rotated-truncated Keys. `decode_attention_backend="triton"`
# (latency runs) routes the dense decode through the Triton split-K kernel so
# every method is timed on Triton.
#
# `_decode_profile` (set by scripts/paper/latency) records per-stage decode
# timings for the layers in `_decode_profile_layers`.
# ------------------------------------------------------------------------

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import is_flash_attn_greater_or_equal_2_10
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaRotaryEmbedding,
    LlamaPreTrainedModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
)

from lmms_eval.models.model_utils.kv_pruning_utils import init_channel_pruner
from lmms_eval.models.model_utils.cache_utils import Cache, DynamicCache
from rotatek.kernels.fused_decode import rotatek_decode_fused_triton
from rotatek.kernels.full_channel_flash_decoding import full_channel_decode_triton


DECODE_STAGES = (
    "qkv_proj", "rope", "cache_update", "recovery", "cat",
    "custom_decode_kernel", "transpose", "fa2", "attn_out_proj",
)


class _StageTimer:
    """CUDA-event timing of named decode stages; a no-op when disabled.

    On `flush`, appends one value per stage in DECODE_STAGES (0.0 for stages
    that did not run this step) to config._decode_timings_by_layer[layer].
    """

    def __init__(self, enabled):
        self.enabled = enabled
        self.events = {}

    def start(self, name):
        if self.enabled:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.events[name] = [e, None]

    def stop(self, name):
        if self.enabled:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.events[name][1] = e

    def flush(self, config, layer_idx):
        if not self.enabled:
            return
        torch.cuda.synchronize()
        by_layer = getattr(config, "_decode_timings_by_layer", None)
        if by_layer is None:
            by_layer = config._decode_timings_by_layer = {}
        timings = by_layer.setdefault(layer_idx, {f"{k}_ms": [] for k in DECODE_STAGES})
        for k in DECODE_STAGES:
            ev = self.events.get(k)
            timings[f"{k}_ms"].append(ev[0].elapsed_time(ev[1]) if ev and ev[1] is not None else 0.0)


class LlamaVisionZipAttention(nn.Module):
    """LLaMA flash-attention forward with the channel-pruned KV cache.

    Weights match `transformers.models.llama.modeling_llama.LlamaAttention`;
    only the forward pass is replaced, so existing checkpoints load as-is.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()
        channel_method = self.config.channel_method
        channel_ratio = self.config.channel_ratio
        use_triton_decode = getattr(self.config, "decode_attention_backend", "fa2") == "triton"
        timer = _StageTimer(
            getattr(self.config, "_decode_profile", False)
            and self.layer_idx in getattr(self.config, "_decode_profile_layers", {14})
            and q_len == 1
        )

        if q_len > 1:
            init_channel_pruner(self)

        timer.start("qkv_proj")
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        timer.stop("qkv_proj")

        timer.start("rope")
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        timer.stop("rope")

        attn_output = None  # set here when a Triton decode kernel runs
        if q_len > 1:  # prefill: attend over the full K/V, cache the pruned one
            if channel_ratio == 0.0:
                past_key_value.store_unified(key_states, value_states, self.layer_idx)
            else:
                update = {"think": self.channel_pruner.update_think,
                          "spark": self.channel_pruner.update_spark,
                          "rotatek": self.channel_pruner.update_rotatek}[channel_method]
                kv_pruned, kv_prompt, kv_text, mask, value_states_compress = update(
                    key_states, query_states, value_states, attention_mask,
                    num_key_value_groups=self.num_key_value_groups,
                )
                past_key_value.store_pruned(kv_pruned, kv_prompt, kv_text, mask, value_states_compress, self.layer_idx)
                if channel_method == "think":
                    past_key_value.think_mask.append(self.channel_pruner.current_think_mask)
                elif channel_method == "spark":
                    past_key_value.spark_mask.append(self.channel_pruner.current_spark_mask)
                    past_key_value.spark_pruned_mean.append(self.channel_pruner.current_spark_pruned_mean)
                else:
                    past_key_value.rotatek_rotations.append(self.channel_pruner.current_rotatek_R_partial)
                    past_key_value.rotatek_means.append(self.channel_pruner.current_rotatek_delta_mu)

        elif channel_ratio == 0.0:  # decode, uncompressed
            timer.start("cache_update")
            key_states, value_states = past_key_value.update_unified(key_states, value_states, self.layer_idx)
            timer.stop("cache_update")
            if use_triton_decode:
                timer.start("custom_decode_kernel")
                attn_output = full_channel_decode_triton(
                    q=query_states.squeeze(-2), k=key_states, v=value_states,
                    num_kv_groups=self.num_key_value_groups,
                )
                timer.stop("custom_decode_kernel")

        else:  # decode, compressed
            timer.start("cache_update")
            text_key_states, value_states, key_pruned, key_prompt, mask = past_key_value.update(
                key_states, value_states, self.layer_idx
            )
            timer.stop("cache_update")

            if channel_method == "rotatek":
                # Keys stay rotated and truncated; the fused kernel rotates q
                # and adds the δμ bias itself.
                timer.start("custom_decode_kernel")
                s_prompt, s_vision = key_prompt.shape[-2], key_pruned.shape[-2]
                q_squeezed = query_states.squeeze(2)  # [B, H_q, D]
                attn_output, _, _ = rotatek_decode_fused_triton(
                    q_full=q_squeezed,
                    R_partial=past_key_value.rotatek_rotations[self.layer_idx],
                    delta_mu=past_key_value.rotatek_means[self.layer_idx],
                    k_full=torch.cat([key_prompt, text_key_states], dim=-2),
                    v_full=torch.cat([value_states[:, :, :s_prompt], value_states[:, :, s_prompt + s_vision:]], dim=-2),
                    mask_full=torch.ones(bsz, s_prompt + text_key_states.shape[-2],
                                         device=q_squeezed.device, dtype=torch.uint8),
                    k_sparse=key_pruned,
                    v_sparse=value_states[:, :, s_prompt:s_prompt + s_vision],
                    num_kv_groups=self.num_key_value_groups,
                )
                timer.stop("custom_decode_kernel")
            else:
                # ThinK / SparK recover full-width visual Keys for dense attention
                timer.start("recovery")
                bsz_r, h_kv, seq_r = key_pruned.shape[:3]
                if channel_method == "think":
                    # pruned channels are zero
                    think_mask = past_key_value.think_mask[self.layer_idx]  # [B, H_kv, D]
                    recovered_key_states = torch.zeros(bsz_r, h_kv, seq_r, self.head_dim,
                                                       dtype=key_pruned.dtype, device=key_pruned.device)
                    recovered_key_states[think_mask.unsqueeze(2).expand(-1, -1, seq_r, -1)] = key_pruned.reshape(-1)
                else:
                    # SparK: pruned channels take the token's pruned-channel mean
                    pruned_mean = past_key_value.spark_pruned_mean[self.layer_idx]
                    recovered_key_states = pruned_mean.expand(bsz_r, h_kv, seq_r, self.head_dim).contiguous()
                    recovered_key_states[past_key_value.spark_mask[self.layer_idx]] = key_pruned.reshape(-1)
                timer.stop("recovery")

                timer.start("cat")
                key_states = torch.cat([key_prompt, recovered_key_states, text_key_states], dim=-2)
                timer.stop("cat")

                if use_triton_decode:
                    timer.start("custom_decode_kernel")
                    attn_output = full_channel_decode_triton(
                        q=query_states.squeeze(-2), k=key_states, v=value_states,
                        num_kv_groups=self.num_key_value_groups,
                    )
                    timer.stop("custom_decode_kernel")

        if attn_output is None:
            timer.start("transpose")
            q_fa = query_states.transpose(1, 2)
            k_fa = key_states.transpose(1, 2)
            v_fa = value_states.transpose(1, 2)
            timer.stop("transpose")

            timer.start("fa2")
            attn_output = _flash_attention_forward(
                q_fa,
                k_fa,
                v_fa,
                attention_mask,
                q_len,
                dropout=0.0 if not self.training else self.attention_dropout,
                sliding_window=None,
                is_causal=self.is_causal,
                use_top_left_mask=self._flash_attn_uses_top_left_mask,
            )
            timer.stop("fa2")

        timer.start("attn_out_proj")
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        timer.stop("attn_out_proj")

        timer.flush(self.config, self.layer_idx)
        return attn_output, None, past_key_value


class LlamaVisionZipDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = LlamaVisionZipAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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


class LlamaVisionZipModel(LlamaPreTrainedModel):
    """LlamaModel with the channel-pruning decoder layers. Swaps HF's cache for
    `cache_utils.DynamicCache`, whose `store_pruned` / `update` keep the
    prompt / visual / text Key spans apart."""

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaVisionZipDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # HF `generate()` seeds its own (empty) cache, which lacks store_pruned;
        # swap it for ours on the first call. Later steps get ours back.
        if not isinstance(past_key_values, DynamicCache):
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length()
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # FA2 only needs the 2D padding mask; it applies causality itself
        # (is_causal=True), so no 4D causal mask is built.
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=None,
            attentions=None,
        )


class LlamaVisionZipForCausalLM(LlamaForCausalLM):
    """Drop-in replacement for `LlamaForCausalLM` that uses the channel-pruning
    decoder layers. Checkpoint keys are identical, so HF loading just works."""

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = LlamaVisionZipModel(config)
        self.post_init()
