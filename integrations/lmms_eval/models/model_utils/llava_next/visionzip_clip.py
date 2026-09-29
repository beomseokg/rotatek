"""CLIP encoder monkey-patches for VisionZip token reduction.

Direct port of `JIA-Lab-research/VisionZip/visionzip/utils.py` adapted for
LLaVA-NeXT's `transformers.models.clip` stack. Stashes raw_key_states.mean(1)
as `metric` on the penultimate encoder layer so the LM-side adapter can run
dominant + contextual selection identical to the original.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from transformers.models.clip.modeling_clip import (
    CLIPAttention,
    CLIPEncoderLayer,
)


def CLIPAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    causal_attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Patched CLIPAttention.forward — additionally returns
    `raw_key_states.mean(1)` as `metric` so VisionZip can use it for
    contextual-token similarity merging.
    """
    bsz, tgt_len, embed_dim = hidden_states.size()

    query_states = self.q_proj(hidden_states) * self.scale
    key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    raw_key_states = key_states.clone()
    value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

    proj_shape = (bsz * self.num_heads, -1, self.head_dim)
    query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    key_states = key_states.view(*proj_shape)
    value_states = value_states.view(*proj_shape)

    src_len = key_states.size(1)
    attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

    if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, "
            f"but is {attn_weights.size()}"
        )

    if causal_attention_mask is not None:
        if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, "
                f"but is {causal_attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, tgt_len, src_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, "
                f"but is {attention_mask.size()}"
            )
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if output_attentions:
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
    else:
        attn_weights_reshaped = None

    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

    attn_output = torch.bmm(attn_probs, value_states)

    if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, "
            f"but is {attn_output.size()}"
        )

    attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

    attn_output = self.out_proj(attn_output)

    # NEW: third return — metric for VisionZip contextual merging.
    return attn_output, attn_weights_reshaped, raw_key_states.mean(1)


def CLIP_EncoderLayer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    causal_attention_mask: torch.Tensor,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.FloatTensor]:
    """Patched CLIPEncoderLayer.forward — receives 3-tuple from patched
    self_attn (output, weights, metric) and stashes metric on the layer
    when `_info["r"]` slot is non-zero. The LM-side adapter reads
    `encoder.layers[-2].metric` for the contextual-token similarity merge.
    """
    residual = hidden_states

    hidden_states = self.layer_norm1(hidden_states)

    hidden_states, attn_weights, metric = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
    )

    hidden_states = residual + hidden_states

    r = self._info["r"].pop(0)
    if r > 0:
        self.metric = metric

    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs


def parse_r(num_layers: int, r: List[int]) -> List[int]:
    """Per-layer flag list: r[i] > 0 marks the layer that stashes `metric`.

    VisionZip's code inherits this schedule from ToMe
    (https://github.com/facebookresearch/ToMe), where r is a per-layer merge
    count; here it is only ever the list set by `apply_info` (1 at the
    penultimate layer), so ToMe's int / tuple schedules are omitted.
    """
    return list(r) + [0] * (num_layers - len(r))


def make_visionzip_class(transformer_class):
    """Wrap the vision_tower class so each forward primes `_info["r"]`."""

    class VisionZipTransformer(transformer_class):
        def forward(self, *args, **kwargs) -> torch.Tensor:
            self._info["r"] = parse_r(
                len(self.vision_model.encoder.layers), self.r
            )
            self._info["size"] = None
            self._info["source"] = None
            return super().forward(*args, **kwargs)

    return VisionZipTransformer


def apply_info(model) -> None:
    """Wire VisionZip metric stashing into a CLIP vision tower.

    1. Promote the vision tower's class to a VisionZipTransformer subclass so
       every forward primes the per-layer r schedule.
    2. Set `model.r` so r=1 only at the penultimate encoder layer; that's the
       layer whose `metric = raw_key_states.mean(1)` will be stashed for the
       LM-side adapter to read via `encoder.layers[-2].metric`.
    3. Share `_info` with every CLIPEncoderLayer so the patched layer forward
       can pop its r value each call.
    """
    num_layers = len(model.vision_model.encoder.layers)
    VisionZipTransformer = make_visionzip_class(model.__class__)
    model.__class__ = VisionZipTransformer

    model.r = [0] * (num_layers - 2) + [1] + [0]
    model._info = {"r": [model.r]}
    for module in model.modules():
        if isinstance(module, CLIPEncoderLayer):
            module._info = model._info


def patch_clip_vision_tower(vision_tower) -> None:
    """Swap CLIPAttention.forward and CLIPEncoderLayer.forward in-place.

    Call once after model load. `apply_info` must be called separately to
    install dominant/contextual numbers and the r schedule.
    """
    import types

    for module in vision_tower.modules():
        if isinstance(module, CLIPAttention):
            module.forward = types.MethodType(CLIPAttention_forward, module)
        elif isinstance(module, CLIPEncoderLayer):
            module.forward = types.MethodType(CLIP_EncoderLayer_forward, module)
