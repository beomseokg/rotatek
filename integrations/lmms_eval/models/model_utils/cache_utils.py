# Adapted from the ThinK implementation:
# https://github.com/SalesforceAIResearch/ThinK/blob/main/ThinK_flash/src/cache_utils.py

from typing import List, Tuple

import torch


class Cache:
    """Name used in the type hints (`Optional[Cache]`, as in the HF code the
    adapters are copied from); `DynamicCache` below is the only implementation."""


class DynamicCache(Cache):
    """KV cache whose Keys are split into prompt / visual / text spans.

    Split mode (channel pruning on): `store_pruned` keeps the pruned visual
    Keys apart from the full-width prompt and text Keys; `update` appends each
    decode step's Keys to the text span. Per-method decode state (ThinK mask,
    SparK mask + mean, RotateK basis + δμ) is kept in the parallel lists below.

    Unified mode (channel_ratio == 0): `store_unified` / `update_unified` keep
    one K and one V tensor per layer, as a standard HF cache does.
    """

    def __init__(self) -> None:
        self.key_cache_pruned: List[torch.Tensor] = []
        self.key_cache_prompt: List[torch.Tensor] = []
        self.key_cache_text: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0  # HF `generate` sizes the attention mask from this
        self.mask = []
        self.key_cache_unified: List[torch.Tensor] = []
        self.is_unified: List[bool] = []
        # ThinK: per-head bool keep mask [B, H, D]; decode zero-fills pruned channels.
        self.think_mask: List[torch.Tensor] = []
        # SparK: per-token bool keep mask [B, H, S, D] + per-token mean of the
        # pruned channels [B, H, S, 1], used as their fill value at decode.
        self.spark_mask: List[torch.Tensor] = []
        self.spark_pruned_mean: List[torch.Tensor] = []
        # RotateK: R_partial [B, H, D, D_keep] and δμ [B, H, D] per layer.
        self.rotatek_rotations: List[torch.Tensor] = []
        self.rotatek_means: List[torch.Tensor] = []

    def __len__(self):
        return len(self.key_cache_text)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        """Decode-time append (split mode): new Keys join the text span."""
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        self.key_cache_text[layer_idx] = torch.cat([self.key_cache_text[layer_idx], key_states], dim=-2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return (self.key_cache_text[layer_idx], self.value_cache[layer_idx],
                self.key_cache_pruned[layer_idx], self.key_cache_prompt[layer_idx], self.mask[layer_idx])

    def store_unified(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int) -> None:
        """Prefill-time write for channel_ratio == 0 (no split).

        `.contiguous()` stops a slice view from pinning its parent tensor in
        memory across all layers; on an already-contiguous tensor it is a no-op.
        """
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        self.key_cache_unified.append(key_states)
        self.is_unified.append(True)
        self.key_cache_pruned.append(None)
        self.key_cache_prompt.append(None)
        # an empty (not None) text slot so len(self) counts this layer
        self.key_cache_text.append(key_states[:, :, :0, :])
        self.value_cache.append(value_states)
        self.mask.append(None)

    def update_unified(self, key_states: torch.Tensor, value_states: torch.Tensor,
                       layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode-time append for the unified path."""
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        self.key_cache_unified[layer_idx] = torch.cat([self.key_cache_unified[layer_idx], key_states], dim=-2)
        self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        return self.key_cache_unified[layer_idx], self.value_cache[layer_idx]

    def store_pruned(self, key_states_pruned, key_states_prompt, key_states_text, mask,
                     value_states, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prefill-time write (split mode). The prompt/text slices are views of
        the full prefill Keys; `.contiguous()` copies them so the parent can be
        freed instead of being held by every layer."""
        key_states_pruned = key_states_pruned.contiguous()
        key_states_prompt = key_states_prompt.contiguous()
        key_states_text = key_states_text.contiguous()
        value_states = value_states.contiguous()
        if layer_idx == 0:
            self._seen_tokens += key_states_pruned.shape[-2] + key_states_prompt.shape[-2] + key_states_text.shape[-2]
        self.key_cache_pruned.append(key_states_pruned)
        self.key_cache_prompt.append(key_states_prompt)
        self.key_cache_text.append(key_states_text)
        self.mask.append(mask)
        self.value_cache.append(value_states)
        self.key_cache_unified.append(None)
        self.is_unified.append(False)
        return self.key_cache_text[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if len(self.key_cache_text) <= layer_idx:
            return 0
        if self.is_unified[layer_idx]:
            return self.key_cache_unified[layer_idx].shape[-2]
        return (self.key_cache_prompt[layer_idx].shape[-2] + self.key_cache_pruned[layer_idx].shape[-2]
                + self.key_cache_text[layer_idx].shape[-2])
