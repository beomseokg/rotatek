# ------------------------------------------------------------------------
# LlavaNextForConditionalGeneration with VisionZip token pruning + LLaMA
# channel pruning. Uses HF's LlavaNext as the host and swaps in our
# `LlamaVisionZipForCausalLM` for the language model so the same VisionZip
# CLI knobs that work for Qwen2.5-VL transfer to LLaVA-NeXT.
#
# Token-pruning strategy (faithful port of original VisionZip + LLaVA-NeXT
# anyres adaptation, per the authors' guidance:
# https://github.com/JIA-Lab-research/VisionZip/issues — "apply VisionZip
# to each part directly, similar to LLaVA-1.5"):
#   * CLIP encoder is monkey-patched (visionzip_clip.py) so the penultimate
#     layer stashes a `metric` (raw_key_states.mean(1)) for similarity merge.
#   * In `get_image_features`, every sub-image (1 base + N anyres tiles) is
#     pruned uniformly via dominant (CLS top-k) + contextual (similarity-
#     merge) selection — identical algorithm per sub-image.
#   * Contextual tokens are aggregated in-place at their original spatial
#     positions (Qwen-style mapping via target_orig_pos) so the LM sees a
#     spatially-coherent pruned grid.
#   * `pack_image_features` uses `restore_image_features_sorted` to scatter
#     each anyres tile's kept tokens back into a 24x24 grid, then
#     spatial-reshape + unpad + image_newline per row as usual.
# ------------------------------------------------------------------------

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers.models.llava_next.modeling_llava_next import (
    LlavaNextForConditionalGeneration,
    LlavaNextConfig,
    LlavaNextCausalLMOutputWithPast,
    image_size_to_num_patches,
)
from transformers.models.auto import AutoModel

from lmms_eval.models.model_utils.llava_next.llama_visionzip import (
    LlamaVisionZipForCausalLM,
    _stamp_channel_defaults,
    DEFAULT_CALIBRATION_RESULT_ROOT,
)

__all__ = [
    "LlavaNextVisionZipForConditionalGeneration",
    "DEFAULT_CALIBRATION_RESULT_ROOT",
]


class LlavaNextVisionZipForConditionalGeneration(LlavaNextForConditionalGeneration):
    """HF LlavaNext wrapper with VisionZip base-patch token pruning.

    Uses `LlamaVisionZipForCausalLM` for the language model; CLIP vision
    tower is the vanilla HF one (no monkey patching) so the anyres geometry
    stays intact.
    """

    def __init__(self, config: LlavaNextConfig):
        # Stamp the LLaMA-side knobs onto text_config so the VisionZip forward
        # sees them regardless of whether they were passed via kwargs.
        _stamp_channel_defaults(config.text_config)

        # Mirror super().__init__ but swap AutoModelForCausalLM with ours.
        super(LlavaNextForConditionalGeneration, self).__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        from transformers.models.llava_next.modeling_llava_next import LlavaNextMultiModalProjector
        self.multi_modal_projector = LlavaNextMultiModalProjector(config)
        embed_std = 1 / math.sqrt(config.text_config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(config.text_config.hidden_size, dtype=self.dtype) * embed_std)

        self.vocab_size = config.text_config.vocab_size
        self.language_model = LlamaVisionZipForCausalLM(config.text_config)
        if self.language_model._tied_weights_keys is not None:
            self._tied_weights_keys = [f"language_model.{k}" for k in self.language_model._tied_weights_keys]

        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self._padding_side = "left"
        self.post_init()

        # ------------------------------------------------------------------
        # Wire CLIP encoder monkey-patches: penultimate layer stashes `metric`
        # (raw_key_states.mean(1)) and the patched attention returns it as
        # the third tuple element. apply_info() promotes the vision tower
        # class so each forward primes the per-layer r schedule that decides
        # which layer stashes metric.
        # ------------------------------------------------------------------
        from .visionzip_clip import patch_clip_vision_tower, apply_info
        patch_clip_vision_tower(self.vision_tower)
        apply_info(self.vision_tower)

    # ------------------------------------------------------------------
    # restore_image_features_sorted — port of
    # `JIA-Lab-research/VisionZip/visionzip/llava_arch.py` extended for
    # LLaVA-NeXT's anyres needs (unpad_image + image_newline row separator).
    # Without the unpad+newline pair LLaVA-NeXT can't read the spatial layout
    # — every dominant_ratio comes out catastrophically low.
    #
    # Output structure: row-major spatial sequence of kept tokens (dominant +
    # in-place contextual aggregations) with image_newline appended at each
    # row boundary after unpad. CLS is dropped here (handled separately by
    # `pack_image_features` for the base patch).
    # ------------------------------------------------------------------
    def restore_image_features_sorted(
        self, image_feature, cur_keep_idx, width, height,
        image_size, image_newline,
    ):
        from transformers.models.llava_next.modeling_llava_next import unpad_image  # noqa: F401  (parity reference)

        num_img, total_patches, feature_dim = image_feature.shape
        num_keep = cur_keep_idx.shape[1]              # 1 (CLS) + dom + ctx

        # Drop the CLS index (=0) from keep_idx and shift -1 so the indices
        # address the 576-position spatial grid (no-CLS).
        cur_keep_idx_sorted, _ = cur_keep_idx.sort(dim=1)
        cur_keep_idx_sorted_restore = cur_keep_idx_sorted[:, 1:] - 1   # [num_img, dom + ctx]

        # Scatter kept tokens (dom + ctx) into the spatial 24x24 grid per tile.
        spatial_L = (
            self.config.vision_config.image_size // self.config.vision_config.patch_size
        ) ** 2
        side = int(spatial_L ** 0.5)

        restored_features = torch.zeros(
            (num_img, spatial_L, feature_dim),
            device=image_feature.device, dtype=image_feature.dtype,
        )
        mask = torch.zeros(num_img, spatial_L, dtype=torch.bool, device=image_feature.device)
        mask.scatter_(1, cur_keep_idx_sorted_restore, True)
        kept_features = image_feature[:, 1:num_keep, :]                # drop CLS at 0
        restored_features[mask] = kept_features.reshape(-1, feature_dim)

        # Spatial reshape into full anyres grid in HF format [D, H*side, W*side].
        assert width * height == num_img, (
            f"width*height={width*height} must equal num_img={num_img}"
        )
        restored_features = restored_features.view(height, width, side, side, feature_dim)
        restored_features = restored_features.permute(4, 0, 2, 1, 3).contiguous()
        restored_features = restored_features.flatten(1, 2).flatten(2, 3)  # [D, H*side, W*side]

        mask_grid = mask.view(height, width, side, side)
        mask_grid = mask_grid.permute(0, 2, 1, 3).contiguous()
        mask_grid = mask_grid.flatten(0, 1).flatten(1, 2)              # [H*side, W*side]

        # ---- unpad_image equivalent (applied to BOTH features and mask) ----
        # Replicate transformers.models.llava_next.modeling_llava_next.unpad_image
        # but parameterised so we can apply the same crop to the boolean mask.
        if isinstance(image_size, torch.Tensor):
            sz = image_size.tolist() if image_size.dim() > 0 else [image_size.item()]
            original_h, original_w = int(sz[0]), int(sz[1])
        else:
            original_h, original_w = int(image_size[0]), int(image_size[1])

        current_h = restored_features.shape[1]
        current_w = restored_features.shape[2]
        orig_ar = original_w / original_h
        curr_ar = current_w / current_h

        if orig_ar > curr_ar:
            scale = current_w / original_w
            new_h = int(round(original_h * scale, 7))
            pad = (current_h - new_h) // 2
            if pad > 0:
                restored_features = restored_features[:, pad : current_h - pad, :]
                mask_grid = mask_grid[pad : current_h - pad, :]
        else:
            scale = current_h / original_h
            new_w = int(round(original_w * scale, 7))
            pad = (current_w - new_w) // 2
            if pad > 0:
                restored_features = restored_features[:, :, pad : current_w - pad]
                mask_grid = mask_grid[:, pad : current_w - pad]

        # ---- Append image_newline as last column (per-row separator) ----
        if image_newline is not None:
            newline_col = (
                image_newline[:, None, None]
                .expand(feature_dim, restored_features.shape[1], 1)
                .to(restored_features.device, restored_features.dtype)
            )
            restored_features = torch.cat([restored_features, newline_col], dim=-1)
            # The newline is a fixed learned vector — always include it in the
            # output regardless of pruning so row boundaries are preserved.
            newline_mask = torch.ones(
                mask_grid.shape[0], 1, dtype=torch.bool, device=mask_grid.device,
            )
            mask_grid = torch.cat([mask_grid, newline_mask], dim=-1)

        # Permute restored_features so the last dim is feature_dim, then mask-
        # select. Returns [num_kept_total, D] in row-major spatial order
        # (per-row kept tokens followed by that row's newline).
        restored_features = restored_features.permute(1, 2, 0).contiguous()
        image_feature_select = restored_features[mask_grid]            # [num_kept, D]
        return image_feature_select

    # ------------------------------------------------------------------
    # VisionZip: prune ALL sub-images (base + anyres tiles) uniformly.
    # Returns a list of (pruned_features, keep_idx) tuples — one per image.
    #   pruned_features: [N_subimages, 1 + dom + ctx, D_lm]  (CLS at index 0,
    #                    then dom + ctx in spatial-position-sorted order)
    #   keep_idx:        [N_subimages, 1 + dom + ctx]        (CLS index + dom
    #                    indices + contextual target_orig_pos, all sorted by
    #                    position into the 577-token sequence)
    # `pack_image_features` below detects this structure and dispatches.
    # ------------------------------------------------------------------
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: Union[int, List[int]],
        vision_feature_select_strategy: str,
    ):
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        if pixel_values.dim() == 5:
            _pixel_values_list = [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)]
            pixel_values = torch.cat(_pixel_values_list, dim=0)
        elif pixel_values.dim() != 4:
            raise ValueError(f"pixel_values of shape {pixel_values.shape}, expect to be of 4 or 5 dimensions")

        # Run patched CLIP forward — penultimate layer stashes `metric` on
        # `encoder.layers[-2].metric` (see visionzip_clip.py).
        image_features = self.vision_tower(
            pixel_values,
            output_hidden_states=True,
            output_attentions=True,
        )

        dom_ratio = float(getattr(self.config.text_config, "dominant_ratio", 0.0) or 0.0)
        ctx_ratio = float(getattr(self.config.text_config, "contextual_ratio", 0.0) or 0.0)
        # Match Qwen's `dominant_num >= N -> skip` semantic: dom_ratio >= 1.0
        # means "keep all tokens" -> fall through to vanilla path. Without this
        # the clamp `dom_num = min(dom_num, L_no_cls - ctx_num)` would still
        # carve out a contextual slot and modify one position.
        vz_active = (
            dom_ratio < 1.0
            and (dom_ratio > 0.0 or ctx_ratio > 0.0)
            and image_features.attentions is not None
        )

        # ---- VisionZip-disabled path: standard processing ----------------
        if not vz_active:
            if isinstance(vision_feature_layer, int):
                selected = image_features.hidden_states[vision_feature_layer]
            else:
                hs_pool = [image_features.hidden_states[layer_idx] for layer_idx in vision_feature_layer]
                selected = torch.cat(hs_pool, dim=-1)
            if vision_feature_select_strategy == "default":
                selected = selected[:, 1:]
            projected = self.multi_modal_projector(selected)
            return torch.split(projected, image_num_patches, dim=0)

        # ---- VisionZip-active path: prune all sub-images uniformly -------
        # Read penultimate-layer signals stashed by the patched CLIP.
        hidden_states_pre = image_features.hidden_states[-2]                        # [N_total, 577, D_vit]
        attn_weights = image_features.attentions[-2]                                # [N_total, heads, 577, 577]
        metric = self.vision_tower.vision_model.encoder.layers[-2].metric           # [N_total, 577, D_metric]

        L_with_cls = hidden_states_pre.shape[1]
        L_no_cls = L_with_cls - 1
        dom_num = max(1, int(dom_ratio * L_no_cls))
        ctx_num = max(1, int(ctx_ratio * L_no_cls))
        # clamp so dominant + contextual + CLS <= L_with_cls
        dom_num = min(dom_num, L_no_cls - ctx_num)

        cls_idx = 0
        cls_attention = attn_weights[:, :, cls_idx, cls_idx + 1:]                   # [N, heads, L_no_cls]
        cls_attention_sum = cls_attention.sum(dim=1)                                # [N, L_no_cls]
        topk_indices = cls_attention_sum.topk(dom_num, dim=1).indices + 1           # +1: CLS at index 0
        all_indices = torch.cat([
            torch.zeros(
                (hidden_states_pre.shape[0], 1),
                dtype=topk_indices.dtype, device=topk_indices.device,
            ),
            topk_indices,
        ], dim=1)                                                                   # [N, 1 + dom]

        # Mask: True for non-kept positions (i.e., positions to be dropped or merged).
        mask = torch.ones_like(hidden_states_pre[:, :, 0], dtype=torch.bool)
        mask = mask.scatter_(1, all_indices, False)
        dominant_tokens = hidden_states_pre.masked_select(~mask.unsqueeze(-1)).view(
            hidden_states_pre.shape[0], dom_num + 1, hidden_states_pre.shape[-1]
        )                                                                           # [N, 1 + dom, D_vit]

        # ---- Contextual: similarity-merge the remaining (non-dominant) tokens ----
        metric_filtered = metric[mask].view(
            metric.shape[0], L_with_cls - (dom_num + 1), metric.shape[-1]
        )
        hidden_filtered = hidden_states_pre.masked_select(mask.unsqueeze(-1)).view(
            hidden_states_pre.shape[0], L_with_cls - (dom_num + 1), hidden_states_pre.shape[-1]
        )
        metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        step = max(1, metric_normalized.shape[1] // ctx_num)
        target_indices = torch.arange(
            0, metric_normalized.shape[1], step, device=metric_normalized.device,
        )[:ctx_num]
        target_tokens = metric_normalized[:, target_indices, :]

        non_target_mask = ~torch.isin(
            torch.arange(metric_normalized.shape[1], device=metric_normalized.device),
            target_indices,
        )
        tokens_to_merge = metric_normalized[:, non_target_mask, :]
        similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
        assign_one_hot = torch.zeros(
            tokens_to_merge.shape[0], tokens_to_merge.shape[1], target_indices.shape[0],
            dtype=hidden_filtered.dtype, device=metric_normalized.device,
        )
        assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
        counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
        hidden_to_merge = hidden_filtered[:, non_target_mask, :]
        aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
        target_hidden = hidden_filtered[:, target_indices, :]
        contextual_tokens = target_hidden + aggregated_hidden                       # [N, ctx, D_vit]

        # ---- Map target_indices (filtered space) back to ORIGINAL spatial positions ----
        # `mask` is True at non-dominant positions in the original 0..L_with_cls-1
        # space. For each tile, the `i`-th True position is the i-th filtered
        # position. target_indices indexes into that filtered list.
        N_total = mask.shape[0]
        # arange[L_with_cls] expanded per batch, masked-select gives per-row True
        # positions in ascending order (n_filtered per row).
        arange_l = (
            torch.arange(L_with_cls, device=mask.device)
            .unsqueeze(0)
            .expand(N_total, -1)
        )
        non_dom_orig_pos = arange_l[mask].view(N_total, -1)                         # [N, n_filtered]
        target_orig_pos = non_dom_orig_pos[:, target_indices]                       # [N, ctx_num]

        # ---- Build combined kept positions + features in spatial-sorted order ----
        # Sort all_indices (currently in [CLS=0, topk_in_attention_order]) so it
        # matches the position-sorted output of masked_select that produced
        # `dominant_tokens`.
        all_dom_indices_sorted, _ = all_indices.sort(dim=1)                         # [N, 1 + dom]

        combined_pos_unsorted = torch.cat(
            [all_dom_indices_sorted, target_orig_pos], dim=1
        )                                                                           # [N, 1 + dom + ctx]
        combined_feat_unsorted = torch.cat(
            [dominant_tokens, contextual_tokens], dim=1
        )                                                                           # [N, 1 + dom + ctx, D_vit]

        # Sort by spatial position so restore() scatters consistently and the
        # LM sees image features in the same row-major order as the unpruned
        # baseline (just sparser).
        sorted_pos, sort_perm = combined_pos_unsorted.sort(dim=1)                   # [N, 1+dom+ctx]
        D_vit = combined_feat_unsorted.shape[-1]
        sorted_feat = torch.gather(
            combined_feat_unsorted,
            dim=1,
            index=sort_perm.unsqueeze(-1).expand(-1, -1, D_vit),
        )                                                                           # [N, 1+dom+ctx, D_vit]

        # Project to LM dim. (multi_modal_projector expects [N, L, D_vit].)
        pruned_projected = self.multi_modal_projector(sorted_feat)                  # [N, 1+dom+ctx, D_lm]

        # Split per-image and pair with the spatial-sorted keep indices.
        pruned_per_image = torch.split(pruned_projected, image_num_patches, dim=0)
        keep_idx_per_image = torch.split(sorted_pos, image_num_patches, dim=0)

        return list(zip(pruned_per_image, keep_idx_per_image))

    # ------------------------------------------------------------------
    # pack_image_features dispatches on element type:
    #   * (features, keep_idx) tuple  -> VisionZip-active path; uses
    #     restore_image_features_sorted for anyres tiles.
    #   * plain Tensor [N_patches, L, D]  -> standard LLaVA-NeXT path
    #     (VisionZip-disabled or trivial single-patch images).
    # ------------------------------------------------------------------
    def pack_image_features(self, image_features, image_sizes, vision_feature_select_strategy, image_newline=None):
        from transformers.models.llava_next.modeling_llava_next import (
            get_anyres_image_grid_shape, unpad_image,
        )
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if isinstance(image_feature, tuple):
                # ----- VisionZip-active path -----
                # image_feature = (pruned_features, keep_idx), both in
                # spatial-position-sorted order (CLS at 0, then dom + ctx
                # at their original 577-space positions).
                #   pruned_features: [N_subimages, 1 + dom + ctx, D_lm]
                #   keep_idx       : [N_subimages, 1 + dom + ctx]
                pruned_features, keep_idx = image_feature
                if pruned_features.shape[0] > 1:
                    # base: drop CLS, keep dom + ctx tokens flat (HF doesn't
                    # spatially reshape the base patch).
                    base_image_feature = pruned_features[0, 1:, :]                  # [dom + ctx, D_lm]

                    # anyres tiles get spatial restore: scatter kept tokens
                    # onto the 24x24 grid using keep_idx, then unpad + append
                    # image_newline at each row boundary.
                    anyres_feature = pruned_features[1:]                            # [N-1, 1+dom+ctx, D_lm]
                    anyres_keep_idx = keep_idx[1:]                                  # [N-1, 1+dom+ctx]

                    num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                        image_sizes[image_idx],
                        self.config.image_grid_pinpoints,
                        self.config.vision_config.image_size,
                    )

                    image_feature = self.restore_image_features_sorted(
                        anyres_feature, anyres_keep_idx,
                        width=num_patch_width, height=num_patch_height,
                        image_size=image_sizes[image_idx],
                        image_newline=image_newline,
                    )
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    # Single sub-image (no anyres) — just drop CLS and use flat.
                    image_feature = pruned_features[0, 1:, :]
                    if image_newline is not None:
                        image_feature = torch.cat(
                            (image_feature, image_newline[None].to(image_feature)),
                            dim=0,
                        )
                new_image_features.append(image_feature)
                feature_lens.append(image_feature.size(0))
                continue

            # ----- VisionZip-disabled path: original LLaVA-NeXT logic -----
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                anyres_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )

                anyres_feature = anyres_feature.view(num_patch_height, num_patch_width, height, width, -1)
                anyres_feature = anyres_feature.permute(4, 0, 2, 1, 3).contiguous()
                anyres_feature = anyres_feature.flatten(1, 2).flatten(2, 3)
                anyres_feature = unpad_image(anyres_feature, image_sizes[image_idx])
                if image_newline is not None:
                    anyres_feature = torch.cat(
                        (
                            anyres_feature,
                            image_newline[:, None, None]
                            .expand(*anyres_feature.shape[:-1], 1)
                            .to(anyres_feature.device, anyres_feature.dtype),
                        ),
                        dim=-1,
                    )
                anyres_feature = anyres_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, anyres_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        image_features = torch.cat(new_image_features, dim=0)
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features.device)
        return image_features, feature_lens

    # ------------------------------------------------------------------
    # Override forward to tell the language model the prompt/query spans
    # (needed by kv_pruning_utils to partition prompt vs vision vs text).
    #
    # IMPORTANT: the signature MUST match HF's LlavaNextForConditionalGeneration
    # — particularly `logits_to_keep` — or `GenerationMixin._supports_logits_to_keep`
    # (which inspects `forward.__signature__`) returns False, `logits_to_keep=1`
    # never gets set, and every decode step returns full-sequence logits → the
    # downstream `torch.cat([input_ids, next_tokens[:, None]])` blows up with a
    # "got 2 and 3" dim mismatch.
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_sizes=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        logits_to_keep=0,
        **lm_kwargs,
    ):
        if input_ids is not None:
            image_tok = self.config.image_token_index
            image_mask_top = (input_ids == image_tok)
            if image_mask_top.any():
                # Batch[0] only — lmms-eval calls with batch_size=1.
                nz = image_mask_top[0].nonzero(as_tuple=True)[0]
                if nz.numel() > 0:
                    self.config.text_config.prompt_seqlen = int(nz[0].item())
                    self.config.text_config.query_seqlen = int(input_ids.shape[1] - nz[-1].item() - 1)

        # ------------------------------------------------------------------
        # VisionZip pre-pass (prefill only).
        #
        # We compute pruned image_features ourselves, build inputs_embeds with
        # the features substituted at the first n_features image_token slots,
        # then ACTUALLY SLICE OFF the unused placeholder positions. Sequence
        # length the LM sees becomes the pruned length (= cache length after
        # prefill).
        #
        # `generate()` tracks its own attention_mask/input_ids at the original
        # length though, so at decode the lengths mismatch. We handle that
        # in `prepare_inputs_for_generation` (override below) — at decode it
        # rewrites attention_mask/position_ids to match cache_len + 1.
        # ------------------------------------------------------------------
        do_visionzip_prefill = (
            pixel_values is not None
            and getattr(pixel_values, "size", None) is not None
            and pixel_values.size(0) > 0
            and inputs_embeds is None
            and input_ids is not None
        )
        if do_visionzip_prefill:
            v_layer = (
                vision_feature_layer
                if vision_feature_layer is not None
                else self.config.vision_feature_layer
            )
            v_strategy = (
                vision_feature_select_strategy
                if vision_feature_select_strategy is not None
                else self.config.vision_feature_select_strategy
            )

            image_features = self.get_image_features(
                pixel_values, image_sizes,
                vision_feature_layer=v_layer,
                vision_feature_select_strategy=v_strategy,
            )
            image_features_packed, feature_lens = self.pack_image_features(
                image_features, image_sizes,
                vision_feature_select_strategy=v_strategy,
                image_newline=self.image_newline,
            )

            image_tok_id = self.config.image_token_index
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            # Per-batch: substitute features at first n_features placeholder
            # positions, then slice off unused placeholders entirely.
            new_embeds_list = []
            new_attn_list = []
            new_labels_list = []
            feat_offset = 0
            for b in range(input_ids.shape[0]):
                placeholder_pos = (input_ids[b] == image_tok_id).nonzero(as_tuple=True)[0]
                n_placeholders = placeholder_pos.numel()
                n_features_raw = int(
                    feature_lens[b].item() if feature_lens.dim() > 0 else feature_lens.item()
                )
                feats_b_raw = image_features_packed[feat_offset:feat_offset + n_features_raw].to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                feat_offset += n_features_raw

                # Cap features at placeholder count. At light pruning (high
                # dominant_ratio) our restore output (spatial dom + newlines +
                # per-tile CLS + contextual) exceeds the original placeholder
                # count because per-tile CLS / contextual are *appended* tokens
                # that don't have placeholder slots in input_ids. Drop the
                # tail (CLS / contextual extras) when we overrun.
                n_features = min(n_features_raw, n_placeholders)
                feats_b = feats_b_raw[:n_features]

                # Substitute at the first n_features image positions
                embeds_b = inputs_embeds[b].clone()
                kept_image_pos = placeholder_pos[:n_features]
                embeds_b[kept_image_pos] = feats_b

                # Build keep_mask: drop excess placeholder positions only when
                # we under-shoot (heavy pruning). At light pruning we capped
                # features instead, so all placeholders stay.
                keep_mask = torch.ones_like(input_ids[b], dtype=torch.bool)
                if n_placeholders > n_features:
                    keep_mask[placeholder_pos[n_features:]] = False

                new_embeds_list.append(embeds_b[keep_mask])
                new_attn_list.append(attention_mask[b][keep_mask])
                if labels is not None:
                    new_labels_list.append(labels[b][keep_mask])

            # batch_size=1 fast path (lmms-eval). Multi-batch: right-pad.
            if len(new_embeds_list) == 1:
                inputs_embeds = new_embeds_list[0].unsqueeze(0)
                attention_mask = new_attn_list[0].unsqueeze(0)
                if new_labels_list:
                    labels = new_labels_list[0].unsqueeze(0)
            else:
                max_len = max(t.shape[0] for t in new_embeds_list)
                pad_id = self.pad_token_id if self.pad_token_id >= 0 else 0
                pad_embed = self.get_input_embeddings()(
                    torch.tensor([pad_id], device=inputs_embeds.device)
                )
                padded_embeds, padded_attn = [], []
                for emb, am in zip(new_embeds_list, new_attn_list):
                    pad_n = max_len - emb.shape[0]
                    if pad_n > 0:
                        emb = torch.cat([emb, pad_embed.expand(pad_n, -1)], dim=0)
                        am = torch.cat([am, torch.zeros(pad_n, dtype=am.dtype, device=am.device)])
                    padded_embeds.append(emb)
                    padded_attn.append(am)
                inputs_embeds = torch.stack(padded_embeds, dim=0)
                attention_mask = torch.stack(padded_attn, dim=0)
                if new_labels_list:
                    padded_labels = []
                    for lb in new_labels_list:
                        pad_n = max_len - lb.shape[0]
                        if pad_n > 0:
                            lb = torch.cat(
                                [lb, torch.full((pad_n,), -100, dtype=lb.dtype, device=lb.device)]
                            )
                        padded_labels.append(lb)
                    labels = torch.stack(padded_labels, dim=0)

            # Skip super's image processing; sequence length is now the pruned
            # length. position_ids / cache_position fall back to auto-compute
            # based on inputs_embeds.shape[1].
            input_ids = None
            pixel_values = None
            position_ids = None
            cache_position = None

        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

    # ------------------------------------------------------------------
    # `prepare_inputs_for_generation` override.
    #
    # Our prefill slices the sequence to (text + pruned_image_tokens), so
    # the KV cache built during prefill has length = pruned_seq_len. But
    # `generate()` keeps its own attention_mask/input_ids tracker at the
    # ORIGINAL prompt length and extends them by 1 each decode step.
    #
    # At decode time the default `prepare_inputs_for_generation` would pass
    # an attention_mask of shape [B, original_len + decoded] — which doesn't
    # match cache_len + 1 = pruned_len + decoded + 1. Flash attention's
    # `_upad_input` then indexes out of bounds.
    #
    # Fix: after super's standard prep, rewrite attention_mask + position_ids
    # to match the actual cache length. For batch_size=1 with no padding
    # tokens this is just a fresh all-1s mask of the right shape.
    # ------------------------------------------------------------------
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        image_sizes=None,
        vision_feature_layer=None,
        vision_feature_select_strategy=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            **kwargs,
        )

        if past_key_values is None:
            return model_inputs
        cache_len = past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else 0
        if cache_len == 0:
            return model_inputs

        # ----- The decisive fix -----
        # generate()'s tracker (input_ids, attention_mask) is at the ORIGINAL
        # sequence length (before our prefill slice). Default super() prepares
        # input_ids via `input_ids[:, cache_position]` where cache_position
        # = [pruned_cache_len] — this points into the middle of the original
        # placeholder span, NOT the newly generated token. The LM then sees
        # an image_token_index instead of the latest decoded token and the
        # generation collapses into garbage.
        #
        # Force input_ids to the last decoded token; rebuild attention_mask
        # / position_ids / cache_position from cache_len so they all agree.
        if model_inputs.get("input_ids") is not None and input_ids is not None:
            model_inputs["input_ids"] = input_ids[:, -1:]
        ref = model_inputs.get("input_ids")
        if ref is None:
            ref = model_inputs.get("inputs_embeds")
        seq_len_in = ref.shape[1] if ref is not None else 1
        expected_len = cache_len + seq_len_in

        am = model_inputs.get("attention_mask")
        if am is None or am.shape[-1] != expected_len:
            model_inputs["attention_mask"] = torch.ones(
                ref.shape[0], expected_len,
                dtype=(am.dtype if am is not None else torch.long),
                device=ref.device,
            )

        model_inputs["position_ids"] = (
            torch.arange(cache_len, cache_len + seq_len_in,
                         dtype=torch.long, device=ref.device)
            .unsqueeze(0).expand(ref.shape[0], -1)
        )

        model_inputs["cache_position"] = torch.arange(
            cache_len, cache_len + seq_len_in,
            dtype=torch.long, device=ref.device,
        )

        return model_inputs
