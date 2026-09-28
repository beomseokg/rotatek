"""Qwen2.5-VL + FastV inplace token pruning evaluation.

Mirrors `fastv_llava_next.py` but for Qwen2.5-VL-7B-Instruct via the
`qwen2_5_vl_fastv` lmms-eval model. FastV-specific knobs:
  * `fast_v_agg_layer` (K)         — layer at which token pruning kicks in
  * `fast_v_keep_ratio`            — fraction of vision tokens kept at K
  * `fast_v_inplace`               — slice hidden_states (KV memory ↓) vs mask-only
  * `channel_ratio` + `channel_method` — ThinK / SparK / RotateK channel
                                          pruning on top of FastV

Vision-span bounds are detected per forward inside the patched
Qwen2_5_VLFastVModel — Qwen does not expand image-token placeholders
post-merge, so bounds are read directly off `input_ids`.
"""

import gc
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["MODEL_VERSION"] = "gpt-4o-mini"

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import torch
torch.set_num_threads(8)

import lmms_eval
from lmms_eval import models
from lmms_eval.evaluator import simple_evaluate
from lmms_eval.tasks import TaskManager

PRETRAINED = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_TAG = "qwen25_vl"


def _clear_state():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_evaluate(
    log_dir,
    dataset,
    method="think",
    log=False,
    fast_v_agg_layer=2,
    fast_v_inplace=True,
    use_cache=True,
    channel_ratios_override=None,
    fast_v_keep_ratios_override=None,
    limit=None,
    log_samples=False,
):
    method = str(method).strip().lower()
    valid_methods = {"think", "spark", "rotatek"}
    if method not in valid_methods:
        raise ValueError(f"Unsupported method={method}. Use one of: {', '.join(valid_methods)}")

    _clear_state()

    # GPT-eval style image benchmarks (long free-form responses scored by
    # an LLM judge). log_samples is auto-enabled for these.
    gpt_eval_datasets = {"llava_in_the_wild", "mmvet", "llava_bench_coco", "vibe_eval", "dc100_en"}

    if "mmstar" in dataset:
        # Single config — channel pruning off, FastV at default keep ratio.
        fast_v_keep_ratio = "0.25"
        channel_ratio = "0.0"

        model_name = "qwen2_5_vl_fastv"
        model_args = (
            f"pretrained={PRETRAINED},"
            f"attn_implementation=eager,"
            f"use_fast_v=True,"
            f"fast_v_agg_layer={int(fast_v_agg_layer)},"
            f"fast_v_keep_ratio={fast_v_keep_ratio},"
            f"fast_v_inplace={bool(fast_v_inplace)},"
            f"channel_ratio={channel_ratio},"
            f"channel_method={method},"
            f"use_cache={bool(use_cache)}"
        )

        LM = models.get_model(model_name, force_simple=False)
        lm = LM.create_from_arg_string(model_args, {"batch_size": 1, "max_batch_size": None, "device": "cuda:0"})
        task_manager = TaskManager()

        results = simple_evaluate(
            model=lm,
            tasks=[dataset],
            num_fewshot=0,
            task_manager=task_manager,
            batch_size=1,
        )

        print("results: ", results["results"][dataset], flush=True)
        del results, lm, task_manager
        _clear_state()
    else:
        channel_ratios = channel_ratios_override or ["0.625", "0.750"]
        # Pass `fast_v_keep_ratios_override=["1.00"]` from __main__ to keep
        # all vision tokens (FastV no-op); useful for baselines that isolate
        # the channel-pruning effect. `channel_ratios_override=["0.000"]`
        # symmetrically isolates the FastV-only effect.
        fast_v_keep_ratios = fast_v_keep_ratios_override or ["0.40", "0.30", "0.20"]
        for fast_v_keep_ratio in fast_v_keep_ratios:
            for channel_ratio in channel_ratios:
                model_name = "qwen2_5_vl_fastv"
                model_args = (
                    f"pretrained={PRETRAINED},"
                    f"attn_implementation=eager,"
                    f"use_fast_v=True,"
                    f"fast_v_agg_layer={int(fast_v_agg_layer)},"
                    f"fast_v_keep_ratio={fast_v_keep_ratio},"
                    f"fast_v_inplace={bool(fast_v_inplace)},"
                    f"channel_ratio={channel_ratio},"
                    f"channel_method={method},"
                    f"use_cache={bool(use_cache)}"
                )

                if log:
                    log_path = os.path.join(
                        log_dir,
                        f"{MODEL_TAG}_{dataset}_eval_fastv_k_{fast_v_agg_layer}_keep_{fast_v_keep_ratio}_inplace_{fast_v_inplace}_channel_ratio_{channel_ratio}_method_{method}.txt",
                    )
                    sys.stdout = open(log_path, "w")

                LM = models.get_model(model_name, force_simple=False)
                lm = LM.create_from_arg_string(model_args, {"batch_size": 1, "max_batch_size": None, "device": "cuda:0"})

                task_manager = TaskManager()

                # Short-answer VQA tasks (1-5 word answers). Cap max_new_tokens
                # so a degeneration loop on a hard sample (FastV+Qwen sometimes
                # emits a token like " addCriterion" repeatedly) wastes at most
                # 32 tokens of compute instead of the wrapper default 1024.
                # Matches infovqa/docvqa task config defaults.
                short_answer_gen_kwargs = "max_new_tokens=32"

                if dataset in [
                    "textvqa_val_lite", "infovqa_val_lite", "chartqa_lite",
                    "docvqa_val_lite", "vqav2_val_lite", "gqa_lite",
                    "vizwiz_vqa_val_lite", "ok_vqa_val2014_lite", "ai2d_lite",
                ]:
                    results = simple_evaluate(
                        model=lm,
                        tasks=[dataset],
                        num_fewshot=0,
                        task_manager=task_manager,
                        batch_size=1,
                        gen_kwargs=short_answer_gen_kwargs,
                        limit=limit,
                    )
                elif dataset in [
                    "textvqa_val", "infovqa_val", "docvqa_val",
                    "chartqa", "vizwiz_vqa_val",
                ]:
                    results = simple_evaluate(
                        model=lm,
                        tasks=[dataset],
                        num_fewshot=0,
                        task_manager=task_manager,
                        batch_size=1,
                        gen_kwargs=short_answer_gen_kwargs,
                        limit=limit,
                    )
                elif dataset in [
                    "coco2017_cap_val", "textcaps_val", "nocaps_val",
                ]:
                    # Caption tasks: yaml sets max_new_tokens=64, don't
                    # override with the short-answer cap (would truncate).
                    results = simple_evaluate(
                        model=lm,
                        tasks=[dataset],
                        num_fewshot=0,
                        task_manager=task_manager,
                        batch_size=1,
                        limit=limit,
                    )
                elif dataset in gpt_eval_datasets:
                    # GPT-eval benchmarks: long free-form responses, yaml
                    # sets max_new_tokens (1024-4096). Don't override with
                    # short-answer cap. log_samples saves per-sample dumps.
                    results = simple_evaluate(
                        model=lm,
                        tasks=[dataset],
                        num_fewshot=0,
                        task_manager=task_manager,
                        batch_size=1,
                        limit=limit,
                        log_samples=log_samples,
                    )
                else:
                    raise ValueError(f"Unsupported dataset: {dataset}")

                print("results: ", results["results"][dataset], flush=True)

                if log_samples and "samples" in results and dataset in results["samples"]:
                    import json
                    samples_path = os.path.join(
                        log_dir,
                        f"{MODEL_TAG}_{dataset}_eval_fastv_k_{fast_v_agg_layer}_keep_{fast_v_keep_ratio}_inplace_{fast_v_inplace}_channel_ratio_{channel_ratio}_method_{method}_samples.jsonl",
                    )
                    with open(samples_path, "w") as f:
                        for s in results["samples"][dataset]:
                            f.write(json.dumps(s, default=str) + "\n")
                    print(f"samples dumped to {samples_path}", flush=True)

                del results, lm, task_manager
                _clear_state()

                if log:
                    sys.stdout.close()


if __name__ == "__main__":
    log_dir = "./results/Qwen2.5_VL_7B/ablation/query-agnostic"
    os.makedirs(log_dir, exist_ok=True)

    # Channel pruning sweep on top of FastV (K=2). For each dataset run
    # all three channel-pruning methods (think / spark / rotatek) over the
    # full keep_ratio × channel_ratio grid defined in run_evaluate:
    #   fast_v_keep_ratio ∈ {0.50, 0.40, 0.30}
    #   channel_ratio     ∈ {0.625, 0.750}
    # channel_ratio=0.0 = FastV alone; >0 = FastV + channel pruning.
    datasets = [
        # "textvqa_val",
        # "coco2017_cap_val",
        # "textcaps_val",
        # "nocaps_val",
        # "llava_in_the_wild",
        # "mmvet",
        "textvqa_val_lite",
        "infovqa_val_lite",
        "chartqa_lite",
        "docvqa_val_lite",
    ]
    methods = ["rotatek"]

    # Channel-only baseline: keep_ratio=1.0 -> FastV keeps all tokens (no-op),
    # channel pruning sweep only. Isolates channel-pruning effect from token
    # pruning so the full sweep below can be compared against it.

    for dataset in datasets:
        for method in methods:
            run_evaluate(
                log_dir,
                dataset=dataset,
                method="rotatek",
                log=True,
                log_samples=True,
                fast_v_keep_ratios_override=["0.30"],
                channel_ratios_override=["0.750"],
            )