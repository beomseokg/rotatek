"""LLaVA-NeXT (llama3-llava-next-8b) + VisionZip / channel pruning evaluation.

Mirrors the Qwen2.5-VL script at `visionzip_qwen.py` but uses the HF
LlavaNextForConditionalGeneration wrapper with VisionZip base-patch token
pruning and LLaMA channel pruning.
"""

import os
import sys

# Make the repo root importable so `rotatek` resolves no matter where this
# script is launched from (mirrors scripts/paper/latency/*).
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import gc

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

PRETRAINED = "llava-hf/llama3-llava-next-8b-hf"
MODEL_TAG = "llava_next"


def _clear_decode_kernel_state():
    try:
        from lmms_eval.models.model_utils.qwen.qwen2_5vl_visionzip import run_custom_decode_kernel
    except Exception:
        run_custom_decode_kernel = None

    if run_custom_decode_kernel is not None:
        for attr in (
            "_idx_cache",
            "_static_cache",
            "_sup_cache",
            "_kernel_cache",
            "_kernel_timings",
            "_compare_count",
        ):
            if hasattr(run_custom_decode_kernel, attr):
                delattr(run_custom_decode_kernel, attr)
        run_custom_decode_kernel._profile_kernel = False

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_evaluate(
    log_dir,
    dataset,
    method="think",
    channel_pruning=True,
    log=False,
    layer_adaptive_channel_budget=False,
    channel_reconstruction="off",
    reconstruction_constant=0.1,
    custom_kernel=False,
    decode_attention_backend=None,
    calibration_mode="off",
    calibration_task="channel_importance",
    exempt_layer_idx=None,
    channel_ratios_override=None,
    dominant_ratios_override=None,
    full_channel_first_n_layers=0,
    log_samples=False,
):
    method = str(method).strip().lower()
    valid_methods = {"think", "spark", "rotatek"}
    if method not in valid_methods:
        raise ValueError(f"Unsupported method={method}. Use one of: {', '.join(valid_methods)}")

    # RotateK's PCA subsample size — read from env var so a single eval
    # script can sweep it without re-importing modules. "full" / "0" /
    # "none" / "" disables subsampling (uses all vision tokens).
    rotatek_pca_k_sample = os.environ.get("ROTATEK_PCA_K_SAMPLE", "full").strip().lower()

    calibration_mode = str(calibration_mode).strip().lower()
    calibration_task = str(calibration_task).strip().lower()

    # RotateK's only calibration-backed path is `rotation_matrix`. If the
    # caller set `calibration_mode="use"` with rotatek but left the task at
    # the default, auto-pick rotation_matrix so the offline R gets loaded.
    # Symmetric convenience is also applied for "collect" mode.
    if method == "rotatek" and calibration_mode in ("use", "collect") and calibration_task == "channel_importance":
        calibration_task = "rotation_matrix"

    if decode_attention_backend is None:
        decode_attention_backend = "fa2"
    decode_attention_backend = str(decode_attention_backend).strip().lower()

    valid_calibration_modes = {"off", "collect", "use"}
    valid_calibration_tasks = {
        "channel_importance", "modality_score", "supplementary_matrix",
        "attention_shift", "attention_kl", "rotation_matrix", "all",
    }
    valid_decode_attention_backends = {"fa2", "triton"}

    if decode_attention_backend not in valid_decode_attention_backends:
        raise ValueError(
            f"Unsupported decode_attention_backend={decode_attention_backend}. Use one of: fa2, triton"
        )
    if calibration_mode not in valid_calibration_modes:
        raise ValueError(
            f"Unsupported calibration_mode={calibration_mode}. Use one of: off, collect, use"
        )
    if calibration_task not in valid_calibration_tasks:
        raise ValueError(
            f"Unsupported calibration_task={calibration_task}. Use one of: {sorted(valid_calibration_tasks)}"
        )

    _clear_decode_kernel_state()

    # GPT-eval style image benchmarks (long free-form responses scored by
    # an LLM judge). log_samples is auto-enabled for these.
    gpt_eval_datasets = {"llava_in_the_wild", "mmvet", "llava_bench_coco", "vibe_eval", "dc100_en"}

    if "mmstar" in dataset:
        channel_ratio = "0.00"
        dominant_ratio = "0.75"

        model_name = "llava_next_visionzip"
        model_args = (
            f"pretrained={PRETRAINED},"
            f"visionzip_dominant_ratio={dominant_ratio},"
            f"visionzip_contextual_ratio=0.00,"
            f"attn_implementation=flash_attention_2,"
            f"channel_ratio={channel_ratio},"
            f"channel_method={method},"
            f"layer_adaptive_channel_budget={layer_adaptive_channel_budget},"
            f"channel_reconstruction={channel_reconstruction},"
            f"reconstruction_constant={reconstruction_constant},"
            f"custom_kernel={custom_kernel},"
            f"decode_attention_backend={decode_attention_backend},"
            f"calibration_mode={calibration_mode},"
            f"offline_calibration_tasks={calibration_task}"
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
        _clear_decode_kernel_state()
    else:
        channel_ratios = channel_ratios_override or ["0.625", "0.750"]
        # Pass `dominant_ratios_override=["1.00"]` from __main__ to bypass token
        # pruning entirely (vz_active=False -> vanilla path); useful for
        # baselines that isolate the channel-pruning effect.
        dominant_ratios = dominant_ratios_override or ["0.40", "0.30", "0.20"]
        for dominant_ratio in dominant_ratios:
            for channel_ratio in channel_ratios:
                # VisionK custom kernel is retired; keep `actual_custom_kernel`
                # always False so legacy filename suffix stays consistent.
                actual_custom_kernel = False

                if channel_pruning:
                    model_name = "llava_next_visionzip"
                    model_args = (
                        f"pretrained={PRETRAINED},"
                        f"visionzip_dominant_ratio={dominant_ratio},"
                        f"visionzip_contextual_ratio=0.05,"
                        f"attn_implementation=flash_attention_2,"
                        f"channel_ratio={channel_ratio},"
                        f"channel_method={method},"
                        f"layer_adaptive_channel_budget={layer_adaptive_channel_budget},"
                        f"channel_reconstruction={channel_reconstruction},"
                        f"reconstruction_constant={reconstruction_constant},"
                        f"custom_kernel={actual_custom_kernel},"
                        f"decode_attention_backend={decode_attention_backend},"
                        f"calibration_mode={calibration_mode},"
                        f"offline_calibration_tasks={calibration_task}"
                    )
                    if exempt_layer_idx is not None:
                        model_args += f",exempt_layer_idx={int(exempt_layer_idx)}"
                    if full_channel_first_n_layers and int(full_channel_first_n_layers) > 0:
                        model_args += f",full_channel_first_n_layers={int(full_channel_first_n_layers)}"
                else:
                    raise NotImplementedError(
                        "channel_pruning=False path: add a plain LlavaNext model wrapper if needed."
                    )

                exempt_suffix = (
                    f"_exempt_layer_{int(exempt_layer_idx):02d}"
                    if exempt_layer_idx is not None
                    else ""
                )
                if full_channel_first_n_layers and int(full_channel_first_n_layers) > 0:
                    exempt_suffix += f"_first_{int(full_channel_first_n_layers)}_full"
                # RotateK PCA-subsample suffix: only include when method is
                # rotatek AND user actually set a non-default value, so legacy
                # (full) runs keep their existing filename.
                rotatek_pca_suffix = (
                    f"_pca_ksample_{rotatek_pca_k_sample}"
                    if method == "rotatek" and rotatek_pca_k_sample not in {"full", "0", "none", ""}
                    else ""
                )
                if log:
                    if calibration_mode == "off":
                        log_path = os.path.join(
                            log_dir,
                            f"{MODEL_TAG}_{dataset}_eval_dominant_ratio_{dominant_ratio}_contextual_ratio_0.05_channel_ratio_{channel_ratio}_method_{method}_kernel_{actual_custom_kernel}{exempt_suffix}{rotatek_pca_suffix}_query_aware.txt",
                        )
                    else:
                        log_path = os.path.join(
                            log_dir,
                            f"{MODEL_TAG}_{dataset}_eval_dominant_ratio_{dominant_ratio}_contextual_ratio_0.05_channel_ratio_{channel_ratio}_mode_{calibration_mode}_task_{calibration_task}_layer_adaptive_{layer_adaptive_channel_budget}_reconstruction_{channel_reconstruction}_method_{method}_kernel_{actual_custom_kernel}_triton_gqa{exempt_suffix}{rotatek_pca_suffix}.txt",
                        )

                    sys.stdout = open(log_path, "w")

                if method == "rotatek":
                    print(f"[rotatek] ROTATEK_PCA_K_SAMPLE = {rotatek_pca_k_sample}", flush=True)

                LM = models.get_model(model_name, force_simple=False)
                lm = LM.create_from_arg_string(model_args, {"batch_size": 1, "max_batch_size": None, "device": "cuda:0"})

                task_manager = TaskManager()

                eval_kwargs = {
                    "model": lm,
                    "tasks": [dataset],
                    "num_fewshot": 0,
                    "task_manager": task_manager,
                    "batch_size": 1,
                }
                _log_samples_eligible = dataset in gpt_eval_datasets
                if log_samples and _log_samples_eligible:
                    eval_kwargs["log_samples"] = True

                if dataset in [
                    "textvqa_val_lite", "infovqa_val_lite", "chartqa_lite",
                    "docvqa_val_lite", "vqav2_val_lite", "gqa_lite",
                    "vizwiz_vqa_val_lite", "ok_vqa_val2014_lite", "ai2d_lite",
                ]:
                    results = simple_evaluate(**eval_kwargs)
                elif dataset in [
                    "textvqa_val", "infovqa_val", "docvqa_val",
                    "chartqa", "vizwiz_vqa_val",
                    "coco2017_cap_val", "textcaps_val", "nocaps_val",
                ] or dataset in gpt_eval_datasets:
                    # limit=1000 if you want a quick sanity check
                    results = simple_evaluate(**eval_kwargs)
                else:
                    raise ValueError(f"Unsupported dataset: {dataset}")

                print("results: ", results["results"][dataset], flush=True)

                if (
                    log_samples
                    and _log_samples_eligible
                    and "samples" in results
                    and dataset in results["samples"]
                ):
                    import json
                    samples_path = os.path.join(
                        log_dir,
                        f"{MODEL_TAG}_{dataset}_eval_dominant_ratio_{dominant_ratio}_contextual_ratio_0.05_channel_ratio_{channel_ratio}_method_{method}_kernel_{actual_custom_kernel}{exempt_suffix}{rotatek_pca_suffix}_samples.jsonl",
                    )
                    with open(samples_path, "w") as f:
                        for s in results["samples"][dataset]:
                            f.write(json.dumps(s, default=str) + "\n")
                    print(f"samples dumped to {samples_path}", flush=True)

                del results, lm, task_manager
                _clear_decode_kernel_state()

                if log:
                    sys.stdout.close()


def run_exempt_layer_sweep(
    log_dir,
    dataset,
    channel_ratio,
    *,
    method="think",
    num_layers=32,
    log=True,
    layer_adaptive_channel_budget=False,
    channel_reconstruction="off",
    reconstruction_constant=0.1,
    custom_kernel=False,
    decode_attention_backend=None,
    calibration_mode="off",
    calibration_task="channel_importance",
):
    """For each decoder layer l ∈ [0, num_layers-1], run eval with layer l
    kept full-channel (channel_ratio=0) while the rest prune at `channel_ratio`.
    Produces one log file per exempted layer (suffix `_exempt_layer_NN.txt`).
    """
    for layer_idx in range(num_layers):
        print(f"\n{'='*60}", flush=True)
        print(f"Sweep: exempt_layer_idx={layer_idx}, channel_ratio={channel_ratio}", flush=True)
        print(f"{'='*60}", flush=True)
        run_evaluate(
            log_dir,
            dataset,
            method=method,
            channel_pruning=True,
            log=log,
            layer_adaptive_channel_budget=layer_adaptive_channel_budget,
            channel_reconstruction=channel_reconstruction,
            reconstruction_constant=reconstruction_constant,
            custom_kernel=custom_kernel,
            decode_attention_backend=decode_attention_backend,
            calibration_mode=calibration_mode,
            calibration_task=calibration_task,
            exempt_layer_idx=layer_idx,
            channel_ratios_override=[str(channel_ratio)],
        )


if __name__ == "__main__":
    log_dir = "./results/LLaVA_NeXT_Llama3_8B/ablation/query-agnostic"
    os.makedirs(log_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # ThinK / SparK / RotateK: all run with calibration_mode="off" by default.
    # Online (per-prefill) PCA is used for RotateK in this mode — no offline
    # rotation-matrix calibration is required (matches InternVL's setup).
    # If you want to use offline calibration for RotateK:
    #   1. Collect: run_evaluate(... method="rotatek", calibration_mode="collect")
    #      (auto-picks calibration_task="rotation_matrix")
    #   2. Use:     run_evaluate(... method="rotatek", calibration_mode="use")
    # ------------------------------------------------------------------

    """ full dataset list
    datasets = [
        "textvqa_val", "infovqa_val", "chartqa", 
        "vizwiz_vqa_val", "docvqa_val"
    ]
    """
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
    methods = ["think"]

    # Channel-only baseline: dom=1.0 -> vanilla token path (vz_active=False),
    # channel pruning sweep only. Isolates channel-pruning effect from token
    # pruning so the full sweep below can be compared against it.
    for dataset in datasets:
        for method in methods:
            run_evaluate(
                log_dir,
                dataset=dataset,
                method="rotatek",
                log=True,
                dominant_ratios_override=["0.30"],
                channel_ratios_override=["0.750"],
                log_samples=True,
            )
