"""Accuracy of token pruning (VisionZip / FastV) combined with Key channel pruning.

    python scripts/paper/accuracy/eval_accuracy.py --model qwen --pruner visionzip \\
        --method rotatek --token_ratio 0.40 --channel_ratio 0.75 \\
        --tasks textvqa_val,infovqa_val,chartqa,docvqa_val,vizwiz_vqa_val

--token_ratio    VisionZip dominant ratio (contextual ratio fixed at 0.05), or
                 the FastV keep ratio.
--channel_ratio  fraction of visual Key channels PRUNED. 0 = token pruning only;
                 0.75 keeps 25% of the channels (the paper's default).

The unpruned "Baseline" row is --token_ratio 1.0 --channel_ratio 0.
"""
import argparse
import gc
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MODEL_VERSION", "gpt-4o-mini")  # judge for llava_in_the_wild / mmvet
# torch can detect hundreds of cores in containers, which slows image
# preprocessing by 30-50x; must be set before torch is imported.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import torch
from lmms_eval import models
from lmms_eval.evaluator import simple_evaluate
from lmms_eval.tasks import TaskManager

torch.set_num_threads(8)

PRETRAINED = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "llava": "llava-hf/llama3-llava-next-8b-hf",
}
# (model, pruner) -> (lmms-eval model name, model_args specific to that pair)
WRAPPERS = {
    ("qwen", "visionzip"): ("qwen2_5_vl_visionzip", "attn_implementation=flash_attention_2"),
    ("llava", "visionzip"): ("llava_next_visionzip", "attn_implementation=flash_attention_2"),
    # FastV scores tokens from layer K-1's attention map (K = fast_v_agg_layer).
    ("qwen", "fastv"): ("qwen2_5_vl_fastv", "attn_implementation=eager,fast_v_agg_layer=2"),
    ("llava", "fastv"): ("llava_next_fastv", "attn_implementation=flash_attention_2,fast_v_agg_layer=2"),
}
# FastV on Qwen occasionally falls into a repetition loop on a hard sample; the
# short-answer VQA tasks are therefore capped at 32 new tokens for that pair.
SHORT_ANSWER_TASKS = {
    "textvqa_val", "infovqa_val", "docvqa_val", "chartqa", "vizwiz_vqa_val",
    "textvqa_val_lite", "infovqa_val_lite", "docvqa_val_lite", "chartqa_lite",
    "vizwiz_vqa_val_lite",
}


def evaluate(model, pruner, method, token_ratio, channel_ratio, task,
             limit=None, log_samples=False):
    name, extra = WRAPPERS[(model, pruner)]
    ratio_arg = "visionzip_dominant_ratio" if pruner == "visionzip" else "fast_v_keep_ratio"
    model_args = (f"pretrained={PRETRAINED[model]},{extra},{ratio_arg}={token_ratio},"
                  f"channel_ratio={channel_ratio},channel_method={method}")
    gen_kwargs = ("max_new_tokens=32"
                  if (model, pruner) == ("qwen", "fastv") and task in SHORT_ANSWER_TASKS else None)

    lm = models.get_model(name).create_from_arg_string(
        model_args, {"batch_size": 1, "max_batch_size": None, "device": "cuda:0"})
    results = simple_evaluate(model=lm, tasks=[task], num_fewshot=0, task_manager=TaskManager(),
                              batch_size=1, gen_kwargs=gen_kwargs, limit=limit,
                              log_samples=log_samples)
    del lm
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(PRETRAINED))
    ap.add_argument("--pruner", required=True, choices=["visionzip", "fastv"])
    ap.add_argument("--method", default="rotatek", choices=["think", "spark", "rotatek"])
    ap.add_argument("--token_ratio", type=float, required=True)
    ap.add_argument("--channel_ratio", type=float, default=0.75)
    ap.add_argument("--tasks", required=True, help="comma-separated lmms-eval task names")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples")
    ap.add_argument("--output_dir", default=os.path.join(_REPO, "results", "accuracy"))
    ap.add_argument("--log_samples", action="store_true", help="also save per-sample responses")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for task in args.tasks.split(","):
        res = evaluate(args.model, args.pruner, args.method, args.token_ratio,
                       args.channel_ratio, task, args.limit, args.log_samples)
        print("results: ", res["results"][task], flush=True)
        stem = (f"{args.model}_{args.pruner}_{args.method}"
                f"_t{args.token_ratio:.2f}_c{args.channel_ratio:.3f}_{task}")
        with open(os.path.join(args.output_dir, stem + ".json"), "w") as f:
            json.dump(res["results"][task], f, indent=1, default=str)
        if args.log_samples:
            with open(os.path.join(args.output_dir, stem + "_samples.jsonl"), "w") as f:
                for s in res["samples"][task]:
                    f.write(json.dumps(s, default=str) + "\n")


if __name__ == "__main__":
    main()
