<h1 align="center">RotateK</h1>
<h3 align="center">Rotation-Aligned Key Channel Pruning for Efficient Vision-Language Model Inference</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2605.19218"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.19218-b31b1b?logo=arxiv&logoColor=white"></a>
  <a href="https://github.com/beomseokg/rotatek"><img alt="code" src="https://img.shields.io/badge/github-code-181717?logo=github&logoColor=white"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-2ea44f"></a>
</p>

![RotateK inference flow](assets/inference_flow.png)

This is the **official implementation** of *"Rotation-Aligned Key Channel Pruning
for Efficient Vision-Language Model Inference"*.

A single image can occupy thousands of KV-cache entries in a VLM. Token pruning
shrinks that cache by discarding visual tokens outright, which costs accuracy on
fine-grained perception. RotateK compresses the *channel* dimension of the Keys
instead: an online PCA-based rotation aligns token-dependent channel importance
into a shared subspace, so a lightweight head-wise mask can keep a quarter of the
Key channels with little loss. Under a fixed KV budget the freed memory buys more
visual tokens — or, at a fixed token count, a smaller cache.

At **prefill** the rotation `R_k` is built once per layer and the visual Keys are
stored rotated and truncated. At **decode** nothing is reconstructed: the query is
rotated into the same subspace (`q_t → R_k → q̃_t`) and a fused Triton kernel runs
the sparse-channel path over the visual span and the full-channel path over the
prompt and text span in one launch, merging them with a single online softmax.

---

## Install

Three kinds of experiment live in this repo and they need different things
installed. Do only the tier you need.

| | needs | directory |
| --- | --- | --- |
| **1. Kernel microbenchmarks** | torch + triton | `scripts/paper/kernel/` |
| **2. Model latency** | $+$ the backbone weights, FlashAttention, lmms-eval integration | `scripts/paper/latency/` |
| **3. Accuracy** | $+$ benchmark datasets | `scripts/paper/accuracy/` |

Python 3.11 and a CUDA-capable GPU are required throughout (the decode kernel is
Triton).

### Tier 1 — kernels

```bash
git clone https://github.com/beomseokg/rotatek.git
cd rotatek
conda create -n rotatek python=3.11 -y && conda activate rotatek

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

No `pip install` of this repo is needed — the scripts put the repo root on
`sys.path` themselves, so they run from any working directory. This is enough for
everything under `scripts/paper/kernel/`, which builds its inputs synthetically
and touches neither a checkpoint nor a dataset.

### Tier 2 — model latency

The latency drivers load a real backbone (on synthetic token spans, with the
vision tower bypassed), so they additionally need lmms-eval, FlashAttention and
the lmms-eval integration:

```bash
# lmms-eval would otherwise upgrade torch; the constraint keeps 2.6.0
printf 'torch==2.6.0\ntorchvision==0.21.0\n' > constraints.txt
pip install lmms-eval==0.5.0 -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu124

pip install flash-attn==2.7.4.post1 --no-build-isolation

python integrations/apply_overrides.py          # copy files + register models
python integrations/apply_overrides.py --check  # verify
```

> FlashAttention is installed separately because its `setup.py` imports `torch` at
> build time, which fails inside pip's isolated build environment. On clusters with
> an NFS home this can also fail with `Invalid cross-device link` — pip builds in a
> local `/tmp` and cannot move the wheel onto NFS. Point both at one filesystem:
> `TMPDIR=~/.cache/pip/tmp pip install flash-attn==2.7.4.post1 --no-build-isolation`

`apply_overrides.py` does three things a plain file copy does not: it copies the
integration files, registers the wrappers in lmms-eval's model registry (otherwise
`--model qwen2_5_vl_visionzip` reports "not found"), and drops a `.pth` so the
copied wrappers can `import rotatek` from site-packages. It is idempotent and
backs up the file it edits. It also replaces one stock file,
`tasks/_task_utils/file_utils.py`, whose TextVQA submission hook crashes when
`simple_evaluate` is called from Python.

### Tier 3 — accuracy

Nothing further to install; the benchmarks download on first use through
lmms-eval. The five VQA benchmarks the paper reports (TextVQA, InfoVQA, ChartQA,
DocVQA, VizWiz) are scored by rule-based metrics — exact match, ANLS, relaxed
accuracy — so no API key is involved.

Only the open-ended benchmarks need one: `llava_in_the_wild` and `mmvet` are
scored by a GPT judge (`gpt-4o-mini`), so running those requires
`OPENAI_API_KEY` and costs roughly \$0.0002 per sample.

---

## Quickstart

Qwen2.5-VL-7B with VisionZip token pruning plus RotateK channel pruning:

```bash
python scripts/paper/accuracy/eval_accuracy.py --model qwen --pruner visionzip \
    --method rotatek --token_ratio 0.40 --channel_ratio 0.75 --tasks textvqa_val
```

Or measure decode latency without any dataset (synthetic inputs, vision tower
bypassed):

```bash
python scripts/paper/latency/model_end_to_end_llava_next.py \
    --methods full,think,spark,rotatek \
    --prefill_length 16k --batch_sizes 1 --decoding_length 128 \
    --channel_ratio 0.75 --n_repeats 5
```

`channel_ratio` is the fraction **pruned**, so `0.75` keeps 25% of the Key
channels — the setting used throughout the paper.

---

## Reproducing the paper

**Kernels** (tier 1 — no checkpoint, no dataset):

```bash
# fused RotateK decode vs the full-channel baseline
python scripts/paper/kernel/kernel_profile.py --kernel both --seqlen 16000 --batch_size 32

# the same comparison under CUDA-graph capture, which removes launch cost,
# for both kernel families (see "Decode kernels" below)
python scripts/paper/kernel/graph_bench.py
```

**Decode kernels.** Two kernel families are included, selected with
`ROTATEK_KERNEL` for every script (RotateK and the full-width baselines switch
together, so methods are always compared on the same family):

| `ROTATEK_KERNEL` | | |
| --- | --- | --- |
| `paper` (default) | `fused_decode.py`, `full_channel_flash_decoding.py` | one program per query head; the paper's latency results (Sec. 4.3, Fig. 7) were measured with these |
| `gqa` | `gqa_decode.py` | the query heads of a GQA group share each K/V tile; 2.9–6.3× faster kernels at 16K context, batch 32 |

With `gqa` the RotateK-over-dense kernel speedup settles near the K/V byte ratio
(about 1.5× from 16K context and batch 4 up) instead of the ~2× seen with
`paper`, where the dense baseline re-reads K/V once per query head. At 1K
context it is about 1.0× at batch 1 and 1.2–1.4× at batch 16–32; there the
kernels are short enough that wave quantization makes timings shape-sensitive.

**Model latency** (tier 2, LLaVA-NeXT-8B as in the paper):

```bash
./scripts/paper/latency/run_llava_next_seqlen_sweep.sh   # prefill 16k-128k, batch 1
./scripts/paper/latency/run_llava_next_batch_sweep.sh    # prefill 16k, batch sweep
python scripts/paper/latency/max_batch_sweep_llava_next.py sweep \
    --prefill_length 64k --methods full,think,spark,rotatek --output sweep_64k.json
python scripts/paper/latency/plot_max_batch_sweep.py sweep_64k.json --output sweep_64k.png
```

Add `--profile_breakdown` to `model_end_to_end_llava_next.py` for the per-stage
decode breakdown. The sweeps set `THINK_COMPILE` / `SPARK_COMPILE` /
`ROTATEK_COMPILE` so every method runs compiled.

**Accuracy** (tier 3). One script covers both backbones and both token pruners:

```bash
TASKS=textvqa_val,infovqa_val,chartqa,docvqa_val,vizwiz_vqa_val

# token pruning + Key channel pruning (method: think | spark | rotatek)
python scripts/paper/accuracy/eval_accuracy.py --model llava --pruner fastv \
    --method rotatek --token_ratio 0.30 --channel_ratio 0.75 --tasks $TASKS

# token pruning only, at a matched KV budget
python scripts/paper/accuracy/eval_accuracy.py --model llava --pruner fastv \
    --token_ratio 0.19 --channel_ratio 0 --tasks $TASKS

# unpruned baseline
python scripts/paper/accuracy/eval_accuracy.py --model llava --pruner visionzip \
    --token_ratio 1.0 --channel_ratio 0 --tasks $TASKS
```

`--model` is `qwen` (Qwen2.5-VL-7B-Instruct) or `llava` (llama3-llava-next-8b);
`--pruner` is `visionzip` (`--token_ratio` = dominant ratio, contextual fixed at
0.05) or `fastv` (`--token_ratio` = keep ratio at layer K=2). Results go to
`results/accuracy/`; `--limit N` evaluates only the first N samples.

`scripts/paper/accuracy/lite_sweep.py` runs the lite ablation (TextVQA / InfoVQA
/ ChartQA lite × token-only, ThinK, SparK and RotateK at matched KV budgets) over
the visible GPUs; the write-up is in [`docs/lite_ablation.pdf`](docs/lite_ablation.pdf).

The appendix ablation (Cholesky vs. `eigh`, query-aware vs. query-agnostic) uses
the same script with environment variables:

```bash
ROTATEK_SOLVER=eigh      python scripts/paper/accuracy/eval_accuracy.py ...   # full eigendecomposition
ROTATEK_QUERY_AWARE=0    python scripts/paper/accuracy/eval_accuracy.py ...   # K-only PCA
```

---

## Repository layout

```
rotatek/                   core method — no model-specific code
  rotation.py              top-k eigenbasis by Cholesky-QR subspace iteration
  kernels/fused_decode.py  decode over rotated-truncated visual Keys (2 Triton kernels)
  kernels/full_channel_flash_decoding.py   full-width decode (baselines)
  kernels/gqa_decode.py    both of the above with K/V shared across a GQA group
  baselines/               ThinK and SparK channel selection
integrations/lmms_eval/    installed into lmms-eval by apply_overrides.py
  models/simple/           lmms-eval wrappers (Qwen2.5-VL / LLaVA-NeXT × VisionZip / FastV)
  models/model_utils/      patched model code; kv_pruning_utils.py holds the
                           per-layer ThinK / SparK / RotateK prefill step
scripts/paper/kernel/      kernel microbenchmarks   (torch + triton only)
scripts/paper/latency/     model latency drivers    (+ backbone, lmms-eval)
scripts/paper/accuracy/    accuracy drivers         (+ datasets)
```

`qwen2_5vl_visionzip.py` is transformers 4.49.0's `modeling_qwen2_5_vl.py` with
the RotateK / VisionZip changes listed at its top, so it can be diffed against
upstream. `rotatek/` has no dependency on any model or cache class.

---

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `ROTATEK_SOLVER` | `power_iter` | `power_iter` (Cholesky-QR subspace iteration) or `eigh` (exact, the solver ablation) |
| `ROTATEK_QUERY_AWARE` | `1` | `0` ablates query-weighted PCA (K-only PCA) |
| `ROTATEK_KERNEL` | `paper` | decode kernel family: `paper` or `gqa` (see "Decode kernels") |
| `ROTATEK_COMPILE` | unset | `1` CUDA-graphs the subspace iteration (latency runs) |
| `THINK_COMPILE` / `SPARK_COMPILE` | unset | `default` compiles the baselines' selection (latency runs) |
| `OPENAI_API_KEY` | — | Only for `llava_in_the_wild` / `mmvet` (GPT-judged) |

---

## Caveats

1. **Triton only.** The decode kernels target NVIDIA GPUs through Triton; there
   is no CPU or ROCm path.
2. **Pinned to `transformers==4.49.0`.** The integration forks HuggingFace
   modeling code, so other versions are not expected to work.
3. **Batch size 1 for accuracy.** The VisionZip prefill for LLaVA-NeXT slices the
   sequence per sample and assumes one sample per batch, as lmms-eval runs it.

---

## Citation

```bibtex
@article{kang2026rotatek,
  title   = {Rotation-Aligned Key Channel Pruning for Efficient Vision-Language Model Inference},
  author  = {Kang, Beomseok and Jo, Dongwon and Song, Jiwon and Son, Donghwee and Kim, Jae-Joon},
  journal = {arXiv preprint arXiv:2605.19218},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
