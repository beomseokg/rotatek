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
| **3. Accuracy** | $+$ benchmark datasets (and an API key for GPT-judged tasks) | `scripts/paper/accuracy/` |

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

The latency drivers load a real backbone (still on synthetic token spans, with
the vision tower bypassed), so they additionally need FlashAttention and the
lmms-eval integration:

```bash
# lmms-eval pulls its own torch; install it BEFORE pinning torch, or re-pin after
pip install lmms-eval==0.5.0
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

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
backs up the file it edits.

### Tier 3 — accuracy

Nothing further to install; the benchmarks download on first use through
lmms-eval. GPT-judged tasks (`mmvet`, `llava_in_the_wild`, `dc100_en`) need
`OPENAI_API_KEY`.

---

## Quickstart

Reproduce one accuracy row — Qwen2.5-VL-7B with VisionZip token pruning plus
RotateK channel pruning:

```bash
python scripts/paper/accuracy/visionzip_qwen.py
```

Or measure decode latency without touching lmms-eval or any dataset (synthetic
inputs, vision tower bypassed):

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
# fused sparse-channel decode vs the full-channel baseline
python scripts/paper/kernel/kernel_profile.py --kernel both --seqlen 16000 --batch_size 32

# the same comparison under CUDA-graph capture, which removes launch cost
python scripts/paper/kernel/graph_bench.py

# basis-construction solvers (Cholesky-QR / eigh / randomized)
python scripts/paper/kernel/kernel_breakdown.py
```

**Model latency** (tier 2):

```bash
./scripts/paper/latency/run_llava_next_seqlen_sweep.sh   # prefill 16k-128k, batch 1
./scripts/paper/latency/run_llava_next_batch_sweep.sh    # prefill 16k, batch sweep
python scripts/paper/latency/max_batch_sweep_llava_next.py sweep \
    --prefill_length 64k --methods full,think,spark,rotatek --decode_tokens 64
```

**Accuracy** (tier 3):

| | |
| --- | --- |
| Qwen2.5-VL $+$ VisionZip | `python scripts/paper/accuracy/visionzip_qwen.py` |
| Qwen2.5-VL $+$ FastV | `python scripts/paper/accuracy/fastv_qwen.py` |
| LLaVA-NeXT $+$ VisionZip | `python scripts/paper/accuracy/visionzip_llava_next.py` |
| LLaVA-NeXT $+$ FastV | `python scripts/paper/accuracy/fastv_llava_next.py` |
| Lite ablation, all arms | `python scripts/paper/accuracy/lite_sweep.py` |

`lite_sweep.py` runs TextVQA / InfoVQA / ChartQA (lmms-eval lite) across
token-only, ThinK, SparK and RotateK at matched KV budgets and shards the 24
cells over the visible GPUs; the write-up is in
[`docs/lite_ablation.pdf`](docs/lite_ablation.pdf). The other wrappers take their
dataset list and channel method from the `__main__` block at the bottom.

The appendix ablation (Cholesky vs. `eigh`, query-aware vs. query-agnostic) comes
from the same wrappers with different environment variables:

```bash
python scripts/paper/accuracy/fastv_qwen.py                          # Cholesky + Q-aware (default)
ROTATEK_SOLVER=eigh python scripts/paper/accuracy/fastv_qwen.py      # full eigendecomposition
ROTATEK_QUERY_AWARE=0 python scripts/paper/accuracy/fastv_qwen.py    # K-only PCA
```

---

## Repository layout

```
rotatek/                 core method — no model-specific code
  rotation.py            online PCA: covariance → subspace iteration → R_k, δμ
  decode.py              decode entry points
  kernels/               Triton kernels (fused sparse-channel + full-channel decode)
  baselines/             ThinK and SparK, for comparison
integrations/lmms_eval/  drop-in overrides for an installed lmms-eval
scripts/paper/kernel/    kernel microbenchmarks   (torch + triton only)
scripts/paper/latency/   model latency drivers    (+ backbone, lmms-eval)
scripts/paper/accuracy/  accuracy drivers         (+ datasets)
assets/                  figures
```

`rotatek/rotation.py` and `rotatek/kernels/` have no dependency on any model or
cache class, so they can be reused outside this repo.

---

## Configuration

Behaviour is controlled by environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ROTATEK_QUERY_AWARE` | `1` | `0` ablates query-weighted PCA (K-only PCA) |
| `ROTATEK_SOLVER` | `power_iter` | `power_iter` (Cholesky-QR subspace iteration), `randomized` (Halko–Martinsson–Tropp), anything else falls back to `torch.linalg.eigh` |
| `ROTATEK_STORAGE` | `truncated` | `truncated` stores rotated Keys at `D_keep` and rotates Q at decode (memory **and** compute savings); `full` stores `K R Rᵀ + μ` at full width (accuracy-equivalent, no savings — useful as a drop-in for the standard FA2 path) |
| `ROTATEK_COMPILE` | unset | `1` or `reduce-overhead` to CUDA-graph the subspace iteration |
| `THINK_COMPILE` / `SPARK_COMPILE` | unset | Same, for the baselines |
| `HF_HOME` | `~/.cache/huggingface` | Model cache |
| `OPENAI_API_KEY` | — | Needed only for GPT-judged tasks (`mmvet`, `llava_in_the_wild`, `dc100_en`) |
| `MODEL_VERSION` | `gpt-4o-mini` | Judge model for those tasks |

---

## Caveats

1. **Triton only.** The fused decode kernel targets Triton ≥ 2.3 with
   FlashAttention-2-style block layouts on NVIDIA GPUs. There is no CPU or ROCm
   path.
2. **Pinned to `transformers==4.47.0`.** The integration files fork HuggingFace
   modeling code, so other versions are not expected to work.
3. **Calibration paths.** Three integration files contain hardcoded directories
   used for offline channel-importance dumps during development. They only matter
   under `calibration_mode=collect`, which none of the reported results use.
4. **GPT-judged tasks cost money.** `mmvet`, `llava_in_the_wild` and `dc100_en`
   call the OpenAI API per sample (~$0.0002/sample with `gpt-4o-mini`).
   `vibe_eval` uses Reka Core instead and needs `reka-api` plus `REKA_API_KEY`.

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
