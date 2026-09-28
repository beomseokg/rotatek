# RotateK — Supplementary Code

This package contains the implementation, evaluation wrappers, and latency
benchmarks for the rotation-based KV-cache channel pruning method
described in the main paper.

## Repository Structure

```
rotatek_supplementary/
├── README.md                    # this file
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT license
├── methods/
│   ├── rotatek/                 # ★ Our method (jacobi.py / decode.py / kernel.py)
│   ├── think.py                 # ThinK baseline
│   └── spark.py                 # SparK baseline
├── kernel/                      # Triton kernels for channel-pruned attention
│   ├── full_channel_flash_decoding_triton.py
│   └── sparse_channel_flash_decoding_triton.py
├── eval/                        # Evaluation wrappers (entry points for accuracy runs)
│   ├── visionzip_qwen.py
│   ├── visionzip_llava_next.py
│   ├── fastv_qwen.py
│   └── fastv_llava_next.py
├── latency/                     # Latency / throughput benchmarks
│   ├── model_end_to_end_*.py    # Per-layer prefill / decode latency
│   ├── max_batch_sweep_llava_next.py  # Max-batch + throughput sweep
│   ├── kernel_breakdown.py      # Per-kernel solver comparison (CholQR / eigh / HMT)
│   └── run_*.sh                 # Sequential sweep drivers
└── lmms_eval_overrides/         # Drop-in overrides for upstream lmms-eval (see below)
    └── lmms_eval/
        ├── models/model_utils/  # RotateK / channel-pruning hooks
        └── tasks/               # Bug fixes for wild_vision_bench / dc100_en
```

## Installation

```bash
# 1) Create a fresh environment (Python 3.11 recommended)
conda create -n rotatek python=3.11 -y && conda activate rotatek

# 2) Install dependencies. PyTorch must come first so triton / flash-attn pick up the right CUDA build.
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 3) Install lmms-eval (we target version 0.6.x — pin to the same minor release we developed against)
pip install lmms-eval==0.6.0

# 4) Apply our drop-in overrides over the installed lmms-eval
LMMS_DIR="$(python -c 'import lmms_eval, os; print(os.path.dirname(lmms_eval.__file__))')"
cp -r lmms_eval_overrides/lmms_eval/* "$LMMS_DIR/"
```

The overrides preserve the original `lmms_eval/...` path structure, so step 4 is a flat copy. Eleven files are touched: eight `models/model_utils/...` files (RotateK + ThinK + SparK hooks into LLaVA-NeXT, Qwen2.5-VL, and InternVL2.5 backbones) and three `tasks/...` files (a `dc100_en` bug fix and a `wild_vision_bench` config fix).

## Reproducing the Main Accuracy Results

```bash
# Qwen2.5-VL-7B-Instruct + VisionZip token pruning + RotateK channel pruning
python eval/visionzip_qwen.py

# Qwen2.5-VL-7B-Instruct + FastV token pruning + RotateK channel pruning
python eval/fastv_qwen.py

# LLaVA-NeXT (llama3-8b) + VisionZip + RotateK
python eval/visionzip_llava_next.py

# LLaVA-NeXT + FastV + RotateK
python eval/fastv_llava_next.py
```

Each wrapper has a `__main__` block that defines the dataset list and method (`think` / `spark` / `rotatek`) to sweep. Edit the lists at the bottom of the wrapper to change which combinations to run. Outputs land in `results/...`.

### Required environment variables

The wrappers read several environment variables; defaults are sensible for most setups but you may want to override:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_HOME` | `~/.cache/huggingface` | HuggingFace cache directory |
| `OPENAI_API_KEY` | — | Required for GPT-judge tasks (`mmvet`, `llava_in_the_wild`, `dc100_en`, etc.) |
| `MODEL_VERSION` | hard-coded `gpt-4o-mini` in the wrappers | Override which GPT model judges open-ended outputs |
| `ROTATEK_QUERY_AWARE` | `1` | Set to `0` to ablate query-weighted PCA (forces K-only PCA) |
| `ROTATEK_SOLVER` | `power_iter` | `power_iter` (Cholesky-QR subspace iteration, default), `randomized` (Halko–Martinsson–Tropp), or any other value (full `torch.linalg.eigh` baseline) |
| `ROTATEK_COMPILE` | unset | Set to `1` or `reduce-overhead` to enable CUDA-graph capture of the subspace iteration |
| `THINK_COMPILE` / `SPARK_COMPILE` | unset | Compile flags for ThinK / SparK baselines |

## Reproducing the Latency / Throughput Benchmarks

```bash
# End-to-end per-layer prefill / decode latency
./latency/run_llava_next_seqlen_sweep.sh    # LLaVA-NeXT, prefill 16k–128k, batch=1
./latency/run_llava_next_batch_sweep.sh     # LLaVA-NeXT, prefill 16k, batch sweep
./latency/run_internvl_seqlen_sweep.sh      # InternVL2.5 sequence-length sweep
./latency/run_internvl_batch_sweep.sh       # InternVL2.5 batch sweep

# Max sustainable batch + throughput (LLaVA-NeXT only)
python latency/max_batch_sweep_llava_next.py sweep \
    --prefill_length 64k \
    --methods full,think,spark,rotatek \
    --decode_tokens 64 \
    --output results/latency_breakdown/throughput_64k.json

# Per-kernel solver comparison (CholQR vs full eigh vs randomized SVD)
python latency/kernel_breakdown.py
```

Each run writes a structured JSON plus a human-readable text log to `results/latency_breakdown/`. Master sweep logs are written to `logs/`.

## Reproducing the Ablation in the Appendix

The ablation table (Cholesky vs `eigh`, query-aware vs query-agnostic) can be reproduced by combining the `ROTATEK_SOLVER` and `ROTATEK_QUERY_AWARE` environment variables with the eval wrappers. For example, on Qwen2.5-VL + FastV:

```bash
# Cholesky + Q-aware (default; our main method)
python eval/fastv_qwen.py

# eigh + Q-aware
ROTATEK_SOLVER=eigh python eval/fastv_qwen.py

# Cholesky + Q-agnostic (K-only PCA baseline)
ROTATEK_QUERY_AWARE=0 python eval/fastv_qwen.py
```

The reported numbers in the appendix use the validation-lite splits curated by `lmms-lab/LMMs-Eval-Lite` (~100–500 examples per benchmark) for tractability across the 36-cell ablation grid. Set the dataset list in `eval/fastv_qwen.py` to e.g. `["textvqa_val_lite", "infovqa_val_lite", "chartqa_lite"]` to match the appendix configuration.

## Known Caveats

1. **Triton kernels.** RotateK's fused decode kernel uses Triton ≥ 2.3 with FlashAttention-2-style block layouts. Older Triton versions or non-NVIDIA GPUs are not supported.
2. **Hardcoded result paths in lmms-eval overrides.** Three of the override files (`qwen2_5vl_visionzip.py`, `llama_visionzip.py`, `internvl2_5_visionzip.py`) contain hardcoded result directories that were used for offline calibration / channel-importance dumps in development. These paths default to `<repo>/...`; they only matter when running with `calibration_mode=collect`, which is **not** required for the main accuracy / latency results. Edit them in place if you need calibration runs.
3. **GPT-judge runs cost API budget.** Tasks like `mmvet`, `llava_in_the_wild`, `dc100_en` invoke the OpenAI API per sample for scoring. Default judge model is `gpt-4o-mini` (~\$0.0002 / sample); change via `MODEL_VERSION`. Plan for a few cents per benchmark.
4. **`vibe_eval` requires Reka API.** The `vibe_eval` task uses Reka Core as judge (not GPT). Install `reka-api` and set `REKA_API_KEY` to use it.

## License

MIT — see `LICENSE`.
