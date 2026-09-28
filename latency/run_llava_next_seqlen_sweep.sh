#!/usr/bin/env bash
# Sequential latency sweep on LLaMA3-LLaVA-NeXT-8B for all channel-pruning methods,
# sweeping prefill length at fixed batch=1.
# Each (method, seqlen) pair runs in its own subprocess (clean GPU state —
# fresh CUDA context, fresh Triton autotune cache, fresh allocator). Without
# per-S subprocess separation, Triton autotune cold-start cost and allocator
# fragmentation leak between sequence lengths and produce non-monotonic
# timings (e.g., 32k decode wall < 16k decode wall in the same process).
# Per-(method, S) logs auto-saved to results/latency_breakdown/<auto-named>.{txt,json}
# Master progress logged to logs/llava_next_seqlen_sweep_master.log

set -uo pipefail   # NOTE: not 'set -e' — one method's failure shouldn't kill the rest

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

mkdir -p logs
MASTER_LOG="$REPO/logs/llava_next_seqlen_sweep_master.log"

# Sweep axes — one subprocess per (method, S).
SEQLENS=(16k 32k 64k 128k)

# Common args (single S per invocation now).
COMMON_PER_S=(
    --batch_sizes 1
    --decoding_length 128
    --channel_ratio 0.75
    --n_repeats 5
)

run_one() {
    local label="$1"; shift
    echo "" | tee -a "$MASTER_LOG"
    echo "==================== [$label] $(date) ====================" | tee -a "$MASTER_LOG"
    echo "$@" | tee -a "$MASTER_LOG"
    "$@" 2>&1 | tee -a "$MASTER_LOG"
    local rc=${PIPESTATUS[0]}
    echo "==================== [$label] done (rc=$rc) $(date) ====================" | tee -a "$MASTER_LOG"
    return $rc
}

echo "Sweep started: $(date)" | tee "$MASTER_LOG"

# 1) Full baseline
for S in "${SEQLENS[@]}"; do
    run_one "full_${S}" \
        python -m latency.model_end_to_end_llava_next \
            --methods full \
            --prefill_length "$S" \
            "${COMMON_PER_S[@]}" \
            --tag "a100_llava_next_full_seqlen_v3_${S}"
done

# 2) ThinK (default-mode compile)
for S in "${SEQLENS[@]}"; do
    run_one "think_${S}" env THINK_COMPILE=default \
        python -m latency.model_end_to_end_llava_next \
            --methods think \
            --prefill_length "$S" \
            "${COMMON_PER_S[@]}" \
            --tag "a100_llava_next_think_seqlen_v3_${S}"
done

# 3) SparK (default-mode compile)
for S in "${SEQLENS[@]}"; do
    run_one "spark_${S}" env SPARK_COMPILE=default \
        python -m latency.model_end_to_end_llava_next \
            --methods spark \
            --prefill_length "$S" \
            "${COMMON_PER_S[@]}" \
            --tag "a100_llava_next_spark_seqlen_v3_${S}"
done

# 4) RotateK (reduce-overhead compile)
for S in "${SEQLENS[@]}"; do
    run_one "rotatek_${S}" env ROTATEK_COMPILE=1 \
        python -m latency.model_end_to_end_llava_next \
            --methods rotatek \
            --prefill_length "$S" \
            "${COMMON_PER_S[@]}" \
            --tag "a100_llava_next_rotatek_seqlen_v3_${S}"
done

echo "" | tee -a "$MASTER_LOG"
echo "Sweep complete: $(date)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Per-(method, S) logs:" | tee -a "$MASTER_LOG"
ls -1 "$REPO/results/latency_breakdown/"*a100_llava_next*seqlen*.txt 2>/dev/null | tee -a "$MASTER_LOG"
