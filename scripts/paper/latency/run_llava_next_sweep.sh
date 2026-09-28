#!/usr/bin/env bash
# Sequential latency sweep on LLaVA-NeXT for all channel-pruning methods.
# Each method runs in its own process (clean GPU state, no cross-method drift).
# Per-method logs auto-saved to results/latency_breakdown/<auto-named>.{txt,json}
# Master progress logged to logs/llava_next_sweep_master.log

set -uo pipefail   # NOTE: not 'set -e' — one method's failure shouldn't kill the rest

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

mkdir -p logs
MASTER_LOG="$REPO/logs/llava_next_sweep_master.log"

# Common args
COMMON=(
    --prefill_length 16k,32k,64k,128k
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
run_one "full" \
    python -m latency.model_end_to_end_llava_next \
        --methods full \
        "${COMMON[@]}" \
        --tag a100_llava_next_full

# 2) ThinK (default-mode compile)
run_one "think" env THINK_COMPILE=default \
    python -m latency.model_end_to_end_llava_next \
        --methods think \
        "${COMMON[@]}" \
        --tag a100_llava_next_think

# 3) SparK (default-mode compile)
run_one "spark" env SPARK_COMPILE=default \
    python -m latency.model_end_to_end_llava_next \
        --methods spark \
        "${COMMON[@]}" \
        --tag a100_llava_next_spark

# 4) RotateK Q-aware (Q-weighted PCA + reduce-overhead compile)
run_one "rotatek_qaware" env ROTATEK_QUERY_AWARE=1 ROTATEK_COMPILE=1 \
    python -m latency.model_end_to_end_llava_next \
        --methods rotatek \
        "${COMMON[@]}" \
        --tag a100_llava_next_rotatek_qaware

# 5) RotateK Q-agnostic (vanilla — for comparison)
run_one "rotatek_qagnostic" env ROTATEK_COMPILE=1 \
    python -m latency.model_end_to_end_llava_next \
        --methods rotatek \
        "${COMMON[@]}" \
        --tag a100_llava_next_rotatek_qagnostic

echo "" | tee -a "$MASTER_LOG"
echo "Sweep complete: $(date)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Per-method logs:" | tee -a "$MASTER_LOG"
ls -1 "$REPO/results/latency_breakdown/"*a100_llava_next*.txt 2>/dev/null | tee -a "$MASTER_LOG"
