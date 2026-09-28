#!/usr/bin/env bash
# Sequential latency sweep on InternVL2.5 for all channel-pruning methods,
# sweeping batch sizes at fixed prefill length 16k.
# Each method runs in its own process (clean GPU state, no cross-method drift).
# Per-method logs auto-saved to results/latency_breakdown/<auto-named>.{txt,json}
# Master progress logged to logs/internvl_batch_sweep_master.log

set -uo pipefail   # NOTE: not 'set -e' — one method's failure shouldn't kill the rest

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

mkdir -p logs
MASTER_LOG="$REPO/logs/internvl_batch_sweep_master.log"

# Common args
COMMON=(
    --prefill_length 16k
    --batch_sizes 4,8
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
    python -m scripts.paper.latency.model_end_to_end_internvl \
        --methods full \
        "${COMMON[@]}" \
        --tag a100_internvl_full_batch_v2

# 2) ThinK (default-mode compile)
run_one "think" env THINK_COMPILE=default \
    python -m scripts.paper.latency.model_end_to_end_internvl \
        --methods think \
        "${COMMON[@]}" \
        --tag a100_internvl_think_batch_v2

# 3) SparK (default-mode compile)
run_one "spark" env SPARK_COMPILE=default \
    python -m scripts.paper.latency.model_end_to_end_internvl \
        --methods spark \
        "${COMMON[@]}" \
        --tag a100_internvl_spark_batch_v2

# 4) RotateK (reduce-overhead compile)
run_one "rotatek" env ROTATEK_COMPILE=1 \
    python -m scripts.paper.latency.model_end_to_end_internvl \
        --methods rotatek \
        "${COMMON[@]}" \
        --tag a100_internvl_rotatek_batch_v2

echo "" | tee -a "$MASTER_LOG"
echo "Sweep complete: $(date)" | tee -a "$MASTER_LOG"
echo "" | tee -a "$MASTER_LOG"
echo "Per-method logs:" | tee -a "$MASTER_LOG"
ls -1 "$REPO/results/latency_breakdown/"*a100_internvl*batch*.txt 2>/dev/null | tee -a "$MASTER_LOG"
