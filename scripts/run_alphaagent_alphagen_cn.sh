#!/usr/bin/env bash
# Symmetric counterpart to run_experiment_alphaagent_qcm_cn.sh:
# AlphaGen (PPO) at 300k with the same alphaagent warm seeds, then judge filter.

set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export LLM_BACKEND=${LLM_BACKEND:-openrouter}
export OPENROUTER_PROVIDER=${OPENROUTER_PROVIDER:-DeepSeek}

N_STEPS=${N_STEPS:-300000}
SEED=${SEED:-42}
JUDGE_THRESHOLD=${JUDGE_THRESHOLD:-0.5}
JUDGE_KEEP_TOP_K=${JUDGE_KEEP_TOP_K:-5}
DATA_SOURCE=cn
DEVICE=${DEVICE:-cuda}

WARM_SEEDS="data/factors/warm_seeds_cn_alphaagent_deepseek_seed${SEED}.json"
if [[ ! -f "$WARM_SEEDS" ]]; then
    echo "ERROR: warm seeds not found at $WARM_SEEDS — run scripts/run_experiment_alphaagent_qcm_cn.sh first" >&2
    exit 1
fi

# Step 1: AlphaGen + alphaagent warm-start (300k)
RUN_B="B_alphaagent_alphagen_cn_seed${SEED}"
if [[ -f "data/factors/${RUN_B}_pool.json" ]]; then
    echo "[$RUN_B] pool exists, skipping"
else
    echo "[$RUN_B] starting AlphaGen with alphaagent warm-start..."
    python -m src.factor_mining.run_alphagen_cn \
        --seed "$SEED" \
        --n-steps "$N_STEPS" \
        --warm-seeds "$WARM_SEEDS" \
        --run-name "$RUN_B" \
        --device "$DEVICE" \
        2>&1 | tee "logs/${RUN_B}.log"
fi

# Step 2: DeepSeek judge filter
DST="C_alphaagent_alphagen_cn_seed${SEED}_judge"
if [[ -f "data/factors/${DST}_pool.json" ]]; then
    echo "[$DST] filtered pool exists, skipping"
else
    echo "[$DST] applying DeepSeek judge filter..."
    python -m src.evaluation.apply_judge_filter \
        --pool "data/factors/${RUN_B}_pool.json" \
        --out  "data/factors/${DST}_pool.json" \
        --data-source "$DATA_SOURCE" \
        --threshold "$JUDGE_THRESHOLD" \
        --keep-top-k "$JUDGE_KEEP_TOP_K" \
        --device "$DEVICE" \
        2>&1 | tee "logs/${DST}.log"
fi

echo
echo "================================================================"
ls -la \
    "data/factors/${RUN_B}_pool.json" \
    "data/factors/${DST}_pool.json" 2>/dev/null || true
echo "================================================================"
