#!/usr/bin/env bash
# AlphaAgent (free-form + structural novelty regularization) warm-start
# ablation on AlphaQCM at 300k steps. CSI300, single seed.
#
# Conditions:
#   A_qcm_300k                   : AlphaQCM vanilla baseline at 300k
#   B_alphaagent_qcm             : AlphaQCM + alphaagent compose warm-start
#   C_alphaagent_qcm_judge       : B post-filtered by DeepSeek judge
#
# All LLM calls route through OpenRouter pinned to DeepSeek provider.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export LLM_BACKEND=${LLM_BACKEND:-openrouter}
LLM_MODEL=${LLM_MODEL:-deepseek/deepseek-v4-pro}
export OPENROUTER_PROVIDER=${OPENROUTER_PROVIDER:-DeepSeek}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set." >&2
    exit 1
fi

N_STEPS=${N_STEPS:-300000}
SEED=${SEED:-42}
TOP_K=${TOP_K:-10}
JUDGE_THRESHOLD=${JUDGE_THRESHOLD:-0.5}
JUDGE_KEEP_TOP_K=${JUDGE_KEEP_TOP_K:-5}
DATA_SOURCE=cn
DEVICE=${DEVICE:-cuda}

mkdir -p data/factors out/results out/tensorboard logs

WARM_SEEDS="data/factors/warm_seeds_cn_alphaagent_deepseek_seed${SEED}.json"

# ---------------------------------------------------------------------------
# Step 1: AlphaAgent free-form warm seeds (DeepSeek + AST novelty regulator)
# ---------------------------------------------------------------------------
if [[ -f "$WARM_SEEDS" ]]; then
    echo "[seed=$SEED] alphaagent warm-seeds exist at $WARM_SEEDS, skipping"
else
    echo "[seed=$SEED] generating alphaagent warm seeds (DeepSeek + regulator)..."
    python -m src.factor_mining.idea_agent \
        --mode alphaagent \
        --seed "$SEED" \
        --top-k "$TOP_K" \
        --data-source "$DATA_SOURCE" \
        --llm-backend openrouter \
        --model "$LLM_MODEL" \
        --provider "$OPENROUTER_PROVIDER" \
        --out "$WARM_SEEDS" \
        2>&1 | tee "logs/idea_agent_cn_alphaagent_deepseek_seed${SEED}.log"
fi

# ---------------------------------------------------------------------------
# Step 2: AlphaQCM vanilla baseline (300k) — distinct from existing 30k A_qcm
# ---------------------------------------------------------------------------
RUN_A="A_qcm_300k_cn_seed${SEED}"
if [[ -f "data/factors/${RUN_A}_pool.json" ]]; then
    echo "[$RUN_A] pool exists, skipping"
else
    echo "[$RUN_A] starting AlphaQCM vanilla 300k..."
    python -m src.factor_mining.run_alphaqcm \
        --data-source "$DATA_SOURCE" \
        --seed "$SEED" \
        --model qrdqn \
        --n-steps "$N_STEPS" \
        --run-name "$RUN_A" \
        2>&1 | tee "logs/${RUN_A}.log"
fi

# ---------------------------------------------------------------------------
# Step 3: AlphaQCM + alphaagent warm-start (300k)
# ---------------------------------------------------------------------------
RUN_B="B_alphaagent_qcm_cn_seed${SEED}"
if [[ -f "data/factors/${RUN_B}_pool.json" ]]; then
    echo "[$RUN_B] pool exists, skipping"
else
    echo "[$RUN_B] starting AlphaQCM with alphaagent warm-start..."
    python -m src.factor_mining.run_alphaqcm \
        --data-source "$DATA_SOURCE" \
        --seed "$SEED" \
        --model qrdqn \
        --n-steps "$N_STEPS" \
        --warm-seeds "$WARM_SEEDS" \
        --run-name "$RUN_B" \
        2>&1 | tee "logs/${RUN_B}.log"
fi

# ---------------------------------------------------------------------------
# Step 4: DeepSeek judge post-filter on B pool
# ---------------------------------------------------------------------------
DST="C_alphaagent_qcm_cn_seed${SEED}_judge"
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
echo "Pools produced by this run:"
ls -la \
    "data/factors/A_qcm_300k_cn_seed${SEED}_pool.json" \
    "data/factors/B_alphaagent_qcm_cn_seed${SEED}_pool.json" \
    "data/factors/C_alphaagent_qcm_cn_seed${SEED}_judge_pool.json" \
    2>/dev/null || true
echo "================================================================"
