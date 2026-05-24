#!/usr/bin/env bash
# Compose-mode warm-start ablation on A-share CSI300, AlphaGen + AlphaQCM.
# All LLM calls (idea-agent + judge) route through OpenRouter -> DeepSeek.
#
# Conditions (1 seed × 5 pools):
#   B_compose_alphagen_cn          : AlphaGen + compose warm-start
#   A_qcm_cn                       : AlphaQCM vanilla baseline
#   B_compose_qcm_cn               : AlphaQCM + compose warm-start
#   C_compose_alphagen_judge_cn    : B_compose_alphagen post-filtered by judge
#   C_compose_qcm_judge_cn         : B_compose_qcm post-filtered by judge

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# All LLM calls go through OpenRouter, pinned to the official DeepSeek provider
export LLM_BACKEND=${LLM_BACKEND:-openrouter}
LLM_MODEL=${LLM_MODEL:-deepseek/deepseek-chat}
export OPENROUTER_PROVIDER=${OPENROUTER_PROVIDER:-DeepSeek}

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "ERROR: OPENROUTER_API_KEY not set; cannot reach OpenRouter."
    exit 1
fi

N_STEPS=${N_STEPS:-30000}
SEED=${SEED:-42}
TOP_K=${TOP_K:-10}
JUDGE_THRESHOLD=${JUDGE_THRESHOLD:-0.5}
JUDGE_KEEP_TOP_K=${JUDGE_KEEP_TOP_K:-5}
DATA_SOURCE=cn
DEVICE=${DEVICE:-cuda}

mkdir -p data/factors out/results out/tensorboard logs

# --- Sanity: factor library must exist ---
if [[ ! -f "data/factor_library.json" ]]; then
    echo "Building factor library..."
    python scripts/build_factor_library.py
fi

WARM_SEEDS="data/factors/warm_seeds_cn_compose_deepseek_seed${SEED}.json"

# ---------------------------------------------------------------------------
# Step 1: LLM (DeepSeek) composes warm seeds from library primitives
# ---------------------------------------------------------------------------
if [[ -f "$WARM_SEEDS" ]]; then
    echo "[seed=$SEED] compose warm-seeds exist at $WARM_SEEDS, skipping"
else
    echo "[seed=$SEED] composing warm seeds via DeepSeek..."
    python -m src.factor_mining.idea_agent \
        --mode compose \
        --seed "$SEED" \
        --top-k "$TOP_K" \
        --data-source "$DATA_SOURCE" \
        --llm-backend openrouter \
        --model "$LLM_MODEL" \
        --provider "$OPENROUTER_PROVIDER" \
        --out "$WARM_SEEDS" \
        2>&1 | tee "logs/idea_agent_cn_compose_deepseek_seed${SEED}.log"
fi

# ---------------------------------------------------------------------------
# Step 2: AlphaGen training (compose warm-start)
# ---------------------------------------------------------------------------
RUN_B_AG="B_compose_alphagen_cn_seed${SEED}"
if [[ -f "data/factors/${RUN_B_AG}_pool.json" ]]; then
    echo "[$RUN_B_AG] pool exists, skipping"
else
    echo "[$RUN_B_AG] starting AlphaGen with compose warm-start..."
    python -m src.factor_mining.run_alphagen_cn \
        --seed "$SEED" \
        --n-steps "$N_STEPS" \
        --warm-seeds "$WARM_SEEDS" \
        --run-name "$RUN_B_AG" \
        --device "$DEVICE" \
        2>&1 | tee "logs/${RUN_B_AG}.log"
fi

# ---------------------------------------------------------------------------
# Step 3: AlphaQCM training — vanilla baseline
# ---------------------------------------------------------------------------
RUN_A_QCM="A_qcm_cn_seed${SEED}"
if [[ -f "data/factors/${RUN_A_QCM}_pool.json" ]]; then
    echo "[$RUN_A_QCM] pool exists, skipping"
else
    echo "[$RUN_A_QCM] starting AlphaQCM vanilla..."
    python -m src.factor_mining.run_alphaqcm \
        --data-source "$DATA_SOURCE" \
        --seed "$SEED" \
        --model qrdqn \
        --n-steps "$N_STEPS" \
        --run-name "$RUN_A_QCM" \
        2>&1 | tee "logs/${RUN_A_QCM}.log"
fi

# ---------------------------------------------------------------------------
# Step 4: AlphaQCM training — compose warm-start
# ---------------------------------------------------------------------------
RUN_B_QCM="B_compose_qcm_cn_seed${SEED}"
if [[ -f "data/factors/${RUN_B_QCM}_pool.json" ]]; then
    echo "[$RUN_B_QCM] pool exists, skipping"
else
    echo "[$RUN_B_QCM] starting AlphaQCM with compose warm-start..."
    python -m src.factor_mining.run_alphaqcm \
        --data-source "$DATA_SOURCE" \
        --seed "$SEED" \
        --model qrdqn \
        --n-steps "$N_STEPS" \
        --warm-seeds "$WARM_SEEDS" \
        --run-name "$RUN_B_QCM" \
        2>&1 | tee "logs/${RUN_B_QCM}.log"
fi

# ---------------------------------------------------------------------------
# Step 5: Judge post-filter (DeepSeek) on both B pools -> C pools
# ---------------------------------------------------------------------------
for SRC in "$RUN_B_AG" "$RUN_B_QCM"; do
    DST="${SRC/B_compose/C_compose}_judge"
    if [[ -f "data/factors/${DST}_pool.json" ]]; then
        echo "[$DST] filtered pool exists, skipping"
    else
        echo "[$DST] applying DeepSeek judge filter (threshold=$JUDGE_THRESHOLD)..."
        python -m src.evaluation.apply_judge_filter \
            --pool "data/factors/${SRC}_pool.json" \
            --out  "data/factors/${DST}_pool.json" \
            --data-source "$DATA_SOURCE" \
            --threshold "$JUDGE_THRESHOLD" \
            --keep-top-k "$JUDGE_KEEP_TOP_K" \
            --device "$DEVICE" \
            2>&1 | tee "logs/${DST}.log"
    fi
done

# ---------------------------------------------------------------------------
# Step 6: Summary report
# ---------------------------------------------------------------------------
echo
echo "================================================================"
echo "All pools generated by this run:"
ls -la \
    "data/factors/A_qcm_cn_seed${SEED}_pool.json" \
    "data/factors/B_compose_alphagen_cn_seed${SEED}_pool.json" \
    "data/factors/B_compose_qcm_cn_seed${SEED}_pool.json" \
    "data/factors/C_compose_alphagen_cn_seed${SEED}_judge_pool.json" \
    "data/factors/C_compose_qcm_cn_seed${SEED}_judge_pool.json" \
    2>/dev/null || true
echo "================================================================"

python scripts/build_compose_report.py --seed "$SEED" 2>&1 | tee logs/compose_report.log

echo
echo "Tensorboard: tensorboard --logdir $(pwd)/out/tensorboard"
