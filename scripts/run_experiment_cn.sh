#!/usr/bin/env bash
# LLM-augmented AlphaGen ablation on A-share CSI300 (Qlib).
#
# Conditions
#   A_vanilla_cn     : AlphaGen, no LLM
#   B_warm_cn        : AlphaGen + LLM idea-agent picks from 80-factor library
#   C_warm_judge_cn  : B_warm_cn pool, then judge post-filter (no extra training)
#
# Seeds : 42, 1337  (2 seeds × 3 conditions = 6 result pools, 4 GPU runs)

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

N_STEPS=${N_STEPS:-100000}
SEEDS=(42 1337)
TOP_K=${TOP_K:-10}
JUDGE_THRESHOLD=${JUDGE_THRESHOLD:-0.5}
JUDGE_KEEP_TOP_K=${JUDGE_KEEP_TOP_K:-5}
IDEA_MODEL=${IDEA_MODEL:-claude-opus-4-7}
DATA_SOURCE=cn

mkdir -p data/factors out/results out/tensorboard logs

# --- Sanity: factor library must exist ---
if [[ ! -f "data/factor_library.json" ]]; then
    echo "Building factor library..."
    python scripts/build_factor_library.py
fi

# ---------------------------------------------------------------------------
# Step 1: LLM picks warm seeds from library (one set per seed)
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    SEEDS_PATH="data/factors/warm_seeds_cn_seed${SEED}.json"
    if [[ -f "$SEEDS_PATH" ]]; then
        echo "[seed=$SEED] warm seeds exist at $SEEDS_PATH, skipping idea-agent"
    else
        echo "[seed=$SEED] picking warm seeds from library..."
        python -m src.factor_mining.idea_agent \
            --seed "$SEED" \
            --top-k "$TOP_K" \
            --data-source "$DATA_SOURCE" \
            --model "$IDEA_MODEL" \
            --out "$SEEDS_PATH" \
            2>&1 | tee "logs/idea_agent_cn_seed${SEED}.log"
    fi
done

# ---------------------------------------------------------------------------
# Step 2: AlphaGen training on CSI300 (A_vanilla + B_warm per seed)
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    RUN_A="A_vanilla_cn_seed${SEED}"
    if [[ -f "data/factors/${RUN_A}_pool.json" ]]; then
        echo "[$RUN_A] pool exists, skipping"
    else
        echo "[$RUN_A] starting..."
        python -m src.factor_mining.run_alphagen_cn \
            --seed "$SEED" \
            --n-steps "$N_STEPS" \
            --run-name "$RUN_A" \
            2>&1 | tee "logs/${RUN_A}.log"
    fi

    RUN_B="B_warm_cn_seed${SEED}"
    if [[ -f "data/factors/${RUN_B}_pool.json" ]]; then
        echo "[$RUN_B] pool exists, skipping"
    else
        echo "[$RUN_B] starting..."
        python -m src.factor_mining.run_alphagen_cn \
            --seed "$SEED" \
            --n-steps "$N_STEPS" \
            --warm-seeds "data/factors/warm_seeds_cn_seed${SEED}.json" \
            --run-name "$RUN_B" \
            2>&1 | tee "logs/${RUN_B}.log"
    fi
done

# ---------------------------------------------------------------------------
# Step 3: judge post-filter on each B pool → C
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    RUN_B="B_warm_cn_seed${SEED}"
    RUN_C="C_warm_judge_cn_seed${SEED}"
    if [[ -f "data/factors/${RUN_C}_pool.json" ]]; then
        echo "[$RUN_C] filtered pool exists, skipping"
    else
        echo "[$RUN_C] applying judge filter (threshold=$JUDGE_THRESHOLD)..."
        python -m src.evaluation.apply_judge_filter \
            --pool "data/factors/${RUN_B}_pool.json" \
            --out  "data/factors/${RUN_C}_pool.json" \
            --data-source "$DATA_SOURCE" \
            --threshold "$JUDGE_THRESHOLD" \
            --keep-top-k "$JUDGE_KEEP_TOP_K" \
            2>&1 | tee "logs/${RUN_C}.log"
    fi
done

echo
echo "================================================================"
echo "All A-share result pools:"
ls -la data/factors/{A_vanilla,B_warm,C_warm_judge}_cn_*_pool.json
echo "================================================================"

# ---------------------------------------------------------------------------
# Step 4: execute analysis notebook → produce HTML report
# ---------------------------------------------------------------------------
echo
echo "================================================================"
echo "[4/4] Executing analysis notebook → HTML"
echo "================================================================"
.venv/bin/jupyter nbconvert \
    --to html --execute \
    --allow-errors \
    --ExecutePreprocessor.timeout=600 \
    notebooks/04_llm_ablation.ipynb \
    2>&1 | tee logs/notebook_execution.log

REPORT=notebooks/04_llm_ablation.html
if [[ -f "$REPORT" ]]; then
    echo
    echo "================================================================"
    echo "DONE. Open the report:"
    echo "  $REPORT"
    echo
    echo "Pull to your laptop with:"
    echo "  scp <user>@<linux-host>:$(pwd)/$REPORT ./"
    echo "================================================================"
else
    echo "WARNING: HTML report was not generated. Check logs/notebook_execution.log"
fi
echo "Tensorboard: tensorboard --logdir $(pwd)/out/tensorboard"
