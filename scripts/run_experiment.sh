#!/usr/bin/env bash
# LLM-augmented AlphaGen ablation: 4 GPU runs + 2 post-hoc judge filters = 6 result pools.
#
# Conditions
#   A_vanilla     : AlphaGen, no LLM
#   B_warm        : AlphaGen + LLM idea-agent warm-start
#   C_warm_judge  : B_warm pool, then judge post-filter (no extra training)
#
# Seeds : 42, 1337
#
# Total GPU time: ~4 x 30min on CUDA + ~5min judge calls.

set -euo pipefail

cd "$(dirname "$0")/.."

N_STEPS=${N_STEPS:-100000}
SEEDS=(42 1337)
N_GENERATE=${N_GENERATE:-50}
TOP_K=${TOP_K:-10}
JUDGE_THRESHOLD=${JUDGE_THRESHOLD:-0.5}
JUDGE_KEEP_TOP_K=${JUDGE_KEEP_TOP_K:-5}
IDEA_MODEL=${IDEA_MODEL:-claude-opus-4-7}

mkdir -p data/factors out/results out/tensorboard logs

# ---------------------------------------------------------------------------
# Step 1: generate warm seeds, one per seed (cached if already on disk)
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    SEEDS_PATH="data/factors/warm_seeds_seed${SEED}.json"
    if [[ -f "$SEEDS_PATH" ]]; then
        echo "[seed=$SEED] warm seeds already exist at $SEEDS_PATH, skipping idea-agent"
    else
        echo "[seed=$SEED] generating warm seeds via idea-agent..."
        python -m src.factor_mining.idea_agent \
            --seed "$SEED" \
            --n-generate "$N_GENERATE" \
            --top-k "$TOP_K" \
            --model "$IDEA_MODEL" \
            --out "$SEEDS_PATH" \
            2>&1 | tee "logs/idea_agent_seed${SEED}.log"
    fi
done

# ---------------------------------------------------------------------------
# Step 2: run AlphaGen for each (condition, seed). Skip C — derived from B.
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    # A_vanilla
    RUN_A="A_vanilla_seed${SEED}"
    if [[ -f "data/factors/${RUN_A}_pool.json" ]]; then
        echo "[$RUN_A] pool exists, skipping"
    else
        echo "[$RUN_A] starting..."
        python -m src.factor_mining.run_alphagen \
            --seed "$SEED" \
            --n-steps "$N_STEPS" \
            --run-name "$RUN_A" \
            2>&1 | tee "logs/${RUN_A}.log"
    fi

    # B_warm
    RUN_B="B_warm_seed${SEED}"
    if [[ -f "data/factors/${RUN_B}_pool.json" ]]; then
        echo "[$RUN_B] pool exists, skipping"
    else
        echo "[$RUN_B] starting..."
        python -m src.factor_mining.run_alphagen \
            --seed "$SEED" \
            --n-steps "$N_STEPS" \
            --warm-seeds "data/factors/warm_seeds_seed${SEED}.json" \
            --run-name "$RUN_B" \
            2>&1 | tee "logs/${RUN_B}.log"
    fi
done

# ---------------------------------------------------------------------------
# Step 3: apply judge post-filter on each B pool to produce C
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    RUN_B="B_warm_seed${SEED}"
    RUN_C="C_warm_judge_seed${SEED}"
    if [[ -f "data/factors/${RUN_C}_pool.json" ]]; then
        echo "[$RUN_C] filtered pool exists, skipping"
    else
        echo "[$RUN_C] applying judge filter (threshold=$JUDGE_THRESHOLD)..."
        python -m src.evaluation.apply_judge_filter \
            --pool "data/factors/${RUN_B}_pool.json" \
            --out  "data/factors/${RUN_C}_pool.json" \
            --threshold "$JUDGE_THRESHOLD" \
            --keep-top-k "$JUDGE_KEEP_TOP_K" \
            2>&1 | tee "logs/${RUN_C}.log"
    fi
done

echo
echo "================================================================"
echo "All 6 result pools:"
ls -la data/factors/{A_vanilla,B_warm,C_warm_judge}_*_pool.json
echo "================================================================"
echo "Tensorboard:  tensorboard --logdir out/tensorboard"
echo "Analysis  :  jupyter notebook notebooks/04_llm_ablation.ipynb"
