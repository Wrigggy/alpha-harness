#!/usr/bin/env bash
# Mac smoke test: validates the full pipeline using crypto data (no Qlib).
# Should complete in ~5-10 min total. Run BEFORE the CSI300 experiment.

set -euo pipefail

cd "$(dirname "$0")/.."

# Activate venv so `python` resolves to the project interpreter
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

mkdir -p data/factors out/results logs

echo
echo "================================================================"
echo "[1/4] Build factor library (if missing)"
echo "================================================================"
if [[ ! -f "data/factor_library.json" ]]; then
    .venv/bin/python scripts/build_factor_library.py
else
    echo "Library exists, skipping"
fi

echo
echo "================================================================"
echo "[2/4] Idea-agent: pick warm seeds (crypto IC scoring)"
echo "================================================================"
.venv/bin/python -m src.factor_mining.idea_agent \
    --seed 42 --top-k 5 \
    --data-source crypto \
    --out data/factors/warm_seeds_smoke.json \
    2>&1 | tee logs/smoke_idea_agent.log

echo
echo "================================================================"
echo "[3/4] AlphaGen small-scale training with warm seeds"
echo "================================================================"
.venv/bin/python -m src.factor_mining.run_alphagen --small-scale \
    --seed 42 \
    --warm-seeds data/factors/warm_seeds_smoke.json \
    --run-name smoke_test \
    --device cpu \
    2>&1 | tee logs/smoke_run_alphagen.log

echo
echo "================================================================"
echo "[4/4] Judge post-filter on smoke pool"
echo "================================================================"
.venv/bin/python -m src.evaluation.apply_judge_filter \
    --pool data/factors/smoke_test_pool.json \
    --out  data/factors/smoke_test_filtered.json \
    --data-source crypto \
    --threshold 0.4 \
    --keep-top-k 3 \
    --device cpu \
    2>&1 | tee logs/smoke_judge_filter.log

echo
echo "================================================================"
echo "Smoke test complete. Outputs:"
ls -la data/factors/warm_seeds_smoke.json data/factors/smoke_test_pool.json data/factors/smoke_test_filtered.json
echo "================================================================"
