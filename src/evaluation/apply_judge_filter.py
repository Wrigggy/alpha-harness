"""Apply LLM judge as a post-filter on a saved AlphaGen pool.

Reads a pool JSON (output of run_alphagen.py), scores each expression with
the configured LLM judge, drops those below threshold, and re-evaluates the
resulting ensemble on val/test calculators.

Safety net: always keep the top-K (default 5) factors by |single IC| even if
they score below the threshold, so we never produce an empty pool.

Usage:
    python -m src.evaluation.apply_judge_filter \
        --pool data/factors/E_warm_judge_seed42_pool.json \
        --out data/factors/E_warm_judge_seed42_filtered.json \
        --threshold 0.5 \
        --keep-top-k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "external" / "alphagen"))

from alphagen.config import OPERATORS  # noqa: E402
from alphagen.data.expression import Feature, FeatureType, Ref  # noqa: E402
from alphagen.data.parser import ExpressionParser  # noqa: E402
from alphagen.models.linear_alpha_pool import MseAlphaPool  # noqa: E402

from src.data_adapter.to_alphagen_format import create_data_splits, CryptoAlphaCalculator  # noqa: E402
from src.llm_judge.claude_agent_judge import ClaudeAgentJudge  # noqa: E402
from src.utils.device import get_device  # noqa: E402


def _build_parser() -> ExpressionParser:
    return ExpressionParser(
        operators=OPERATORS,
        ignore_case=False,
        time_deltas_need_suffix=True,
        non_positive_time_deltas_allowed=False,
        feature_need_dollar_sign=True,
    )


def _load_judge_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f).get("judge", {})


def apply_filter(
    pool_path: Path,
    out_path: Path,
    threshold: float,
    keep_top_k: int,
    judge_config_path: str,
    data_config_path: str,
) -> dict:
    pool_dict = json.loads(pool_path.read_text(encoding="utf-8"))
    exprs: list[str] = pool_dict["exprs"]
    weights: list[float] = pool_dict["weights"]
    n = len(exprs)
    logger.info(f"Loaded pool with {n} factors from {pool_path}")

    # Rebuild data + calculators
    with open(data_config_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    processed_dir = data_cfg["data"]["processed_dir"]
    device = get_device("auto")
    splits = create_data_splits(
        processed_dir, data_config_path, device=device,
        max_backtrack_days=100, max_future_days=10,
    )
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -8) / close - 1
    train_calc = CryptoAlphaCalculator(splits["train"], target)
    valid_calc = CryptoAlphaCalculator(splits["val"], target)
    test_calc = CryptoAlphaCalculator(splits["test"], target)

    # Compute per-factor IC on train so we can rank for the "keep top-K" safety net
    parser = _build_parser()
    parsed = [parser.parse(e) for e in exprs]
    single_ics = []
    for e in parsed:
        try:
            single_ics.append(float(train_calc.calc_single_IC_ret(e)))
        except Exception:
            single_ics.append(0.0)
    abs_ics = np.abs(single_ics)

    # Score with judge
    judge_cfg = _load_judge_config(judge_config_path)
    prompts = judge_cfg.get("prompts", {})
    judge = ClaudeAgentJudge(
        model=judge_cfg.get("model", "claude-opus-4-7"),
        score_prompt_path=prompts.get("scoring", "prompts/score.txt"),
        translate_prompt_path=prompts.get("translation", "prompts/translate.txt"),
    )
    candidates = [
        {"expression": e, "ic": float(ic), "matched_papers": []}
        for e, ic in zip(exprs, single_ics)
    ]
    logger.info(f"Scoring {n} factors with judge (threshold={threshold})")
    results = judge.batch_score(candidates)
    scores = np.array([r.interpretability_score for r in results])

    # Decide which factors survive
    keep_mask = scores >= threshold
    if keep_top_k > 0:
        topk_idx = np.argsort(-abs_ics)[:keep_top_k]
        keep_mask[topk_idx] = True
    kept = [i for i in range(n) if keep_mask[i]]
    dropped = [i for i in range(n) if not keep_mask[i]]
    logger.info(f"Kept {len(kept)}/{n} factors; dropped {len(dropped)}")

    # Rebuild a fresh pool with surviving expressions (lets the combiner re-fit weights)
    pool = MseAlphaPool(
        capacity=max(len(kept), 1),
        calculator=train_calc,
        ic_lower_bound=None,
        l1_alpha=5e-3,
        device=device,
    )
    kept_exprs = [parsed[i] for i in kept]
    pool.force_load_exprs(kept_exprs)

    val_ic, val_ric = pool.test_ensemble(valid_calc)
    test_ic, test_ric = pool.test_ensemble(test_calc)

    out_state = pool.to_json_dict()
    out_state["val_ic"] = val_ic
    out_state["val_ric"] = val_ric
    out_state["test_ic"] = test_ic
    out_state["test_ric"] = test_ric
    out_state["timestamp"] = datetime.now().isoformat()
    out_state["filter"] = {
        "source_pool": str(pool_path),
        "threshold": threshold,
        "keep_top_k": keep_top_k,
        "n_input": n,
        "n_kept": len(kept),
        "kept_scores": [float(scores[i]) for i in kept],
        "dropped_scores": [float(scores[i]) for i in dropped],
        "dropped_exprs": [exprs[i] for i in dropped],
        "judge_results": [
            {
                "expr": exprs[i],
                "score": float(scores[i]),
                "kept": bool(keep_mask[i]),
                "nl": results[i].nl_description,
                "narrative": results[i].economic_narrative,
            }
            for i in range(n)
        ],
    }
    out_state["run_name"] = pool_dict.get("run_name", pool_path.stem) + "_filtered"
    out_state["seed"] = pool_dict.get("seed")
    out_state["n_steps"] = pool_dict.get("n_steps")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_state, indent=2), encoding="utf-8")
    logger.info(
        f"Filtered pool saved: size={pool.size}, "
        f"val_IC={val_ic:.4f}, test_IC={test_ic:.4f} -> {out_path}"
    )
    return out_state


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--keep-top-k", type=int, default=5)
    ap.add_argument("--judge-config", default="config/judge_config.yaml")
    ap.add_argument("--data-config", default="config/data_config.yaml")
    args = ap.parse_args()
    apply_filter(
        pool_path=args.pool,
        out_path=args.out,
        threshold=args.threshold,
        keep_top_k=args.keep_top_k,
        judge_config_path=args.judge_config,
        data_config_path=args.data_config,
    )
