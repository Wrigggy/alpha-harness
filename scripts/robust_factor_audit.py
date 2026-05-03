"""Audit a factor pool for cross-split stability and noise sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ALPHAGEN = ROOT / "external" / "alphagen"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXTERNAL_ALPHAGEN) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ALPHAGEN))

from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen.data.parser import parse_expression

from src.data_adapter.to_alphagen_format import PanelAlphaCalculator, _load_local_panel, create_data_splits
from src.evaluation.factor_correlation import compute_factor_correlation, find_redundant_pairs
from src.evaluation.ic_analysis import evaluate_factor
from src.evaluation.validation_gate import ValidationConfig, ValidationGate
from src.pipeline import build_forward_returns, tensor_to_factor_frame
from src.utils.pool_io import load_pool, normalize_weights


def _load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _noise_stress(frame: pd.DataFrame, seed: int = 7, sigma: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noisy = frame.copy().astype(float)
    noise = rng.normal(0.0, sigma, size=noisy.shape)
    noisy = noisy * (1.0 + noise)
    return noisy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stability audits on a factor pool")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-rank-ic", type=float, default=0.02)
    parser.add_argument("--min-rank-icir", type=float, default=0.20)
    parser.add_argument("--max-correlation", type=float, default=0.65)
    parser.add_argument("--noise-sigma", type=float, default=0.02)
    args = parser.parse_args()

    data_cfg = _load_cfg(args.data_config)
    source_cfg = data_cfg.get("source", {})
    source_name = source_cfg.get("name", "crypto")
    horizon = int(source_cfg.get("target_horizon", 5))
    processed_dir = (
        data_cfg["data"]["processed_dir"]
        if source_name == "crypto"
        else source_cfg.get("panel_dir", data_cfg["data"]["processed_dir"])
    )

    panel = _load_local_panel(processed_dir, data_config=args.data_config)
    splits = create_data_splits(
        processed_dir,
        args.data_config,
        max_backtrack_days=100,
        max_future_days=max(5, horizon),
    )
    split_data = splits[args.split]
    target_expr = Ref(Feature(FeatureType.CLOSE), -horizon) / Feature(FeatureType.CLOSE) - 1
    calculator = PanelAlphaCalculator(split_data, target_expr)
    date_index = split_data.make_dataframe(calculator.target, ["target"]).index.levels[0]
    close_df = panel["close"].loc[date_index, split_data.stock_ids].astype(float)
    forward_returns = build_forward_returns(close_df, horizon)

    exprs, weights = load_pool(args.pool)
    weights = normalize_weights(weights)
    gate = ValidationGate(
        ValidationConfig(
            min_rank_ic=args.min_rank_ic,
            min_rank_icir=args.min_rank_icir,
            max_turnover=0.50,
            min_decay_halflife=2,
            max_pool_correlation=args.max_correlation,
            min_judge_score=0.0,
        )
    )

    rows: list[dict] = []
    factor_map: dict[str, pd.DataFrame] = {}
    for idx, (expr_str, weight) in enumerate(zip(exprs, weights), start=1):
        expr = parse_expression(expr_str)
        factor_df = tensor_to_factor_frame(split_data, calculator.evaluate_alpha(expr), f"factor_{idx:02d}")
        factor_map[f"f{idx:02d}"] = factor_df
        result = gate.validate(factor_df, forward_returns, existing_pool=[])
        noisy_result = gate.validate(_noise_stress(factor_df, sigma=args.noise_sigma), forward_returns, existing_pool=[])
        rows.append(
            {
                "expression": expr_str,
                "weight": weight,
                "rank_ic": result.metrics.get("rank_ic", 0.0),
                "rank_icir": result.metrics.get("rank_icir", 0.0),
                "turnover": result.metrics.get("turnover", 0.0),
                "decay_halflife": result.metrics.get("decay_halflife", 0),
                "noise_rank_ic": noisy_result.metrics.get("rank_ic", 0.0),
                "noise_rank_icir": noisy_result.metrics.get("rank_icir", 0.0),
                "noise_drop": result.metrics.get("rank_ic", 0.0) - noisy_result.metrics.get("rank_ic", 0.0),
                "passed": bool(
                    abs(result.metrics.get("rank_ic", 0.0)) >= args.min_rank_ic
                    and abs(result.metrics.get("rank_icir", 0.0)) >= args.min_rank_icir
                ),
            }
        )

    corr = compute_factor_correlation(factor_map)
    redundant = find_redundant_pairs(corr, threshold=args.max_correlation)
    rows_df = pd.DataFrame(rows).sort_values(
        ["passed", "rank_ic", "rank_icir"],
        ascending=[False, False, False],
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_df.to_csv(out_dir / "robustness_audit.csv", index=False)
    corr.to_csv(out_dir / "factor_correlation.csv")
    with (out_dir / "robustness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_factors": len(rows),
                "n_passed": int(rows_df["passed"].sum()) if not rows_df.empty else 0,
                "redundant_pairs": redundant,
                "max_correlation": float(corr.values[np.triu_indices_from(corr.values, k=1)].max()) if len(corr) > 1 else 0.0,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("Robustness audit written to {}", out_dir)


if __name__ == "__main__":
    main()
