"""Soft-filter an AlphaGen pool while preserving multi-factor combinations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from src.evaluation.validation_gate import ValidationConfig, ValidationGate
from src.pipeline import load_pool, normalize_weights, tensor_to_factor_frame


def load_data_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_forward_returns(close_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return close_df.shift(-horizon) / close_df - 1.0


def soft_filter_pool(
    pool_path: str,
    data_config_path: str,
    split: str,
    output_path: str,
    min_rank_ic: float,
    min_rank_icir: float,
    max_turnover: float,
    min_decay_halflife: int,
    max_correlation: float,
    keep_top_k: int,
) -> dict:
    data_cfg = load_data_config(data_config_path)
    source_cfg = data_cfg.get("source", {})
    source_name = source_cfg.get("name", "crypto")
    horizon = int(source_cfg.get("target_horizon", 5))
    processed_dir = (
        data_cfg["data"]["processed_dir"]
        if source_name == "crypto"
        else source_cfg.get("panel_dir", data_cfg["data"]["processed_dir"])
    )

    panel = _load_local_panel(processed_dir, data_config=data_config_path)
    splits = create_data_splits(
        processed_dir,
        data_config_path,
        max_backtrack_days=100,
        max_future_days=max(5, horizon),
    )
    stock_data = splits[split]

    expr_strings, raw_weights = load_pool(pool_path)
    weights = normalize_weights(raw_weights)
    parsed_exprs = [parse_expression(expr) for expr in expr_strings]

    target_expr = Ref(Feature(FeatureType.CLOSE), -horizon) / Feature(FeatureType.CLOSE) - 1
    calculator = PanelAlphaCalculator(stock_data, target_expr)
    date_index = stock_data.make_dataframe(calculator.target, ["target"]).index.levels[0]
    close_df = panel["close"].loc[date_index, stock_data.stock_ids].astype(float)
    forward_returns = build_forward_returns(close_df, horizon)

    gate = ValidationGate(
        ValidationConfig(
            min_rank_ic=min_rank_ic,
            min_rank_icir=min_rank_icir,
            max_turnover=max_turnover,
            min_decay_halflife=min_decay_halflife,
            max_pool_correlation=max_correlation,
            min_judge_score=0.0,
        )
    )

    candidates: list[dict] = []
    for expr_str, expr, weight in zip(expr_strings, parsed_exprs, weights):
        factor_df = tensor_to_factor_frame(stock_data, calculator.evaluate_alpha(expr), "factor")
        result = gate.validate(factor_df, forward_returns, existing_pool=[])
        score = abs(result.metrics.get("rank_ic", 0.0)) + 0.5 * abs(result.metrics.get("rank_icir", 0.0))
        candidates.append(
            {
                "expression": expr_str,
                "weight": weight,
                "frame": factor_df,
                "passed": result.passed,
                "score": score,
                **result.metrics,
                "failures": "; ".join(result.failures),
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)

    selected: list[dict] = []
    selected_frames: list[pd.DataFrame] = []

    for item in candidates:
        corr_result = gate.validate(
            item["frame"],
            forward_returns,
            existing_pool=selected_frames,
        )
        corr_ok = corr_result.metrics.get("max_pool_correlation", 0.0) <= max_correlation
        quality_ok = (
            abs(item.get("rank_ic", 0.0)) >= min_rank_ic
            and abs(item.get("rank_icir", 0.0)) >= min_rank_icir
            and item.get("turnover", 1.0) <= max_turnover
            and item.get("decay_halflife", 0) >= min_decay_halflife
        )
        if quality_ok and corr_ok:
            selected.append(item)
            selected_frames.append(item["frame"])
        if len(selected) >= keep_top_k:
            break

    if not selected:
        logger.warning("No factor satisfied soft-filter thresholds; falling back to top {} by score", keep_top_k)
        selected = candidates[:keep_top_k]

    output = {
        "exprs": [item["expression"] for item in selected],
        "weights": [item["weight"] for item in selected],
        "source_pool": pool_path,
        "filter_split": split,
        "filter_mode": "soft",
        "thresholds": {
            "min_rank_ic": min_rank_ic,
            "min_rank_icir": min_rank_icir,
            "max_turnover": max_turnover,
            "min_decay_halflife": min_decay_halflife,
            "max_correlation": max_correlation,
            "keep_top_k": keep_top_k,
        },
        "n_input": len(expr_strings),
        "n_accepted": len(selected),
    }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    review = pd.DataFrame(
        [
            {
                k: v
                for k, v in item.items()
                if k not in {"frame"}
            }
            for item in candidates
        ]
    )
    review_path = out_path.with_name(f"{out_path.stem}_review.csv")
    review.to_csv(review_path, index=False)
    logger.info("Soft-filtered pool saved to {}", out_path)
    logger.info("Review CSV saved to {}", review_path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Soft-filter an AlphaGen pool")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-rank-ic", type=float, default=0.015)
    parser.add_argument("--min-rank-icir", type=float, default=0.15)
    parser.add_argument("--max-turnover", type=float, default=0.5)
    parser.add_argument("--min-decay-halflife", type=int, default=2)
    parser.add_argument("--max-correlation", type=float, default=0.7)
    parser.add_argument("--keep-top-k", type=int, default=3)
    args = parser.parse_args()

    soft_filter_pool(
        pool_path=args.pool,
        data_config_path=args.data_config,
        split=args.split,
        output_path=args.output,
        min_rank_ic=args.min_rank_ic,
        min_rank_icir=args.min_rank_icir,
        max_turnover=args.max_turnover,
        min_decay_halflife=args.min_decay_halflife,
        max_correlation=args.max_correlation,
        keep_top_k=args.keep_top_k,
    )


if __name__ == "__main__":
    main()
