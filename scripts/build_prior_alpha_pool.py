"""Build a prior-guided alpha pool from canonical factor templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]


PRIOR_TEMPLATES = [
    {"name": "medium_term_momentum_close", "expr": "WMA(Rank(Delta($close,20d),20d),10d)", "weight": 1.0},
    {"name": "medium_term_momentum_vwap", "expr": "WMA(Rank(Delta($vwap,20d),20d),10d)", "weight": 1.0},
    {"name": "short_term_reversal_vwap", "expr": "WMA(Mul(-1.0,Rank(Delta($vwap,10d),10d)),5d)", "weight": 1.0},
    {"name": "short_term_reversal_open", "expr": "WMA(Mul(-1.0,Rank(Delta($open,5d),5d)),5d)", "weight": 1.0},
    {"name": "volume_acceleration", "expr": "WMA(Rank(Delta($volume,20d),20d),10d)", "weight": 1.0},
    {"name": "volume_trend", "expr": "Rank(Mean($volume,20d),20d)", "weight": 1.0},
    {"name": "amihud_proxy", "expr": "WMA(Mul(-1.0,Rank(Div($close,$volume),20d)),5d)", "weight": 1.0},
    {"name": "low_volatility_close", "expr": "WMA(Mul(-1.0,Rank(Std($close,20d),20d)),10d)", "weight": 1.0},
    {"name": "low_range_volatility", "expr": "WMA(Mul(-1.0,Rank(Std(Sub($high,$low),20d),20d)),10d)", "weight": 1.0},
    {"name": "range_mean_reversion", "expr": "Mul(-1.0,Rank(Mean(Sub($high,$low),10d),10d))", "weight": 1.0},
    {"name": "price_vs_vwap_gap", "expr": "WMA(Mul(-1.0,Rank(Div($close,$vwap),10d)),5d)", "weight": 1.0},
    {"name": "intraday_range_stability", "expr": "Mul(-1.0,Rank(Std(Div(Sub($high,$low),$close),20d),20d))", "weight": 1.0},
    {"name": "price_volume_corr_reversal", "expr": "Mul(-1.0,Rank(Corr($close,$volume,20d),20d))", "weight": 1.0},
    {"name": "vwap_volume_corr_reversal", "expr": "Mul(-1.0,Rank(Corr($vwap,$volume,20d),20d))", "weight": 1.0},
    {"name": "range_volume_corr", "expr": "Mul(-1.0,Rank(Corr(Sub($high,$low),$volume,20d),20d))", "weight": 1.0},
    {"name": "return_autocorr_reversal", "expr": "Mul(-1.0,Rank(Corr(Delta($close,5d),Delta($close,1d),20d),20d))", "weight": 1.0},
    {"name": "volume_volatility_interaction", "expr": "Mul(-1.0,Rank(Mul(Std($close,20d),Std($volume,20d)),20d))", "weight": 1.0},
    {"name": "price_volume_ratio_stability", "expr": "Mul(-1.0,Rank(Std(Div($close,Mean($volume,20d)),20d),20d))", "weight": 1.0},
    {"name": "vwap_pullback_with_volume", "expr": "Mul(-1.0,Rank(Mul(Delta($vwap,10d),Delta($volume,10d)),10d))", "weight": 1.0},
    {"name": "short_reversal_low_vol", "expr": "Mul(-1.0,Rank(Mul(Delta($close,5d),Std($close,20d)),10d))", "weight": 1.0},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a prior-guided alpha pool")
    parser.add_argument("--output", default="data/factors/prior_alpha_pool.json")
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    pool = {
        "exprs": [item["expr"] for item in PRIOR_TEMPLATES[: args.top_k]],
        "weights": [item["weight"] for item in PRIOR_TEMPLATES[: args.top_k]],
        "templates": PRIOR_TEMPLATES[: args.top_k],
        "source": "manual_prior_templates",
        "notes": "Prior templates emphasize momentum, reversal, low-volatility, and liquidity structures that are later neutralized and decayed in the BRAIN rewrite stage.",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pool, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Prior alpha pool written to {}", out_path)


if __name__ == "__main__":
    main()
