"""Build a US-specific short-expression alpha pool for daily equity mean reversion and liquidity effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger


def _dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        expr = item["expr"]
        if expr in seen:
            continue
        seen.add(expr)
        out.append(item)
    return out


def build_templates() -> list[dict]:
    templates = [
        ("volume_shock_8_2", "us_liquidity", "WMA(Mul(-1.0,Rank(Div($volume,Mean($volume,8d)),8d)),2d)"),
        ("volume_shock_10_2", "us_liquidity", "WMA(Mul(-1.0,Rank(Div($volume,Mean($volume,10d)),10d)),2d)"),
        ("volume_shock_10_3", "us_liquidity", "WMA(Mul(-1.0,Rank(Div($volume,Mean($volume,10d)),10d)),3d)"),
        ("volume_shock_15_3", "us_liquidity", "WMA(Mul(-1.0,Rank(Div($volume,Mean($volume,15d)),15d)),3d)"),
        ("close_liq_std_8", "us_liquidity", "WMA(Mul(-1.0,Rank(Std(Div($close,Mean($volume,8d)),8d),8d)),2d)"),
        ("close_liq_std_10", "us_liquidity", "WMA(Mul(-1.0,Rank(Std(Div($close,Mean($volume,10d)),10d),10d)),2d)"),
        ("vwap_liq_std_8", "us_liquidity", "WMA(Mul(-1.0,Rank(Std(Div($vwap,Mean($volume,8d)),8d),8d)),2d)"),
        ("vwap_liq_std_10", "us_liquidity", "WMA(Mul(-1.0,Rank(Std(Div($vwap,Mean($volume,10d)),10d),10d)),2d)"),
        ("corr_close_vol_8", "us_interaction", "WMA(Mul(-1.0,Rank(Corr(Delta($close,1d),Delta($volume,1d),8d),8d)),2d)"),
        ("corr_vwap_vol_8", "us_interaction", "WMA(Mul(-1.0,Rank(Corr(Delta($vwap,1d),Delta($volume,1d),8d),8d)),2d)"),
        ("ret_vol_corr_8", "us_interaction", "WMA(Mul(-1.0,Rank(Corr(Div(Sub($close,$open),$open),Delta($volume,1d),8d),8d)),2d)"),
        ("range_compress_8", "us_volatility", "WMA(Mul(-1.0,Rank(Std(Div(Sub($high,$low),$close),8d),8d)),2d)"),
    ]
    return _dedupe(
        [{"name": name, "family": family, "expr": expr, "weight": 1.0} for name, family, expr in templates]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a US-specific short alpha pool")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    templates = build_templates()
    payload = {
        "exprs": [item["expr"] for item in templates],
        "weights": [item["weight"] for item in templates],
        "templates": templates,
        "source": "us_specific_short_alpha_pool",
        "notes": "US-specific short expressions focused on overnight/intraday reversal, gap effects, range compression, and liquidity shock.",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("US short alpha pool written to {}", out_path)


if __name__ == "__main__":
    main()
