"""Build a directed alpha pool from literature-inspired low-frequency templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger


ROOT = Path(__file__).resolve().parents[1]


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


def _keep_short(items: list[dict], max_len: int = 120) -> list[dict]:
    return [item for item in items if len(item["expr"]) <= max_len]


def build_templates() -> list[dict]:
    templates: list[dict] = []

    short_templates = [
        ("close_reversal_3", "short_term_reversal", "WMA(Mul(-1.0,Rank(Delta($close,3d),3d)),3d)"),
        ("close_reversal_5", "short_term_reversal", "WMA(Mul(-1.0,Rank(Delta($close,5d),5d)),5d)"),
        ("vwap_reversal_3", "short_term_reversal", "WMA(Mul(-1.0,Rank(Delta($vwap,3d),3d)),3d)"),
        ("vwap_reversal_5", "short_term_reversal", "WMA(Mul(-1.0,Rank(Delta($vwap,5d),5d)),5d)"),
        ("low_vol_close_10", "low_volatility", "WMA(Mul(-1.0,Rank(Std($close,10d),10d)),3d)"),
        ("low_vol_range_10", "low_volatility", "WMA(Mul(-1.0,Rank(Std(Sub($high,$low),10d),10d)),3d)"),
        ("liquidity_proxy_10", "liquidity_proxy", "WMA(Mul(-1.0,Rank(Std(Div($close,Mean($volume,10d)),10d),10d)),3d)"),
        ("vwap_liquidity_10", "liquidity_proxy", "WMA(Mul(-1.0,Rank(Std(Div($vwap,Mean($volume,10d)),10d),10d)),3d)"),
        ("price_volume_corr_10", "price_volume_interaction", "WMA(Mul(-1.0,Rank(Corr(Delta($close,1d),Delta($volume,1d),10d),10d)),3d)"),
        ("range_volume_corr_10", "price_volume_interaction", "WMA(Mul(-1.0,Rank(Corr(Delta(Sub($high,$low),1d),Delta($volume,1d),10d),10d)),3d)"),
    ]

    for name, family, expr in short_templates:
        templates.append({"name": name, "family": family, "expr": expr, "weight": 1.0})

    mix_templates = [
        ("mix_close_vwap", "directed_combo", "WMA(Add(WMA(Mul(-1.0,Rank(Delta($close,3d),3d)),3d),WMA(Mul(-1.0,Rank(Delta($vwap,3d),3d)),3d)),3d)"),
        ("mix_close_liq", "directed_combo", "WMA(Add(WMA(Mul(-1.0,Rank(Delta($close,3d),3d)),3d),WMA(Mul(-1.0,Rank(Std(Div($close,Mean($volume,10d)),10d),10d)),3d)),3d)"),
    ]
    for name, family, expr in mix_templates:
        templates.append({"name": name, "family": family, "expr": expr, "weight": 1.0})

    return _keep_short(_dedupe(templates), max_len=120)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a directed prior alpha pool")
    parser.add_argument("--output", default="data/factors/directed_alpha_pool.json")
    args = parser.parse_args()

    templates = build_templates()
    payload = {
        "exprs": [item["expr"] for item in templates],
        "weights": [item["weight"] for item in templates],
        "templates": templates,
        "source": "literature_inspired_directed_templates",
        "notes": (
            "Directed pool emphasizes short-term reversal, low-volatility, "
            "illiquidity proxy, and smoothed interaction structures inspired by common "
            "cross-sectional equity factor families used in academic and practitioner research."
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Directed alpha pool written to {}", out_path)


if __name__ == "__main__":
    main()
