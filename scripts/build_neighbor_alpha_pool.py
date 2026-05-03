"""Build a neighborhood search pool around already-accepted factor families."""

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


def _keep_short(items: list[dict], max_len: int = 120) -> list[dict]:
    return [item for item in items if len(item["expr"]) <= max_len]


def build_templates() -> list[dict]:
    items: list[dict] = []

    templates = [
        ("liq_close_10", "accepted_liquidity_stability", "WMA(Mul(-1.0,Rank(Std(Div($close,Mean($volume,10d)),10d),10d)),3d)"),
        ("liq_close_20", "accepted_liquidity_stability", "WMA(Mul(-1.0,Rank(Std(Div($close,Mean($volume,20d)),20d),20d)),3d)"),
        ("liq_vwap_10", "accepted_liquidity_stability_vwap", "WMA(Mul(-1.0,Rank(Std(Div($vwap,Mean($volume,10d)),10d),10d)),3d)"),
        ("liq_vwap_20", "accepted_liquidity_stability_vwap", "WMA(Mul(-1.0,Rank(Std(Div($vwap,Mean($volume,20d)),20d),20d)),3d)"),
        ("range_10", "low_volatility", "WMA(Mul(-1.0,Rank(Std(Div(Sub($high,$low),$close),10d),10d)),3d)"),
        ("range_20", "low_volatility", "WMA(Mul(-1.0,Rank(Std(Div(Sub($high,$low),$close),20d),20d)),3d)"),
        ("corr_close_vol", "price_volume_interaction", "WMA(Mul(-1.0,Rank(Corr(Delta($close,1d),Delta($volume,1d),10d),10d)),3d)"),
        ("corr_vwap_vol", "price_volume_interaction", "WMA(Mul(-1.0,Rank(Corr(Delta($vwap,1d),Delta($volume,1d),10d),10d)),3d)"),
        ("open_rev_3", "open_reversal", "WMA(Mul(-1.0,Rank(Delta($open,3d),3d)),3d)"),
        ("vwap_rev_3", "vwap_reversal", "WMA(Mul(-1.0,Rank(Delta($vwap,3d),3d)),3d)"),
    ]
    for name, family, expr in templates:
        items.append({"name": name, "family": family, "expr": expr, "weight": 1.0})

    combo_templates = [
        ("mix_liq_range", "liquidity_range_combo", "WMA(Add(WMA(Mul(-1.0,Rank(Std(Div($close,Mean($volume,10d)),10d),10d)),3d),WMA(Mul(-1.0,Rank(Std(Div(Sub($high,$low),$close),10d),10d)),3d)),3d)"),
        ("mix_liq_corr", "liquidity_corr_combo", "WMA(Add(WMA(Mul(-1.0,Rank(Std(Div($close,Mean($volume,10d)),10d),10d)),3d),WMA(Mul(-1.0,Rank(Corr(Delta($close,1d),Delta($volume,1d),10d),10d)),3d)),3d)"),
    ]
    for name, family, expr in combo_templates:
        items.append({"name": name, "family": family, "expr": expr, "weight": 1.0})

    return _keep_short(_dedupe(items), max_len=120)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build neighborhood search alpha pool")
    parser.add_argument("--output", default="data/factors/neighbor_alpha_pool.json")
    args = parser.parse_args()

    templates = build_templates()
    payload = {
        "exprs": [item["expr"] for item in templates],
        "weights": [item["weight"] for item in templates],
        "templates": templates,
        "source": "accepted_family_neighborhood_search",
        "notes": "Neighborhood search around accepted liquidity-stability family and nearby low-frequency combinations.",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Neighbor alpha pool written to {}", out_path)


if __name__ == "__main__":
    main()
