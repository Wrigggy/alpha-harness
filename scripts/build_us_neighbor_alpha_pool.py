"""Build a narrow US neighbor pool around already accepted short expressions."""

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
        ("vol_ratio_6_2", "accepted_volume_shock", "WMA(Mul(-1.0,Rank(Div($volume,Mean($volume,6d)),6d)),2d)"),
        ("vol_ratio_8_2", "accepted_volume_shock", "WMA(Mul(-1.0,Rank(Div($volume,Mean($volume,8d)),8d)),2d)"),
        ("vol_ratio_8_3", "accepted_volume_shock", "WMA(Mul(-1.0,Rank(Div($volume,Mean($volume,8d)),8d)),3d)"),
        ("vol_ratio_10_2", "accepted_volume_shock", "WMA(Mul(-1.0,Rank(Div($volume,Mean($volume,10d)),10d)),2d)"),
        ("vol_ratio_10_3", "accepted_volume_shock", "WMA(Mul(-1.0,Rank(Div($volume,Mean($volume,10d)),10d)),3d)"),
        ("vol_ratio_12_3", "accepted_volume_shock", "WMA(Mul(-1.0,Rank(Div($volume,Mean($volume,12d)),12d)),3d)"),
        ("range_std_6_2", "accepted_range_compress", "Mul(-1.0,WMA(Mul(-1.0,Rank(Std(Div(Sub($high,$low),$close),6d),6d)),2d))"),
        ("range_std_8_2", "accepted_range_compress", "Mul(-1.0,WMA(Mul(-1.0,Rank(Std(Div(Sub($high,$low),$close),8d),8d)),2d))"),
        ("range_std_8_3", "accepted_range_compress", "Mul(-1.0,WMA(Mul(-1.0,Rank(Std(Div(Sub($high,$low),$close),8d),8d)),3d))"),
        ("range_std_10_2", "accepted_range_compress", "Mul(-1.0,WMA(Mul(-1.0,Rank(Std(Div(Sub($high,$low),$close),10d),10d)),2d))"),
        ("range_std_10_3", "accepted_range_compress", "Mul(-1.0,WMA(Mul(-1.0,Rank(Std(Div(Sub($high,$low),$close),10d),10d)),3d))"),
        ("vwap_liq_std_8_2", "nearby_liquidity", "Mul(-1.0,WMA(Mul(-1.0,Rank(Std(Div($vwap,Mean($volume,8d)),8d),8d)),2d))"),
        ("close_liq_std_8_2", "nearby_liquidity", "Mul(-1.0,WMA(Mul(-1.0,Rank(Std(Div($close,Mean($volume,8d)),8d),8d)),2d))"),
        ("ret_vol_corr_6_2", "nearby_interaction", "WMA(Mul(-1.0,Rank(Corr(Div(Sub($close,$open),$open),Delta($volume,1d),6d),6d)),2d)"),
        ("ret_vol_corr_8_2", "nearby_interaction", "WMA(Mul(-1.0,Rank(Corr(Div(Sub($close,$open),$open),Delta($volume,1d),8d),8d)),2d)"),
    ]
    return _dedupe(
        [{"name": name, "family": family, "expr": expr, "weight": 1.0} for name, family, expr in templates]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build US neighbor alpha pool")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    templates = build_templates()
    payload = {
        "exprs": [item["expr"] for item in templates],
        "weights": [item["weight"] for item in templates],
        "templates": templates,
        "source": "us_neighbor_search_from_accepted",
        "notes": "Narrow neighborhood search around accepted US volume shock and range compression branches.",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("US neighbor alpha pool written to {}", out_path)


if __name__ == "__main__":
    main()
