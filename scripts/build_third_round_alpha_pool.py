"""Build a very narrow third-round pool around the strongest accepted branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger


def _dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        if item["expr"] in seen:
            continue
        seen.add(item["expr"])
        out.append(item)
    return out


def _keep_short(items: list[dict], max_len: int = 120) -> list[dict]:
    return [item for item in items if len(item["expr"]) <= max_len]


def build_templates() -> list[dict]:
    items: list[dict] = []

    templates = [
        ("core_close_10", "third_round_core", "WMA(Mul(-1.0,Rank(Std(Div($close,Mean($volume,10d)),10d),10d)),2d)"),
        ("core_close_20", "third_round_core", "WMA(Mul(-1.0,Rank(Std(Div($close,Mean($volume,20d)),20d),20d)),2d)"),
        ("core_vwap_10", "third_round_core", "WMA(Mul(-1.0,Rank(Std(Div($vwap,Mean($volume,10d)),10d),10d)),2d)"),
        ("core_vwap_20", "third_round_core", "WMA(Mul(-1.0,Rank(Std(Div($vwap,Mean($volume,20d)),20d),20d)),2d)"),
        ("core_corr_close", "third_round_add", "WMA(Mul(-1.0,Rank(Corr(Delta($close,1d),Delta($volume,1d),10d),10d)),2d)"),
        ("core_corr_vwap", "third_round_add", "WMA(Mul(-1.0,Rank(Corr(Delta($vwap,1d),Delta($volume,1d),10d),10d)),2d)"),
    ]
    for name, family, expr in templates:
        items.append({"name": name, "family": family, "expr": expr, "weight": 1.0})

    return _keep_short(_dedupe(items), max_len=120)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build third-round alpha pool")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    templates = build_templates()
    payload = {
        "exprs": [item["expr"] for item in templates],
        "weights": [item["weight"] for item in templates],
        "templates": templates,
        "source": "third_round_narrow_search",
        "notes": "Very narrow search around the best accepted liquidity stability branches.",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Third-round alpha pool written to {}", out_path)


if __name__ == "__main__":
    main()
