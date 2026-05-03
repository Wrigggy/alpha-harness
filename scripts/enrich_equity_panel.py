"""Enrich an existing local equity panel with proxy metadata fields.

Offline-first enrichments:
- `market_cap` from `amount / (turnover / 100)` when direct market cap is absent
- `vwap` from `amount / volume`
- `board` from symbol prefix / exchange style buckets
- optional `industry` from a user-supplied symbol mapping file

This keeps the workflow usable when live metadata sources are unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_adapter.to_alphagen_format import _load_local_panel


def infer_board(symbol: str) -> str:
    raw = str(symbol).upper().replace(".SH", "").replace(".SZ", "")
    if raw.startswith("688"):
        return "STAR"
    if raw.startswith("300"):
        return "CHINEXT"
    if raw.startswith(("600", "601", "603", "605")):
        return "SSE_MAIN"
    if raw.startswith("002"):
        return "SZSE_SME"
    if raw.startswith(("000", "001", "003")):
        return "SZSE_MAIN"
    if raw.startswith(("200", "900")):
        return "B_SHARE"
    return "OTHER"


def build_board_frame(index: pd.Index, columns: pd.Index) -> pd.DataFrame:
    labels = {symbol: infer_board(symbol) for symbol in columns}
    row = pd.Series(labels)
    frame = pd.DataFrame([row.values] * len(index), index=index, columns=columns)
    return frame.astype(str)


def load_industry_map(path: str | None) -> pd.Series | None:
    if path is None:
        return None
    mapping_path = Path(path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"Industry map not found: {mapping_path}")

    if mapping_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(mapping_path)
    else:
        df = pd.read_csv(mapping_path)

    required = {"symbol", "industry"}
    if not required.issubset(df.columns):
        raise ValueError(f"Industry map must contain columns: {sorted(required)}")
    series = df.assign(symbol=df["symbol"].astype(str)) \
        .drop_duplicates("symbol", keep="last") \
        .set_index("symbol")["industry"] \
        .astype(str)
    return series


def write_field(df: pd.DataFrame, output_dir: Path, field: str) -> None:
    df.to_parquet(output_dir / f"{field}.parquet")
    logger.info("Wrote field {} to {}", field, output_dir / f"{field}.parquet")


def update_metadata(output_dir: Path, additions: dict) -> None:
    meta_path = output_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(additions)
    if "fields" in meta:
        meta["fields"] = sorted(set(meta["fields"]).union(additions.get("fields_added", [])))
    if "fields_added" in meta:
        meta.pop("fields_added", None)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def compute_market_cap_proxy(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "market_cap" in panel:
        return panel["market_cap"]
    if "amount" not in panel or "turnover" not in panel:
        raise KeyError("Need amount and turnover to infer market_cap")
    amount = panel["amount"].astype(float)
    turnover = panel["turnover"].astype(float).replace(0.0, np.nan)
    market_cap = amount / (turnover / 100.0)
    market_cap = market_cap.replace([np.inf, -np.inf], np.nan)
    return market_cap


def compute_vwap_proxy(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "vwap" in panel:
        return panel["vwap"]
    if "amount" in panel and "volume" in panel:
        volume = panel["volume"].astype(float).replace(0.0, np.nan)
        vwap = panel["amount"].astype(float) / volume
        return vwap.replace([np.inf, -np.inf], np.nan)
    close = panel["close"].astype(float)
    return close.copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich a local equity panel with proxy metadata fields")
    parser.add_argument("--input-panel", required=True)
    parser.add_argument("--output-panel", required=True)
    parser.add_argument("--industry-map", default=None, help="Optional csv/parquet with symbol,industry")
    parser.add_argument("--copy-existing", action="store_true", default=False, help="Copy all current fields into output")
    args = parser.parse_args()

    input_dir = Path(args.input_panel)
    output_dir = Path(args.output_panel)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = _load_local_panel(str(input_dir))
    if "close" not in panel:
        raise ValueError(f"Invalid panel: missing close field in {input_dir}")

    if args.copy_existing:
        for field, matrix in panel.items():
            write_field(matrix, output_dir, field)
        if (input_dir / "symbols.csv").exists():
            (output_dir / "symbols.csv").write_text((input_dir / "symbols.csv").read_text(encoding="utf-8"), encoding="utf-8")

    market_cap = compute_market_cap_proxy(panel)
    vwap = compute_vwap_proxy(panel)
    board = build_board_frame(panel["close"].index, panel["close"].columns)

    write_field(market_cap, output_dir, "market_cap")
    write_field(vwap, output_dir, "vwap")
    write_field(board, output_dir, "board")

    industry_source = "none"
    industry_map = load_industry_map(args.industry_map)
    if industry_map is not None:
        row = industry_map.reindex(panel["close"].columns).fillna("UNKNOWN")
        industry = pd.DataFrame([row.values] * len(panel["close"].index), index=panel["close"].index, columns=panel["close"].columns)
        write_field(industry.astype(str), output_dir, "industry")
        industry_source = str(Path(args.industry_map))

    if (input_dir / "meta.json").exists() and not (output_dir / "meta.json").exists():
        (output_dir / "meta.json").write_text((input_dir / "meta.json").read_text(encoding="utf-8"), encoding="utf-8")
    if (input_dir / "symbols.csv").exists() and not (output_dir / "symbols.csv").exists():
        (output_dir / "symbols.csv").write_text((input_dir / "symbols.csv").read_text(encoding="utf-8"), encoding="utf-8")

    update_metadata(
        output_dir,
        {
            "enriched_from": str(input_dir),
            "market_cap_source": "amount_div_turnover_proxy",
            "vwap_source": "amount_div_volume_proxy",
            "board_source": "symbol_prefix_rule",
            "industry_source": industry_source,
            "fields_added": ["market_cap", "vwap", "board"] + (["industry"] if industry_map is not None else []),
        },
    )
    logger.info("Enriched panel written to {}", output_dir)


if __name__ == "__main__":
    main()
