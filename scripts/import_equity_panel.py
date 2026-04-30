"""Import daily equity OHLCV data into the repo's local panel format.

Accepted input layout:
1. A single parquet/csv file with columns:
   date, symbol, open, high, low, close, volume
2. Optional extra columns:
   amount, turnover, vwap, market_cap, industry

Example:
    python scripts/import_equity_panel.py ^
      --input data/raw_equity/csi500_daily.parquet ^
      --output data/equity_panel/csi500_daily ^
      --market cn --universe csi500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_collection.equity_panel_builder import (
    build_panel,
    normalize_frame,
    save_panel,
    validate_columns,
)


def read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input format: {path}")
    return df
def main() -> None:
    parser = argparse.ArgumentParser(description="Import daily equity data into local panel format")
    parser.add_argument("--input", required=True, help="Input parquet/csv path")
    parser.add_argument("--output", default="data/equity_panel/csi500_daily", help="Output panel directory")
    parser.add_argument("--market", default="cn", help="Market label, e.g. cn/us/hk")
    parser.add_argument("--universe", default="csi500", help="Universe label")
    parser.add_argument("--frequency", default="1d", help="Sampling frequency label")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    df = read_input(input_path)
    validate_columns(df)
    frame = normalize_frame(df)
    panel = build_panel(frame)

    close = panel["close"]
    metadata = {
        "asset_class": "equity",
        "frequency": args.frequency,
        "source": str(input_path),
        "market": args.market,
        "universe": args.universe,
        "date_range": [str(close.index.min().date()), str(close.index.max().date())],
        "n_symbols": int(len(close.columns)),
        "fields": list(panel.keys()),
    }
    save_panel(panel, output_dir, metadata)

    logger.info(
        "Imported equity panel: {} symbols, {} dates, fields={}",
        len(close.columns),
        len(close.index),
        list(panel.keys()),
    )
    logger.info("Saved to {}", output_dir)


if __name__ == "__main__":
    main()
