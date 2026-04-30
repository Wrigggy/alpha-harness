"""Fetch the latest CSI 500 constituents and optional weights via AkShare."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError("akshare is not installed. Run: pip install akshare") from exc
    return ak


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the latest CSI 500 constituents via AkShare")
    parser.add_argument("--index", default="000905", help="CSI index code, default=000905 (CSI500)")
    parser.add_argument("--output", default="data/raw_equity/csi500_constituents_latest.parquet")
    parser.add_argument("--with-weights", action="store_true", default=False, help="Fetch weights snapshot instead of plain constituents")
    args = parser.parse_args()

    ak = _load_akshare()
    if args.with_weights:
        df = ak.index_stock_cons_weight_csindex(symbol=args.index)
        df = df.rename(
            columns={
                "日期": "date",
                "指数代码": "index_code",
                "指数名称": "index_name",
                "成分券代码": "symbol",
                "成分券名称": "name",
                "权重": "weight",
            }
        )
    else:
        df = ak.index_stock_cons_csindex(symbol=args.index)
        df = df.rename(
            columns={
                "日期": "date",
                "指数代码": "index_code",
                "指数名称": "index_name",
                "成分券代码": "symbol",
                "成分券名称": "name",
            }
        )

    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        df.to_parquet(output_path, index=False)

    logger.info("Saved {} rows to {}", len(df), output_path)
    if "date" in df.columns and len(df) > 0:
        logger.info("Snapshot date: {}", df["date"].iloc[0])


if __name__ == "__main__":
    main()
