"""Fetch a broad free US stock symbol universe from Nasdaq Trader symbol directories."""

from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from loguru import logger


NASDAQ_TRADED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def _read_pipe_text(url: str) -> pd.DataFrame:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    text = response.text
    lines = [line for line in text.splitlines() if line and not line.startswith("File Creation Time")]
    return pd.read_csv(StringIO("\n".join(lines)), sep="|")


def _clean_symbol_frame(df: pd.DataFrame, symbol_col: str, exchange: str) -> pd.DataFrame:
    out = df.copy()
    out["symbol"] = out[symbol_col].astype(str).str.upper().str.strip()
    out["exchange"] = exchange
    out = out[~out["symbol"].isin(["", "NAN", "NONE"])]
    out = out[~out["symbol"].str.contains(r"[\$\^/ ]", regex=True)]
    out = out[~out["symbol"].str.endswith("W")]
    out = out[~out["symbol"].str.endswith("R")]
    out = out[~out["symbol"].str.endswith("P")]
    return out


def build_universe(include_etf: bool = False) -> pd.DataFrame:
    nasdaq = _read_pipe_text(NASDAQ_TRADED_URL)
    other = _read_pipe_text(OTHER_LISTED_URL)

    nasdaq = _clean_symbol_frame(nasdaq, "Symbol", "NASDAQ")
    other = _clean_symbol_frame(other, "ACT Symbol", "OTHER")

    if "Test Issue" in nasdaq.columns:
        nasdaq = nasdaq[nasdaq["Test Issue"].astype(str).str.upper() != "Y"]
    if "ETF" in nasdaq.columns and not include_etf:
        nasdaq = nasdaq[nasdaq["ETF"].astype(str).str.upper() != "Y"]
    if "Financial Status" in nasdaq.columns:
        nasdaq = nasdaq[nasdaq["Financial Status"].astype(str).fillna("").ne("D")]

    if "Test Issue" in other.columns:
        other = other[other["Test Issue"].astype(str).str.upper() != "Y"]
    if "ETF" in other.columns and not include_etf:
        other = other[other["ETF"].astype(str).str.upper() != "Y"]

    merged = pd.concat(
        [
            nasdaq[["symbol", "exchange"]],
            other[["symbol", "exchange"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["symbol"])
    merged = merged.sort_values("symbol").reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch broad US stock symbol universe")
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-etf", action="store_true")
    args = parser.parse_args()

    universe = build_universe(include_etf=args.include_etf)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        universe.to_parquet(out_path, index=False)
    else:
        universe.to_csv(out_path, index=False)
    logger.info("Saved {} US symbols to {}", len(universe), out_path)


if __name__ == "__main__":
    main()
