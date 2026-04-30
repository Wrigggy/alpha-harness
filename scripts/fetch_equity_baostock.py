"""Fetch free A-share daily data via BaoStock and build a local equity panel.

BaoStock is slower than AkShare but usually more stable for repeated daily
history requests. It requires explicit symbol lists in BaoStock format:
    sh.600519, sz.000858, ...
"""

from __future__ import annotations

import argparse
import sys
import time
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
)


def _load_baostock():
    try:
        import baostock as bs
    except ImportError as exc:
        raise ImportError("baostock is not installed. Run: pip install baostock") from exc
    return bs


def normalize_symbol(symbol: str) -> str:
    raw = symbol.strip().lower()
    if raw.startswith(("sh.", "sz.")):
        return raw
    if raw.startswith(("6", "9")):
        return f"sh.{raw.zfill(6)}"
    return f"sz.{raw.zfill(6)}"


def fetch_one_symbol(bs, symbol: str, start: str, end: str, adjustflag: str) -> pd.DataFrame:
    rs = bs.query_history_k_data_plus(
        symbol,
        "date,code,open,high,low,close,volume,amount,turn",
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag=adjustflag,
    )
    rows: list[list[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != "0":
        raise RuntimeError(f"{symbol}: {rs.error_msg}")
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=rs.fields)
    df = df.rename(
        columns={
            "date": "date",
            "code": "symbol",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
            "turn": "turnover",
        }
    )
    df["symbol"] = df["symbol"].str.split(".").str[-1]
    return df[["date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover"]]


def fetch_history(symbols: list[str], start: str, end: str, adjustflag: str, sleep: float) -> pd.DataFrame:
    bs = _load_baostock()
    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock login failed: {lg.error_msg}")

    frames: list[pd.DataFrame] = []
    try:
        for idx, symbol in enumerate(symbols, start=1):
            try:
                hist = fetch_one_symbol(bs, symbol, start, end, adjustflag)
                if not hist.empty:
                    frames.append(hist)
                logger.info("Fetched {}/{}: {} rows for {}", idx, len(symbols), len(hist), symbol)
            except Exception as exc:
                logger.warning("Fetch failed for {}: {}", symbol, exc)
            if sleep > 0:
                time.sleep(sleep)
    finally:
        bs.logout()

    if not frames:
        raise RuntimeError("No history downloaded from BaoStock")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch free A-share daily data via BaoStock")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--symbols", required=True, help="Comma-separated stock codes, e.g. 600519,000858")
    parser.add_argument("--adjustflag", default="2", choices=["1", "2", "3"], help="1=hfq 2=qfq 3=raw")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--output", default="data/equity_panel/cn_baostock_daily")
    parser.add_argument("--raw-output", default=None)
    args = parser.parse_args()

    symbols = [normalize_symbol(item) for item in args.symbols.split(",") if item.strip()]
    logger.info("Using BaoStock symbols: {}", symbols)

    raw = fetch_history(symbols, args.start, args.end, args.adjustflag, args.sleep)
    raw = normalize_frame(raw)

    if args.raw_output:
        raw_path = Path(args.raw_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(raw_path, index=False)
        logger.info("Saved long-form history to {}", raw_path)

    panel = build_panel(raw)
    close = panel["close"]
    metadata = {
        "asset_class": "equity",
        "frequency": "1d",
        "source": "baostock.query_history_k_data_plus",
        "market": "cn",
        "universe": f"explicit_{len(close.columns)}_stocks",
        "date_range": [str(close.index.min().date()), str(close.index.max().date())],
        "n_symbols": int(len(close.columns)),
        "fields": list(panel.keys()),
        "adjustflag": args.adjustflag,
    }
    save_panel(panel, Path(args.output), metadata)
    logger.info("Saved panel to {}", args.output)


if __name__ == "__main__":
    main()
