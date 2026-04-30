"""Fetch free A-share daily data via AkShare and build a local equity panel.

This script targets a practical local-research workflow:
1. Pull a stock universe, optionally capped to the top N liquid names
2. Download daily OHLCV history per symbol
3. Normalize to the repo's local panel format

Example:
    python scripts/fetch_equity_akshare.py ^
      --start 2019-01-01 --end 2024-12-31 ^
      --top-n 300 --output data/equity_panel/cn_top300_daily
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


def _load_akshare():
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError(
            "akshare is not installed. Run: pip install akshare"
        ) from exc
    return ak


def get_universe(top_n: int | None = None, retries: int = 3, retry_sleep: float = 2.0) -> pd.DataFrame:
    ak = _load_akshare()
    last_error: Exception | None = None
    spot = None
    for attempt in range(1, retries + 1):
        try:
            spot = ak.stock_zh_a_spot_em()
            break
        except Exception as exc:
            last_error = exc
            logger.warning("Universe fetch attempt {}/{} failed: {}", attempt, retries, exc)
            if attempt < retries:
                time.sleep(retry_sleep)
    if spot is None:
        raise RuntimeError(f"Failed to fetch A-share universe from AkShare: {last_error}")
    spot = spot.rename(
        columns={
            "代码": "symbol",
            "名称": "name",
            "成交量": "volume",
            "成交额": "amount",
            "总市值": "market_cap",
        }
    )
    cols = [col for col in ["symbol", "name", "volume", "amount", "market_cap"] if col in spot.columns]
    universe = spot[cols].copy()
    if "amount" in universe.columns:
        universe["amount"] = pd.to_numeric(universe["amount"], errors="coerce")
        universe = universe.sort_values("amount", ascending=False)
    if top_n is not None:
        universe = universe.head(top_n)
    universe["symbol"] = universe["symbol"].astype(str).str.zfill(6)
    return universe.reset_index(drop=True)


def fetch_one_symbol(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    ak = _load_akshare()
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start.replace("-", ""),
        end_date=end.replace("-", ""),
        adjust=adjust,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(
        columns={
            "日期": "date",
            "股票代码": "symbol",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
        }
    )
    if "symbol" not in df.columns:
        df["symbol"] = symbol

    keep_cols = [col for col in ["date", "symbol", "open", "high", "low", "close", "volume", "amount", "turnover"] if col in df.columns]
    return df[keep_cols].copy()


def fetch_history(
    symbols: list[str],
    start: str,
    end: str,
    adjust: str,
    sleep: float,
    symbol_retries: int = 3,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for idx, symbol in enumerate(symbols, start=1):
        hist = pd.DataFrame()
        last_error: Exception | None = None
        for attempt in range(1, symbol_retries + 1):
            try:
                hist = fetch_one_symbol(symbol, start, end, adjust)
                logger.info(
                    "Fetched {}/{} attempt {}/{}: {} rows for {}",
                    idx, len(symbols), attempt, symbol_retries, len(hist), symbol,
                )
                if not hist.empty:
                    frames.append(hist)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Fetch failed for {} attempt {}/{}: {}",
                    symbol, attempt, symbol_retries, exc,
                )
                time.sleep(max(sleep, 0.2))
        if last_error is not None:
            logger.warning("Giving up on {} after {} attempts", symbol, symbol_retries)
        if sleep > 0:
            time.sleep(sleep)

    if not frames:
        raise RuntimeError("No history downloaded from AkShare")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch free A-share daily data via AkShare")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--top-n", type=int, default=300, help="Top N liquid A-shares by turnover amount")
    parser.add_argument("--symbols", default=None, help="Comma-separated stock codes; bypass universe fetch")
    parser.add_argument("--adjust", default="qfq", choices=["", "qfq", "hfq"], help="AkShare adjust flag")
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds between symbol requests")
    parser.add_argument("--symbol-retries", type=int, default=3, help="Retries per symbol request")
    parser.add_argument("--output", default="data/equity_panel/cn_top300_daily")
    parser.add_argument("--raw-output", default=None, help="Optional long-form parquet output path")
    args = parser.parse_args()

    if args.symbols:
        symbols = [item.strip().zfill(6) for item in args.symbols.split(",") if item.strip()]
        logger.info("Using explicit symbol list: {}", symbols)
    else:
        universe = get_universe(top_n=args.top_n)
        symbols = universe["symbol"].tolist()
        logger.info("Universe size: {}", len(symbols))

    raw = fetch_history(
        symbols=symbols,
        start=args.start,
        end=args.end,
        adjust=args.adjust,
        sleep=args.sleep,
        symbol_retries=args.symbol_retries,
    )
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
        "source": "akshare.stock_zh_a_hist",
        "market": "cn",
        "universe": f"top_{len(close.columns)}_a_share_by_amount",
        "date_range": [str(close.index.min().date()), str(close.index.max().date())],
        "n_symbols": int(len(close.columns)),
        "fields": list(panel.keys()),
        "adjust": args.adjust,
    }
    save_panel(panel, Path(args.output), metadata)
    logger.info("Saved panel to {}", args.output)


if __name__ == "__main__":
    main()
