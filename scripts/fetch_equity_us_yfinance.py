"""Fetch free US daily equity data via yfinance and convert it into local long-form OHLCV."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / ".cache" / "yfinance"


def configure_yfinance_cache() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(ROOT / ".cache")
    os.environ["YF_CACHE_DIR"] = str(CACHE_DIR)
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(CACHE_DIR))


def load_symbols(path: str | None, symbols_csv: str | None) -> list[str]:
    if symbols_csv:
        return [s.strip().upper() for s in symbols_csv.split(",") if s.strip()]
    if not path:
        raise ValueError("Provide --symbols-file or --symbols")
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    col = "symbol" if "symbol" in df.columns else df.columns[0]
    return df[col].astype(str).str.upper().tolist()


def _fetch_history_batch(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        tickers=symbols,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if df.empty:
        raise RuntimeError("yfinance returned no price history")
    return df


def fetch_history(symbols: list[str], start: str, end: str, batch_size: int) -> pd.DataFrame:
    batches: list[pd.DataFrame] = []
    for idx in range(0, len(symbols), batch_size):
        chunk = symbols[idx : idx + batch_size]
        logger.info("Fetching batch {}/{} ({} symbols)", idx // batch_size + 1, (len(symbols) + batch_size - 1) // batch_size, len(chunk))
        try:
            batches.append(_fetch_history_batch(chunk, start, end))
        except Exception as exc:
            logger.warning("history fetch failed for batch starting at {}: {}", idx, exc)
    if not batches:
        raise RuntimeError("No yfinance batches succeeded")
    if len(batches) == 1:
        return batches[0]
    return pd.concat(batches, axis=1)


def extract_metadata(symbols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        info = {}
        fast = {}
        try:
            info = ticker.info or {}
        except Exception as exc:
            logger.warning("info fetch failed for {}: {}", symbol, exc)
        try:
            fast = dict(ticker.fast_info or {})
        except Exception as exc:
            logger.warning("fast_info fetch failed for {}: {}", symbol, exc)

        shares_outstanding = info.get("sharesOutstanding") or fast.get("shares")
        market_cap = info.get("marketCap") or fast.get("market_cap")
        industry = info.get("industryKey") or info.get("industry") or info.get("sectorKey") or info.get("sector")
        rows.append(
            {
                "symbol": symbol,
                "shares_outstanding": pd.to_numeric(shares_outstanding, errors="coerce"),
                "market_cap_latest": pd.to_numeric(market_cap, errors="coerce"),
                "industry": industry,
            }
        )
    return pd.DataFrame(rows)


def to_long_frame(history: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if isinstance(history.columns, pd.MultiIndex):
        frames: list[pd.DataFrame] = []
        for symbol in history.columns.levels[0]:
            if symbol not in history.columns.get_level_values(0):
                continue
            block = history[symbol].copy()
            if block.empty:
                continue
            block = block.reset_index().rename(columns=str)
            rename_map = {c: c.lower().replace(" ", "_") for c in block.columns}
            block = block.rename(columns=rename_map)
            block["symbol"] = symbol
            frames.append(block)
        df = pd.concat(frames, ignore_index=True)
    else:
        raise ValueError("Expected yfinance multi-index columns for multi-symbol download")

    df = df.rename(columns={"adj_close": "adj_close"})
    keep_cols = [c for c in ["date", "symbol", "open", "high", "low", "close", "volume"] if c in df.columns]
    out = df[keep_cols].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["vwap"] = out[["open", "high", "low", "close"]].mean(axis=1)
    out["amount"] = out["close"] * out["volume"]

    meta = metadata.copy()
    meta["symbol"] = meta["symbol"].astype(str).str.upper()
    out = out.merge(meta, on="symbol", how="left")
    out["market_cap"] = out["shares_outstanding"] * out["close"]
    out["market_cap"] = out["market_cap"].fillna(out["market_cap_latest"])
    out["turnover"] = (out["volume"] / out["shares_outstanding"]) * 100.0
    out = out.drop(columns=["market_cap_latest"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch US daily equities via yfinance")
    parser.add_argument("--symbols-file", default=None, help="CSV/parquet with a symbol column")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--raw-output", required=True)
    parser.add_argument("--metadata-output", default=None)
    args = parser.parse_args()

    configure_yfinance_cache()
    symbols = load_symbols(args.symbols_file, args.symbols)
    history = fetch_history(symbols, args.start, args.end, batch_size=args.batch_size)
    metadata = extract_metadata(symbols)
    long_df = to_long_frame(history, metadata)

    raw_output = Path(args.raw_output)
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    if raw_output.suffix.lower() == ".parquet":
        long_df.to_parquet(raw_output, index=False)
    else:
        long_df.to_csv(raw_output, index=False)
    logger.info("Saved {} rows to {}", len(long_df), raw_output)

    if args.metadata_output:
        meta_path = Path(args.metadata_output)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        if meta_path.suffix.lower() == ".parquet":
            metadata.to_parquet(meta_path, index=False)
        else:
            metadata.to_csv(meta_path, index=False)
        logger.info("Saved metadata to {}", meta_path)


if __name__ == "__main__":
    main()
