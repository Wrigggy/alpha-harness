"""Clean and align raw OHLCV data into a panel DataFrame."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger


def load_raw_data(raw_dir: str) -> dict[str, pd.DataFrame]:
    """Load all raw parquet files into a dict keyed by symbol."""
    raw_path = Path(raw_dir)
    data = {}
    for f in sorted(raw_path.glob("*_1h.parquet")):
        symbol = f.stem.replace("_1h", "")
        df = pd.read_parquet(f)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime").sort_index()
        data[symbol] = df
    logger.info(f"Loaded raw data for {len(data)} symbols")
    return data


def build_common_index(data: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Build a common 1H datetime index covering the union of all symbols."""
    all_starts = [df.index.min() for df in data.values()]
    all_ends = [df.index.max() for df in data.values()]
    start = min(all_starts)
    end = max(all_ends)
    index = pd.date_range(start=start, end=end, freq="1h", tz="UTC")
    logger.info(f"Common index: {start} to {end}, {len(index)} bars")
    return index


def clean_and_align(
    data: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    max_ffill_hours: int = 4,
    max_missing_pct: float = 0.10,
) -> dict[str, pd.DataFrame]:
    """Reindex all symbols to common index, forward-fill short gaps, remove bad symbols."""
    cleaned = {}
    removed = []

    for symbol, df in data.items():
        # Reindex to common timeline
        df = df.reindex(common_index)

        # Forward-fill gaps up to max_ffill_hours
        df = df.ffill(limit=max_ffill_hours)

        # Check missing data percentage (only within the symbol's active range)
        first_valid = df["close"].first_valid_index()
        last_valid = df["close"].last_valid_index()
        if first_valid is None or last_valid is None:
            removed.append(symbol)
            continue

        active_slice = df.loc[first_valid:last_valid, "close"]
        missing_pct = active_slice.isna().mean()

        if missing_pct > max_missing_pct:
            removed.append(symbol)
            logger.debug(f"{symbol}: removed ({missing_pct:.1%} missing)")
            continue

        cleaned[symbol] = df

    logger.info(f"Cleaned: {len(cleaned)} symbols kept, {len(removed)} removed")
    return cleaned


def load_funding_rates(raw_dir: str, symbols: list[str], common_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load funding rate parquet files and build a timestamp x symbol matrix."""
    raw_path = Path(raw_dir)
    matrix = pd.DataFrame(index=common_index, columns=symbols, dtype=float)

    loaded = 0
    for sym in symbols:
        path = raw_path / f"{sym}_funding.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime")["funding_rate"]
        # Reindex to common 1H index, forward-fill (rate is constant between updates)
        df = df.reindex(common_index).ffill()
        matrix[sym] = df
        loaded += 1

    logger.info(f"Loaded funding rates for {loaded}/{len(symbols)} symbols")
    return matrix


def build_panel(
    cleaned: dict[str, pd.DataFrame],
    common_index: pd.DatetimeIndex,
    raw_dir: str = "data/raw",
) -> dict[str, pd.DataFrame]:
    """Build field-level DataFrames (timestamp x symbol matrices).

    Returns dict with keys: open, high, low, close, volume, quote_volume,
    funding_rate, ret_1h, ret_4h
    """
    symbols = sorted(cleaned.keys())
    fields = ["open", "high", "low", "close", "volume", "quote_volume"]

    panel = {}
    for field in fields:
        matrix = pd.DataFrame(index=common_index, columns=symbols, dtype=float)
        for sym in symbols:
            if field in cleaned[sym].columns:
                matrix[sym] = cleaned[sym][field]
        panel[field] = matrix

    # Load funding rate data
    panel["funding_rate"] = load_funding_rates(raw_dir, symbols, common_index)

    # Compute forward returns (shift negative = look forward)
    close = panel["close"]
    panel["ret_1h"] = close.shift(-1) / close - 1
    panel["ret_4h"] = close.shift(-4) / close - 1
    panel["ret_20h"] = close.shift(-20) / close - 1  # Matches AlphaGen target horizon

    logger.info(f"Panel built: {len(symbols)} symbols, {len(common_index)} bars, {len(panel)} fields")
    return panel


def save_panel(panel: dict[str, pd.DataFrame], processed_dir: str):
    """Save each field as a separate parquet file."""
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    for field, df in panel.items():
        path = out / f"{field}.parquet"
        df.to_parquet(path)
        logger.debug(f"Saved {path}")

    # Also save symbol list
    symbols = list(panel["close"].columns)
    pd.Series(symbols).to_csv(out / "symbols.csv", index=False, header=False)

    logger.info(f"Panel saved to {processed_dir}")


def load_panel(processed_dir: str) -> dict[str, pd.DataFrame]:
    """Load previously saved panel data."""
    out = Path(processed_dir)
    panel = {}
    for f in out.glob("*.parquet"):
        field = f.stem
        panel[field] = pd.read_parquet(f)
    logger.info(f"Loaded panel from {processed_dir}: {list(panel.keys())}")
    return panel


def run_cleaning(config_path: str = "config/data_config.yaml"):
    """Full cleaning pipeline: load raw -> clean -> build panel -> save."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["data"]["raw_dir"]
    processed_dir = cfg["data"]["processed_dir"]

    # Load
    data = load_raw_data(raw_dir)
    if not data:
        logger.error("No raw data found. Run binance_fetcher first.")
        return

    # Clean
    common_index = build_common_index(data)
    cleaned = clean_and_align(data, common_index)

    # Build panel (includes funding rate if available)
    panel = build_panel(cleaned, common_index, raw_dir=raw_dir)

    # Save
    save_panel(panel, processed_dir)


if __name__ == "__main__":
    run_cleaning()
