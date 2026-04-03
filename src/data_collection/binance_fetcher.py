"""Download USDT-margined perpetual futures OHLCV from data.binance.vision.

Uses Binance's public bulk data repository (no API key, no rate limits).
Downloads monthly zip files containing CSV klines.

URL pattern:
  https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{YYYY-MM}.zip
"""

import asyncio
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
import yaml
from loguru import logger

VISION_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


def _generate_months(start_year: int, start_month: int, end_year: int, end_month: int) -> list[str]:
    """Generate list of 'YYYY-MM' strings between start and end (inclusive)."""
    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


async def download_month(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    year_month: str,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame | None:
    """Download and extract one monthly kline zip file."""
    url = f"{VISION_BASE}/{symbol}/{interval}/{symbol}-{interval}-{year_month}.zip"

    async with semaphore:
        try:
            async with session.get(url) as resp:
                if resp.status == 404:
                    return None  # Month not available (symbol not listed yet)
                resp.raise_for_status()
                data = await resp.read()
        except Exception as e:
            logger.debug(f"{symbol} {year_month}: download failed - {e}")
            return None

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)
    except Exception as e:
        logger.debug(f"{symbol} {year_month}: extract failed - {e}")
        return None

    # Keep only columns we need
    df = df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]
    df = df.rename(columns={"open_time": "timestamp"})

    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["timestamp"] = df["timestamp"].astype(int)

    return df


async def download_symbol(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    months: list[str],
    raw_dir: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Download all monthly kline files for a symbol and save as parquet."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    file_path = raw_path / f"{symbol}_1h.parquet"

    tasks = [download_month(session, symbol, interval, m, semaphore) for m in months]
    results = await asyncio.gather(*tasks)

    dfs = [df for df in results if df is not None and not df.empty]
    if not dfs:
        logger.debug(f"{symbol}: no data available")
        return False

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    df.to_parquet(file_path, index=False)
    logger.info(f"{symbol}: saved {len(df)} candles from {len(dfs)} months")
    return True


async def download_universe(
    symbols: list[str],
    config_path: str = "config/data_config.yaml",
):
    """Download OHLCV data for all symbols from data.binance.vision."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    interval = cfg["data"]["interval"]
    raw_dir = cfg["data"]["raw_dir"]
    lookback_years = cfg["data"]["lookback_years"]
    concurrency = cfg["binance"].get("concurrency", 10)

    now = datetime.now(timezone.utc)
    start_year = now.year - lookback_years
    start_month = now.month
    # Don't include current month (incomplete)
    end_month = now.month - 1 if now.month > 1 else 12
    end_year = now.year if now.month > 1 else now.year - 1

    months = _generate_months(start_year, start_month, end_year, end_month)
    logger.info(f"Downloading {len(months)} months of {interval} data for {len(symbols)} symbols")
    logger.info(f"Range: {months[0]} to {months[-1]}")

    semaphore = asyncio.Semaphore(concurrency)
    success = 0
    failed = 0

    # Process in batches for progress reporting
    batch_size = 20
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            tasks = [
                download_symbol(session, sym, interval, months, raw_dir, semaphore)
                for sym in batch
            ]
            results = await asyncio.gather(*tasks)
            success += sum(results)
            failed += len(results) - sum(results)
            logger.info(f"Progress: {i + len(batch)}/{len(symbols)} ({success} ok, {failed} failed)")

    logger.info(f"Download complete: {success} succeeded, {failed} failed out of {len(symbols)}")


async def main():
    from src.data_collection.universe_selector import load_universe

    universe = load_universe()
    symbols = [s["symbol"] for s in universe]
    await download_universe(symbols)


if __name__ == "__main__":
    asyncio.run(main())
