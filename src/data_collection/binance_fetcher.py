"""Async Binance OHLCV data downloader with incremental updates."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
import yaml
from loguru import logger

BASE_URL = "https://api.binance.com"
KLINE_ENDPOINT = "/api/v3/klines"
MAX_CANDLES_PER_REQUEST = 1000

# Column names for kline response
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


async def fetch_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    start_time: int | None = None,
    end_time: int | None = None,
    limit: int = MAX_CANDLES_PER_REQUEST,
) -> list[list]:
    """Fetch a single batch of klines from Binance."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    async with session.get(f"{BASE_URL}{KLINE_ENDPOINT}", params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def fetch_all_klines(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    start_time: int,
    end_time: int | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> pd.DataFrame:
    """Fetch all klines for a symbol between start_time and end_time."""
    all_candles = []
    current_start = start_time

    while True:
        if semaphore:
            async with semaphore:
                candles = await fetch_klines(
                    session, symbol, interval,
                    start_time=current_start, end_time=end_time,
                )
        else:
            candles = await fetch_klines(
                session, symbol, interval,
                start_time=current_start, end_time=end_time,
            )

        if not candles:
            break

        all_candles.extend(candles)

        # Move to next batch
        last_open_time = candles[-1][0]
        current_start = last_open_time + 1  # +1ms to avoid overlap

        if len(candles) < MAX_CANDLES_PER_REQUEST:
            break  # No more data available

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=KLINE_COLUMNS)

    # Keep only the columns we need
    df = df[["open_time", "open", "high", "low", "close", "volume", "quote_volume", "trades"]]
    df = df.rename(columns={"open_time": "timestamp"})

    # Convert types
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["timestamp"] = df["timestamp"].astype(int)

    # Drop duplicates by timestamp
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    return df


def get_last_timestamp(raw_dir: str, symbol: str) -> int | None:
    """Get the last stored timestamp for incremental updates."""
    path = Path(raw_dir) / f"{symbol}_1h.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["timestamp"])
    if df.empty:
        return None
    return int(df["timestamp"].max())


async def download_symbol(
    session: aiohttp.ClientSession,
    symbol: str,
    interval: str,
    start_time: int,
    raw_dir: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Download (or incrementally update) OHLCV data for a single symbol."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    file_path = raw_path / f"{symbol}_1h.parquet"

    # Check for incremental update
    last_ts = get_last_timestamp(raw_dir, symbol)
    fetch_start = last_ts + 1 if last_ts else start_time

    try:
        new_df = await fetch_all_klines(
            session, symbol, interval,
            start_time=fetch_start,
            semaphore=semaphore,
        )

        if new_df.empty:
            logger.debug(f"{symbol}: no new data")
            return True

        # Merge with existing data if incremental
        if last_ts and file_path.exists():
            existing_df = pd.read_parquet(file_path)
            df = pd.concat([existing_df, new_df]).drop_duplicates(subset="timestamp")
            df = df.sort_values("timestamp").reset_index(drop=True)
        else:
            df = new_df

        df.to_parquet(file_path, index=False)
        logger.info(f"{symbol}: saved {len(df)} candles ({len(new_df)} new)")
        return True

    except Exception as e:
        logger.error(f"{symbol}: download failed - {e}")
        return False


async def download_universe(
    symbols: list[str],
    config_path: str = "config/data_config.yaml",
):
    """Download OHLCV data for all symbols in the universe."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    interval = cfg["data"]["interval"]
    raw_dir = cfg["data"]["raw_dir"]
    concurrency = cfg["binance"]["concurrency"]
    lookback_years = cfg["data"]["lookback_years"]

    # Calculate start time
    now = datetime.now(timezone.utc)
    start_dt = now.replace(year=now.year - lookback_years)
    start_time = int(start_dt.timestamp() * 1000)

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = [
            download_symbol(session, sym, interval, start_time, raw_dir, semaphore)
            for sym in symbols
        ]
        results = await asyncio.gather(*tasks)

    success = sum(results)
    failed = len(results) - success
    logger.info(f"Download complete: {success} succeeded, {failed} failed out of {len(symbols)}")


async def main():
    from src.data_collection.universe_selector import load_universe

    universe = load_universe()
    symbols = [s["symbol"] for s in universe]
    await download_universe(symbols)


if __name__ == "__main__":
    asyncio.run(main())
