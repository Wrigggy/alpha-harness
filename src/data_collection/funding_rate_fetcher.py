"""Download historical funding rate data from data.binance.vision.

Funding rates are published every 8 hours for USDT-margined perpetual futures.
We download monthly zips and forward-fill to 1H frequency to match kline data.

URL pattern:
  https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{YYYY-MM}.zip
"""

import asyncio
import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
import numpy as np
import yaml
from loguru import logger

VISION_BASE = "https://data.binance.vision/data/futures/um/monthly/fundingRate"

FR_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]


def _generate_months(start_year: int, start_month: int, end_year: int, end_month: int) -> list[str]:
    months = []
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


async def download_funding_month(
    session: aiohttp.ClientSession,
    symbol: str,
    year_month: str,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame | None:
    url = f"{VISION_BASE}/{symbol}/{symbol}-fundingRate-{year_month}.zip"

    async with semaphore:
        try:
            async with session.get(url) as resp:
                if resp.status == 404:
                    return None
                resp.raise_for_status()
                data = await resp.read()
        except Exception as e:
            logger.debug(f"{symbol} {year_month}: funding download failed - {e}")
            return None

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=FR_COLUMNS)
    except Exception as e:
        logger.debug(f"{symbol} {year_month}: extract failed - {e}")
        return None

    # Drop header rows if present
    df = df[pd.to_numeric(df["calc_time"], errors="coerce").notna()].reset_index(drop=True)

    df["calc_time"] = df["calc_time"].astype(int)
    df["last_funding_rate"] = df["last_funding_rate"].astype(float)

    return df


async def download_symbol_funding(
    session: aiohttp.ClientSession,
    symbol: str,
    months: list[str],
    raw_dir: str,
    semaphore: asyncio.Semaphore,
) -> bool:
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    file_path = raw_path / f"{symbol}_funding.parquet"

    tasks = [download_funding_month(session, symbol, m, semaphore) for m in months]
    results = await asyncio.gather(*tasks)

    dfs = [df for df in results if df is not None and not df.empty]
    if not dfs:
        logger.debug(f"{symbol}: no funding rate data")
        return False

    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset="calc_time").sort_values("calc_time").reset_index(drop=True)

    # Convert to datetime and resample to 1H (forward-fill the 8H rate)
    df["datetime"] = pd.to_datetime(df["calc_time"], unit="ms", utc=True)
    df = df.set_index("datetime")[["last_funding_rate"]]
    df = df.rename(columns={"last_funding_rate": "funding_rate"})

    # Resample to 1H — forward fill (rate stays constant until next update)
    df_1h = df.resample("1h").ffill()

    # Save as parquet with timestamp column for consistency
    df_1h = df_1h.reset_index()
    df_1h["timestamp"] = df_1h["datetime"].astype(np.int64) // 10**6  # to unix ms
    df_1h = df_1h[["timestamp", "funding_rate"]]
    df_1h.to_parquet(file_path, index=False)

    logger.info(f"{symbol}: saved {len(df_1h)} funding rate records from {len(dfs)} months")
    return True


async def download_universe_funding(
    symbols: list[str],
    config_path: str = "config/data_config.yaml",
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    raw_dir = cfg["data"]["raw_dir"]
    lookback_years = cfg["data"]["lookback_years"]
    concurrency = cfg["binance"].get("concurrency", 20)

    now = datetime.now(timezone.utc)
    start_year = now.year - lookback_years
    end_month = now.month - 1 if now.month > 1 else 12
    end_year = now.year if now.month > 1 else now.year - 1

    months = _generate_months(start_year, 1, end_year, end_month)
    logger.info(f"Downloading funding rates: {len(months)} months for {len(symbols)} symbols")

    semaphore = asyncio.Semaphore(concurrency)
    success = 0
    failed = 0

    batch_size = 20
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            tasks = [
                download_symbol_funding(session, sym, months, raw_dir, semaphore)
                for sym in batch
            ]
            results = await asyncio.gather(*tasks)
            success += sum(results)
            failed += len(results) - sum(results)
            logger.info(f"Funding progress: {i + len(batch)}/{len(symbols)} ({success} ok, {failed} failed)")

    logger.info(f"Funding download complete: {success} succeeded, {failed} failed out of {len(symbols)}")


async def main():
    from src.data_collection.universe_selector import load_universe

    universe = load_universe()
    symbols = [s["symbol"] for s in universe]
    await download_universe_funding(symbols)


if __name__ == "__main__":
    asyncio.run(main())
