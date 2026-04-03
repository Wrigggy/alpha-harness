"""Select top USDT-margined perpetual futures pairs from Binance by volume."""

import json
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import asyncio
import yaml
from loguru import logger

BASE_URL = "https://fapi.binance.com"

STABLECOINS = {"USDT", "USDC", "DAI", "BUSD", "TUSD", "FDUSD", "USDP", "PYUSD"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> dict:
    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def select_universe(config_path: str = "config/data_config.yaml") -> list[dict]:
    """Select trading universe from Binance based on config filters.

    Returns a list of symbol dicts sorted by 24h quote volume descending.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    uni_cfg = cfg["universe"]
    quote_asset = uni_cfg["quote_asset"]
    min_volume = uni_cfg["min_24h_volume_usd"]
    min_listing_months = uni_cfg["min_listing_months"]
    max_symbols = uni_cfg["max_symbols"]
    exclude_stables = set(uni_cfg.get("exclude_stablecoins", STABLECOINS))
    exclude_leveraged = uni_cfg.get("exclude_leveraged_tokens", True)

    async with aiohttp.ClientSession() as session:
        # Fetch futures tickers and exchange info concurrently
        tickers_task = fetch_json(session, f"{BASE_URL}/fapi/v1/ticker/24hr")
        exchange_info_task = fetch_json(session, f"{BASE_URL}/fapi/v1/exchangeInfo")
        tickers, exchange_info = await asyncio.gather(tickers_task, exchange_info_task)

    # Build symbol metadata from exchange info (perpetual contracts only)
    symbol_meta = {}
    now = datetime.now(timezone.utc)
    for s in exchange_info["symbols"]:
        if s["quoteAsset"] != quote_asset:
            continue
        if s["status"] != "TRADING":
            continue
        # Only perpetual contracts, skip delivery futures
        if s.get("contractType") != "PERPETUAL":
            continue
        symbol_meta[s["symbol"]] = {
            "symbol": s["symbol"],
            "base": s["baseAsset"],
            "quote": s["quoteAsset"],
            "contract_type": "PERPETUAL",
        }

    # Build volume map from tickers
    volume_map = {}
    for t in tickers:
        sym = t["symbol"]
        if sym in symbol_meta:
            volume_map[sym] = float(t["quoteVolume"])

    # Apply filters
    candidates = []
    for sym, meta in symbol_meta.items():
        base = meta["base"]

        # Exclude stablecoins
        if base in exclude_stables:
            continue

        # Exclude leveraged tokens
        if exclude_leveraged and any(base.endswith(s) for s in LEVERAGED_SUFFIXES):
            continue

        # Minimum volume filter
        vol = volume_map.get(sym, 0)
        if vol < min_volume:
            continue

        meta["quote_volume_24h"] = vol
        candidates.append(meta)

    # Sort by volume descending, take top N
    candidates.sort(key=lambda x: x["quote_volume_24h"], reverse=True)
    universe = candidates[:max_symbols]

    logger.info(f"Selected {len(universe)} symbols from {len(symbol_meta)} {quote_asset} pairs")

    return universe


def save_universe(universe: list[dict], output_path: str = "config/universe.json"):
    """Save universe snapshot with a date stamp."""
    snapshot = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "count": len(universe),
        "symbols": universe,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    logger.info(f"Universe saved to {output_path} ({len(universe)} symbols)")


def load_universe(path: str = "config/universe.json") -> list[dict]:
    """Load a previously saved universe."""
    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded universe from {path}: {data['count']} symbols (snapshot {data['date']})")
    return data["symbols"]


async def main():
    universe = await select_universe()
    save_universe(universe)


if __name__ == "__main__":
    asyncio.run(main())
