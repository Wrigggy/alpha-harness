"""Crypto data source wrapping the existing Binance pipeline."""

from pathlib import Path

import pandas as pd
from loguru import logger

from src.data_collection.data_cleaner import load_panel as load_panel_from_disk
from src.data_sources.base import DataSource


class CryptoSource(DataSource):
    """Load crypto OHLCV panel data from processed parquet files."""

    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = Path(processed_dir)

    def load_panel(self) -> dict[str, pd.DataFrame]:
        """Load panel from data/processed/ directory."""
        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"Processed data directory not found: {self.processed_dir}. "
                "Run the data cleaning pipeline first."
            )
        panel = load_panel_from_disk(str(self.processed_dir))
        logger.info(f"CryptoSource loaded {len(panel)} fields from {self.processed_dir}")
        return panel

    def get_metadata(self) -> dict:
        """Return crypto-specific metadata."""
        panel = load_panel_from_disk(str(self.processed_dir))
        close = panel.get("close")
        n_symbols = len(close.columns) if close is not None else 0
        date_range = (
            (str(close.index.min()), str(close.index.max()))
            if close is not None and len(close) > 0
            else (None, None)
        )
        return {
            "asset_class": "crypto",
            "frequency": "1h",
            "n_symbols": n_symbols,
            "date_range": date_range,
            "source": "binance",
        }
