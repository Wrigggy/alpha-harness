"""Qlib/CSI500 data source for Chinese A-share equities."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from loguru import logger

from src.data_sources.base import DataSource
from src.data_sources.local_panel_source import LocalPanelSource, panel_directory_exists

QLIB_CACHE_DIR = Path("data/qlib_cache")
OHLCV_FIELDS = ["open", "close", "high", "low", "volume"]


class QlibSource(DataSource):
    """Load equity OHLCV data via qlib (CSI500 or other instrument pools)."""

    def __init__(
        self,
        instruments: str = "csi500",
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        dataset: str = "Alpha158",
        cache_dir: str = str(QLIB_CACHE_DIR),
        provider_uri: str | None = None,
        fallback_panel_dir: str | None = None,
        prefer_local_panel: bool = False,
    ):
        self.instruments = instruments
        self.start_date = start_date
        self.end_date = end_date
        self.dataset = dataset
        self.cache_dir = Path(cache_dir)
        self.provider_uri = provider_uri
        self.fallback_panel_dir = Path(fallback_panel_dir) if fallback_panel_dir else None
        self.prefer_local_panel = prefer_local_panel

    def _init_qlib(self):
        """Initialize qlib with default provider."""
        try:
            import qlib
            from qlib.config import REG_CN

            provider_uri = self.provider_uri or os.path.expanduser("~/.qlib/qlib_data/cn_data")
            qlib.init(provider_uri=provider_uri, region=REG_CN)
            logger.info("Qlib initialized with CN provider: {}", provider_uri)
        except ImportError:
            raise ImportError(
                "qlib is not installed. Install it with:\n"
                "  pip install pyqlib\n"
                "Then download data with:\n"
                "  python -m qlib.run.get_data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn"
            )

    def _cache_path(self) -> Path:
        """Path for cached panel data."""
        name = f"{self.instruments}_{self.start_date}_{self.end_date}_{self.dataset}"
        return self.cache_dir / name

    @property
    def cache_path(self) -> Path:
        return self._cache_path()

    def _load_from_cache(self) -> dict[str, pd.DataFrame] | None:
        """Load panel from cache if it exists."""
        cache = self._cache_path()
        if not cache.exists():
            return None
        panel = {}
        for f in cache.glob("*.parquet"):
            panel[f.stem] = pd.read_parquet(f)
        if panel:
            logger.info(f"Loaded {len(panel)} fields from cache: {cache}")
            return panel
        return None

    def _save_to_cache(self, panel: dict[str, pd.DataFrame]):
        """Save panel to cache directory."""
        cache = self._cache_path()
        cache.mkdir(parents=True, exist_ok=True)
        for field, df in panel.items():
            df.to_parquet(cache / f"{field}.parquet")
        logger.info(f"Cached {len(panel)} fields to {cache}")

    def download(self) -> dict[str, pd.DataFrame]:
        """Download data from qlib and convert to panel format.

        Returns dict of field -> DataFrame (date x symbol).
        """
        self._init_qlib()

        from qlib.data import D

        instruments = D.instruments(self.instruments)
        fields = ["$open", "$close", "$high", "$low", "$volume"]
        field_names = ["open", "close", "high", "low", "volume"]

        df = D.features(
            instruments,
            fields,
            start_time=self.start_date,
            end_time=self.end_date,
        )
        df.columns = field_names
        logger.info(f"Downloaded {len(df)} rows from qlib ({self.instruments})")

        # Convert MultiIndex (datetime, instrument) -> dict of pivoted DataFrames
        panel = {}
        for field in field_names:
            pivoted = df[field].unstack(level="instrument")
            pivoted.index.name = "datetime"
            panel[field] = pivoted

        self._save_to_cache(panel)
        return panel

    def load_panel(self) -> dict[str, pd.DataFrame]:
        """Load panel from cache, downloading if necessary."""
        if self.prefer_local_panel and self.fallback_panel_dir and panel_directory_exists(self.fallback_panel_dir):
            logger.info("Using local panel fallback instead of qlib: {}", self.fallback_panel_dir)
            return LocalPanelSource(
                panel_dir=str(self.fallback_panel_dir),
                asset_class="equity",
                frequency="1d",
                source_name=f"local_panel/{self.instruments}",
            ).load_panel()

        cached = self._load_from_cache()
        if cached is not None:
            return cached
        try:
            logger.info("No cache found, downloading from qlib...")
            return self.download()
        except ImportError:
            if self.fallback_panel_dir and panel_directory_exists(self.fallback_panel_dir):
                logger.warning(
                    "qlib unavailable; falling back to local panel: {}",
                    self.fallback_panel_dir,
                )
                return LocalPanelSource(
                    panel_dir=str(self.fallback_panel_dir),
                    asset_class="equity",
                    frequency="1d",
                    source_name=f"local_panel/{self.instruments}",
                ).load_panel()
            raise

    def get_metadata(self) -> dict:
        """Return equity-specific metadata."""
        panel = self.load_panel()
        close = panel.get("close")
        n_symbols = len(close.columns) if close is not None else 0
        date_range = (
            (str(close.index.min()), str(close.index.max()))
            if close is not None and len(close) > 0
            else (None, None)
        )
        return {
            "asset_class": "equity",
            "frequency": "1d",
            "n_symbols": n_symbols,
            "date_range": date_range,
            "source": f"qlib/{self.instruments}" if not self.prefer_local_panel else f"equity/{self.instruments}",
            "dataset": self.dataset,
        }
