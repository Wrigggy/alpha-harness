"""Local panel data source for equity-style datasets.

This source loads the repo's native panel layout:
    panel_dir/
      open.parquet
      high.parquet
      low.parquet
      close.parquet
      volume.parquet
      ...
      meta.json   # optional
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.data_collection.data_cleaner import load_panel as load_panel_from_disk
from src.data_sources.base import DataSource


def panel_directory_exists(panel_dir: str | Path) -> bool:
    """Return True when a directory contains at least one panel parquet file."""
    path = Path(panel_dir)
    return path.exists() and any(path.glob("*.parquet"))


class LocalPanelSource(DataSource):
    """Load a local OHLCV panel from disk."""

    def __init__(
        self,
        panel_dir: str = "data/equity_panel",
        asset_class: str = "equity",
        frequency: str = "1d",
        source_name: str = "local_panel",
    ):
        self.panel_dir = Path(panel_dir)
        self.asset_class = asset_class
        self.frequency = frequency
        self.source_name = source_name

    @property
    def metadata_path(self) -> Path:
        return self.panel_dir / "meta.json"

    def _load_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {}
        with open(self.metadata_path, encoding="utf-8") as f:
            raw = json.load(f)
        return raw if isinstance(raw, dict) else {}

    def load_panel(self) -> dict[str, pd.DataFrame]:
        if not panel_directory_exists(self.panel_dir):
            raise FileNotFoundError(
                f"Panel directory not found or empty: {self.panel_dir}. "
                "Use scripts/import_equity_panel.py to create a local equity panel."
            )
        panel = load_panel_from_disk(str(self.panel_dir))
        logger.info("LocalPanelSource loaded {} fields from {}", len(panel), self.panel_dir)
        return panel

    def get_metadata(self) -> dict[str, Any]:
        meta = self._load_metadata()
        panel = self.load_panel()
        close = panel.get("close")
        inferred = {
            "asset_class": meta.get("asset_class", self.asset_class),
            "frequency": meta.get("frequency", self.frequency),
            "source": meta.get("source", self.source_name),
            "market": meta.get("market"),
            "universe": meta.get("universe"),
            "date_range": meta.get(
                "date_range",
                (
                    str(close.index.min()) if close is not None and len(close) > 0 else None,
                    str(close.index.max()) if close is not None and len(close) > 0 else None,
                ),
            ),
            "n_symbols": meta.get("n_symbols", len(close.columns) if close is not None else 0),
        }
        return inferred
