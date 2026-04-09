"""Abstract base class for data sources."""

from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    """Unified interface for loading OHLCV panel data from any source."""

    @abstractmethod
    def load_panel(self) -> dict[str, pd.DataFrame]:
        """Return dict of field -> DataFrame (timestamp x symbols).

        Required keys: open, close, high, low, volume.
        Optional: quote_volume, vwap, etc.
        """

    @abstractmethod
    def get_metadata(self) -> dict:
        """Return dataset metadata.

        Required keys: asset_class, frequency, n_symbols, date_range.
        """
