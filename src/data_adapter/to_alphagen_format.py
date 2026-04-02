"""Convert Binance crypto data into AlphaGen/AlphaQCM tensor format.

This module provides CryptoStockData and CryptoAlphaCalculator as drop-in
replacements for AlphaGen's Qlib-based data loading.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from scipy.stats import pearsonr, spearmanr

from src.data_adapter.feature_engineering import compute_normalized_features
from src.data_collection.data_cleaner import load_panel


class CryptoStockData:
    """Drop-in replacement for AlphaGen's StockData.

    Loads Binance crypto data and exposes the same interface expected by
    AlphaGen's expression calculator and RL environment.

    Attributes:
        data: Tensor of shape (n_timestamps, n_symbols, n_features)
            Features are [open, close, high, low, volume, vwap] (all normalized).
        returns: Tensor of shape (n_timestamps, n_symbols) — forward 1H returns.
    """

    FEATURE_NAMES = ["open", "close", "high", "low", "volume", "vwap"]

    def __init__(
        self,
        processed_dir: str,
        start_date: str | None = None,
        end_date: str | None = None,
        device: str = "cpu",
    ):
        # Load panel data
        panel = load_panel(processed_dir)

        # Compute normalized features
        features = compute_normalized_features(panel)

        # Apply date filter
        index = panel["close"].index
        if start_date:
            index = index[index >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            index = index[index <= pd.Timestamp(end_date, tz="UTC")]

        self._timestamps = index
        self._symbols = list(panel["close"].columns)

        # Stack features into tensor: (n_timestamps, n_symbols, n_features)
        feature_arrays = []
        for fname in self.FEATURE_NAMES:
            arr = features[fname].loc[index].values.astype(np.float32)
            feature_arrays.append(arr)

        # shape: (n_features, n_timestamps, n_symbols) -> (n_timestamps, n_symbols, n_features)
        stacked = np.stack(feature_arrays, axis=-1)

        # Replace NaN with 0 for tensor computation
        stacked = np.nan_to_num(stacked, nan=0.0)

        self.data = torch.tensor(stacked, dtype=torch.float32, device=device)

        # Forward returns
        ret = panel["ret_1h"].loc[index].values.astype(np.float32)
        ret = np.nan_to_num(ret, nan=0.0)
        self.returns = torch.tensor(ret, dtype=torch.float32, device=device)

        logger.info(
            f"CryptoStockData loaded: {self.n_days} bars x {self.n_stocks} symbols x {len(self.FEATURE_NAMES)} features"
        )

    @property
    def n_stocks(self) -> int:
        return self.data.shape[1]

    @property
    def n_days(self) -> int:
        """Number of time bars (1H bars, not actual days)."""
        return self.data.shape[0]

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return self._timestamps

    @property
    def symbols(self) -> list[str]:
        return self._symbols

    def make_dataframe(self, alpha_tensor: torch.Tensor) -> pd.DataFrame:
        """Convert an alpha values tensor to a labeled DataFrame.

        Args:
            alpha_tensor: shape (n_timestamps, n_symbols)

        Returns:
            DataFrame with DatetimeIndex and symbol columns.
        """
        values = alpha_tensor.detach().cpu().numpy()
        return pd.DataFrame(values, index=self._timestamps, columns=self._symbols)


class CryptoAlphaCalculator:
    """Drop-in replacement for AlphaGen's QlibStockDataCalculator.

    Computes IC, RankIC, and pool-level metrics for factor evaluation.
    This is what the RL agent uses as its reward signal.
    """

    def __init__(self, stock_data: CryptoStockData):
        self.stock_data = stock_data

    def calc_single_IC(self, alpha: torch.Tensor) -> float:
        """Compute mean cross-sectional Pearson IC between alpha and forward returns.

        Args:
            alpha: Tensor of shape (n_timestamps, n_symbols)

        Returns:
            Mean IC across all timestamps.
        """
        alpha_np = alpha.detach().cpu().numpy()
        ret_np = self.stock_data.returns.detach().cpu().numpy()

        ics = []
        for t in range(alpha_np.shape[0]):
            a = alpha_np[t]
            r = ret_np[t]
            # Mask NaN and zero values
            mask = np.isfinite(a) & np.isfinite(r) & (a != 0)
            if mask.sum() < 10:
                continue
            ic, _ = pearsonr(a[mask], r[mask])
            if np.isfinite(ic):
                ics.append(ic)

        return float(np.mean(ics)) if ics else 0.0

    def calc_single_rIC(self, alpha: torch.Tensor) -> float:
        """Compute mean cross-sectional Spearman Rank IC."""
        alpha_np = alpha.detach().cpu().numpy()
        ret_np = self.stock_data.returns.detach().cpu().numpy()

        rics = []
        for t in range(alpha_np.shape[0]):
            a = alpha_np[t]
            r = ret_np[t]
            mask = np.isfinite(a) & np.isfinite(r) & (a != 0)
            if mask.sum() < 10:
                continue
            ric, _ = spearmanr(a[mask], r[mask])
            if np.isfinite(ric):
                rics.append(ric)

        return float(np.mean(rics)) if rics else 0.0

    def calc_pool_IC(self, pool_alphas: list[torch.Tensor], weights: list[float]) -> float:
        """Compute IC for a weighted combination of alpha factors.

        Args:
            pool_alphas: List of alpha tensors, each (n_timestamps, n_symbols).
            weights: Weights for each alpha in the pool.

        Returns:
            Mean IC of the combined alpha.
        """
        if not pool_alphas:
            return 0.0

        combined = torch.zeros_like(pool_alphas[0])
        for alpha, w in zip(pool_alphas, weights):
            combined += w * alpha

        return self.calc_single_IC(combined)

    def calc_pool_rIC(self, pool_alphas: list[torch.Tensor], weights: list[float]) -> float:
        """Compute Rank IC for a weighted combination of alpha factors."""
        if not pool_alphas:
            return 0.0

        combined = torch.zeros_like(pool_alphas[0])
        for alpha, w in zip(pool_alphas, weights):
            combined += w * alpha

        return self.calc_single_rIC(combined)


def create_data_splits(
    processed_dir: str,
    config_path: str = "config/data_config.yaml",
    device: str = "cpu",
) -> dict[str, CryptoStockData]:
    """Create train/val/test CryptoStockData splits based on config ratios.

    Returns:
        dict with keys "train", "val", "test", each a CryptoStockData instance.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    split_cfg = cfg["split"]

    # Load panel to get full date range
    panel = load_panel(processed_dir)
    index = panel["close"].index

    n = len(index)
    train_end = int(n * split_cfg["train_ratio"])
    val_end = train_end + int(n * split_cfg["val_ratio"])

    train_start = str(index[0])
    train_end_date = str(index[train_end - 1])
    val_start = str(index[train_end])
    val_end_date = str(index[val_end - 1])
    test_start = str(index[val_end])
    test_end_date = str(index[-1])

    logger.info(f"Train: {train_start} -> {train_end_date}")
    logger.info(f"Val:   {val_start} -> {val_end_date}")
    logger.info(f"Test:  {test_start} -> {test_end_date}")

    return {
        "train": CryptoStockData(processed_dir, train_start, train_end_date, device),
        "val": CryptoStockData(processed_dir, val_start, val_end_date, device),
        "test": CryptoStockData(processed_dir, test_start, test_end_date, device),
    }
