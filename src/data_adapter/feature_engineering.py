"""Compute derived features for AlphaGen input (normalized OHLCV + VWAP)."""

import numpy as np
import pandas as pd
from loguru import logger


def compute_normalized_features(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Compute AlphaGen-compatible normalized features from raw panel data.

    Features:
        open_norm  = open / prev_close
        close_norm = close / prev_close
        high_norm  = high / prev_close
        low_norm   = low / prev_close
        volume_norm = volume / rolling_mean(volume, 20)
        vwap_norm  = (quote_volume / volume) / prev_close

    Args:
        panel: dict of field DataFrames (timestamp x symbol matrices)

    Returns:
        dict with keys: open, close, high, low, volume, vwap (all normalized)
    """
    close = panel["close"]
    prev_close = close.shift(1)

    features = {}

    # Price features normalized by previous close
    features["open"] = panel["open"] / prev_close
    features["close"] = close / prev_close
    features["high"] = panel["high"] / prev_close
    features["low"] = panel["low"] / prev_close

    # Volume normalized by 20-period rolling mean
    vol = panel["volume"]
    vol_mean = vol.rolling(window=20, min_periods=1).mean()
    features["volume"] = vol / vol_mean.replace(0, np.nan)

    # VWAP normalized by previous close
    # VWAP = quote_volume / volume (average price weighted by volume)
    vwap = panel["quote_volume"] / panel["volume"].replace(0, np.nan)
    features["vwap"] = vwap / prev_close

    # Replace inf with nan, then fill remaining nans with 1.0 (neutral value)
    for key in features:
        features[key] = features[key].replace([np.inf, -np.inf], np.nan)

    logger.info(f"Computed {len(features)} normalized features")
    return features
