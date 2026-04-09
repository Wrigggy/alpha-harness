"""Feature expansion from OHLCV to 50+ dimensions.

Transforms a raw OHLCV panel (dict of field -> DataFrame where each DataFrame
is timestamp x symbol) into an expanded feature dict with 50+ features spanning
returns, volatility, momentum, volume profile, price position, and more.

All features use STRICT point-in-time computation (no future data leakage).
"""

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Feature registry: populated during expand_features, queried via helpers
# ---------------------------------------------------------------------------

_FEATURE_METADATA: dict[str, dict] = {}


def _register(name: str, category: str, description: str) -> str:
    """Register a feature in the metadata registry and return its name."""
    _FEATURE_METADATA[name] = {"category": category, "description": description}
    return name


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_vwap(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return VWAP from panel, computing from HLC/3 if not available."""
    if "vwap" in panel:
        return panel["vwap"].copy()
    if "quote_volume" in panel and "volume" in panel:
        vwap = panel["quote_volume"] / panel["volume"].replace(0, np.nan)
        logger.debug("Computed VWAP from quote_volume / volume")
        return vwap
    # Fallback: typical price
    vwap = (panel["high"] + panel["low"] + panel["close"]) / 3
    logger.debug("Computed VWAP from (high + low + close) / 3")
    return vwap


def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf with NaN."""
    return df.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Main expansion function
# ---------------------------------------------------------------------------


def expand_features(panel: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Expand raw OHLCV panel into 50+ derived feature DataFrames.

    Args:
        panel: Dict with keys ``open``, ``close``, ``high``, ``low``,
            ``volume``, and optionally ``vwap`` / ``quote_volume``.
            Each value is a ``pd.DataFrame`` indexed by timestamp with
            columns representing symbols.

    Returns:
        Dict mapping feature name -> DataFrame (same shape as inputs).
        All values are sanitized (no inf).
    """
    _FEATURE_METADATA.clear()

    close = panel["close"]
    high = panel["high"]
    low = panel["low"]
    volume = panel["volume"]
    vwap = _ensure_vwap(panel)

    pct = close.pct_change()
    features: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # 1. Multi-window returns (7)
    # ------------------------------------------------------------------
    return_windows = [1, 2, 4, 8, 20, 60, 120]
    for w in return_windows:
        name = _register(
            f"ret_{w}", "returns", f"{w}-bar simple return"
        )
        features[name] = close / close.shift(w) - 1

    # ------------------------------------------------------------------
    # 2. Multi-window volatility (5)
    # ------------------------------------------------------------------
    vol_windows = [5, 10, 20, 60, 120]
    for w in vol_windows:
        name = _register(
            f"vol_{w}", "volatility", f"{w}-bar rolling volatility of returns"
        )
        features[name] = pct.rolling(w, min_periods=max(w // 2, 2)).std()

    # ------------------------------------------------------------------
    # 3. Momentum composites (4)
    # ------------------------------------------------------------------
    mom_pairs = [(5, 20), (10, 60), (20, 60), (5, 60)]
    for s, l in mom_pairs:
        ret_s = features.get(f"ret_{s}")
        ret_l = features.get(f"ret_{l}")
        # Compute on-the-fly if the return window wasn't already created
        if ret_s is None:
            ret_s = close / close.shift(s) - 1
        if ret_l is None:
            ret_l = close / close.shift(l) - 1
        name = _register(
            f"mom_{s}_{l}",
            "momentum",
            f"Momentum composite: ret_{s} / ret_{l}",
        )
        features[name] = ret_s / ret_l.replace(0, np.nan)

    # ------------------------------------------------------------------
    # 4. VWAP deviation (3)
    # ------------------------------------------------------------------
    vwap_dev_windows = [5, 10, 20]
    for w in vwap_dev_windows:
        name = _register(
            f"vwap_dev_{w}",
            "vwap",
            f"{w}-bar rolling VWAP deviation: (close - vwap) / vwap",
        )
        raw_dev = (close - vwap) / vwap.replace(0, np.nan)
        features[name] = raw_dev.rolling(w, min_periods=max(w // 2, 1)).mean()

    # ------------------------------------------------------------------
    # 5. Volume profile (3)
    # ------------------------------------------------------------------
    vol_ratio_pairs = [(5, 20), (10, 60)]
    for s, l in vol_ratio_pairs:
        name = _register(
            f"vol_ratio_{s}_{l}",
            "volume_profile",
            f"Volume ratio: {s}-bar MA / {l}-bar MA",
        )
        features[name] = (
            volume.rolling(s, min_periods=max(s // 2, 1)).mean()
            / volume.rolling(l, min_periods=max(l // 2, 1)).mean()
        )

    name = _register(
        "vol_skew_20",
        "volume_profile",
        "20-bar rolling skewness of volume",
    )
    features[name] = volume.rolling(20, min_periods=10).skew()

    # ------------------------------------------------------------------
    # 6. Intra-bar features (3)
    # ------------------------------------------------------------------
    hl_range = high - low

    name = _register("hl_range", "intra_bar", "(high - low) / close")
    features[name] = hl_range / close

    name = _register(
        "close_to_hl_mid",
        "intra_bar",
        "(close - HL midpoint) / HL range",
    )
    features[name] = (close - (high + low) / 2) / hl_range.replace(0, np.nan)

    name = _register(
        "close_to_vwap", "intra_bar", "(close - vwap) / close"
    )
    features[name] = (close - vwap) / close.replace(0, np.nan)

    # ------------------------------------------------------------------
    # 7. Price position (2)
    # ------------------------------------------------------------------
    name = _register(
        "close_to_high_20",
        "price_position",
        "(close - 20-bar high) / close",
    )
    features[name] = (close - close.rolling(20, min_periods=10).max()) / close

    name = _register(
        "close_to_low_20",
        "price_position",
        "(close - 20-bar low) / close",
    )
    features[name] = (close - close.rolling(20, min_periods=10).min()) / close

    # ------------------------------------------------------------------
    # 8. Rolling statistics (3)
    # ------------------------------------------------------------------
    name = _register(
        "skew_20", "rolling_stats", "20-bar rolling skewness of returns"
    )
    features[name] = pct.rolling(20, min_periods=10).skew()

    name = _register(
        "kurt_20", "rolling_stats", "20-bar rolling kurtosis of returns"
    )
    features[name] = pct.rolling(20, min_periods=10).kurt()

    name = _register(
        "autocorr_5",
        "rolling_stats",
        "Simplified 5-lag autocorrelation of returns (rolling 10-bar window)",
    )
    features[name] = pct.rolling(10, min_periods=10).apply(
        _autocorr_5, raw=True
    )

    # ------------------------------------------------------------------
    # 9. Cross-sectional rank features (3)
    # ------------------------------------------------------------------
    name = _register(
        "rank_ret_20",
        "cross_sectional",
        "Cross-sectional percentile rank of 20-bar return",
    )
    features[name] = features["ret_20"].rank(axis=1, pct=True)

    name = _register(
        "rank_vol_20",
        "cross_sectional",
        "Cross-sectional percentile rank of 20-bar volume MA",
    )
    features[name] = (
        volume.rolling(20, min_periods=10).mean().rank(axis=1, pct=True)
    )

    name = _register(
        "rank_vwap_dev",
        "cross_sectional",
        "Cross-sectional percentile rank of 20-bar VWAP deviation",
    )
    features[name] = features["vwap_dev_20"].rank(axis=1, pct=True)

    # ------------------------------------------------------------------
    # 10. Additional derived features
    # ------------------------------------------------------------------
    name = _register(
        "turnover_20",
        "derived",
        "Volume turnover: 20-bar MA / 60-bar MA",
    )
    features[name] = (
        volume.rolling(20, min_periods=10).mean()
        / volume.rolling(60, min_periods=30).mean()
    )

    name = _register(
        "amihud_20",
        "derived",
        "Amihud illiquidity: rolling 20-bar mean of |ret| / volume",
    )
    abs_ret = features["ret_1"].abs()
    raw_amihud = abs_ret / volume.replace(0, np.nan)
    features[name] = raw_amihud.rolling(20, min_periods=10).mean()

    name = _register(
        "high_low_corr_20",
        "derived",
        "20-bar rolling correlation between HL range and volume",
    )
    features[name] = hl_range.rolling(20, min_periods=10).corr(
        volume.rolling(20, min_periods=10).mean()
    )

    # Additional window variants to push past 50 features
    # Extra price-position windows
    for w in [60, 120]:
        name = _register(
            f"close_to_high_{w}",
            "price_position",
            f"(close - {w}-bar high) / close",
        )
        features[name] = (
            close - close.rolling(w, min_periods=w // 2).max()
        ) / close

        name = _register(
            f"close_to_low_{w}",
            "price_position",
            f"(close - {w}-bar low) / close",
        )
        features[name] = (
            close - close.rolling(w, min_periods=w // 2).min()
        ) / close

    # Extra rolling statistics windows
    name = _register(
        "skew_60", "rolling_stats", "60-bar rolling skewness of returns"
    )
    features[name] = pct.rolling(60, min_periods=30).skew()

    name = _register(
        "kurt_60", "rolling_stats", "60-bar rolling kurtosis of returns"
    )
    features[name] = pct.rolling(60, min_periods=30).kurt()

    # Volume-price correlation
    name = _register(
        "vol_price_corr_20",
        "derived",
        "20-bar rolling correlation between volume and returns",
    )
    features[name] = volume.rolling(20, min_periods=10).corr(pct)

    # Normalized ATR
    for w in [14, 20]:
        name = _register(
            f"natr_{w}",
            "volatility",
            f"{w}-bar normalized average true range",
        )
        tr = _true_range(high, low, close)
        features[name] = (
            tr.rolling(w, min_periods=w // 2).mean() / close
        )

    # Log volume ratio
    name = _register(
        "log_vol_ratio_5_20",
        "volume_profile",
        "Log of volume ratio 5/20",
    )
    features[name] = np.log1p(
        volume.rolling(5, min_periods=3).mean()
        / volume.rolling(20, min_periods=10).mean()
    )

    # Return acceleration (second derivative of price)
    name = _register(
        "ret_accel_5",
        "momentum",
        "Return acceleration: ret_1 - ret_1.shift(5)",
    )
    features[name] = features["ret_1"] - features["ret_1"].shift(5)

    # Gap (open vs previous close)
    if "open" in panel:
        name = _register(
            "gap",
            "intra_bar",
            "Gap: (open - prev_close) / prev_close",
        )
        features[name] = (panel["open"] - close.shift(1)) / close.shift(1)

    # Realized variance ratio (5 vs 20)
    name = _register(
        "var_ratio_5_20",
        "volatility",
        "Variance ratio: vol_5^2 / vol_20^2",
    )
    features[name] = features["vol_5"] ** 2 / (features["vol_20"] ** 2).replace(0, np.nan)

    # Down volatility (semi-deviation)
    name = _register(
        "downvol_20",
        "volatility",
        "20-bar downside volatility (returns < 0 only)",
    )
    neg_ret = pct.where(pct < 0, 0.0)
    features[name] = neg_ret.rolling(20, min_periods=10).std()

    # Volume surge
    name = _register(
        "vol_surge_5",
        "volume_profile",
        "Current volume / 5-bar volume MA",
    )
    features[name] = volume / volume.rolling(5, min_periods=3).mean()

    # ------------------------------------------------------------------
    # Sanitize all features
    # ------------------------------------------------------------------
    for key in features:
        features[key] = _sanitize(features[key])

    logger.info(
        f"Expanded {len(features)} features across "
        f"{len(set(m['category'] for m in _FEATURE_METADATA.values()))} categories"
    )
    return features


# ---------------------------------------------------------------------------
# Numba-free helper functions
# ---------------------------------------------------------------------------


def _autocorr_5(x: np.ndarray) -> float:
    """Compute simplified 5-lag autocorrelation from a 10-element array."""
    if len(x) < 10:
        return np.nan
    first = x[:5]
    second = x[5:]
    if np.std(first) == 0 or np.std(second) == 0:
        return np.nan
    return float(np.corrcoef(first, second)[0, 1])


def _true_range(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame
) -> pd.DataFrame:
    """Compute true range (vectorized, no future data)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.DataFrame(
        np.maximum(np.maximum(tr1.values, tr2.values), tr3.values),
        index=high.index,
        columns=high.columns,
    )


# ---------------------------------------------------------------------------
# Public query helpers
# ---------------------------------------------------------------------------


def get_feature_names() -> list[str]:
    """Return the list of all feature names produced by ``expand_features``.

    Note: This reflects the features from the most recent call to
    ``expand_features``. If ``expand_features`` has not been called yet,
    an empty list is returned.
    """
    return list(_FEATURE_METADATA.keys())


def get_feature_metadata() -> dict[str, dict]:
    """Return metadata dict mapping feature name -> {category, description}.

    Note: This reflects the features from the most recent call to
    ``expand_features``. If ``expand_features`` has not been called yet,
    an empty dict is returned.
    """
    return dict(_FEATURE_METADATA)
