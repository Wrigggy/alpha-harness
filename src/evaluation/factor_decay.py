"""Analyze IC decay over multiple forward horizons."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from loguru import logger


def compute_ic_decay(
    factor_values: pd.DataFrame,
    close_prices: pd.DataFrame,
    horizons: list[int] | None = None,
    min_observations: int = 10,
) -> pd.DataFrame:
    """Compute IC at multiple forward return horizons.

    Args:
        factor_values: DataFrame (timestamp x symbols).
        close_prices: Close price DataFrame (timestamp x symbols).
        horizons: List of forward bar counts. Default: [1, 2, 4, 8, 24].
        min_observations: Min valid observations per cross-section.

    Returns:
        DataFrame with columns: horizon, IC, RankIC.
    """
    if horizons is None:
        horizons = [1, 2, 4, 8, 24]

    common_idx = factor_values.index.intersection(close_prices.index)
    common_cols = factor_values.columns.intersection(close_prices.columns)
    fv = factor_values.loc[common_idx, common_cols]
    close = close_prices.loc[common_idx, common_cols]

    results = []
    for h in horizons:
        # Compute forward returns at horizon h
        fwd_ret = close.shift(-h) / close - 1

        ics = []
        rics = []
        for t in common_idx:
            a = fv.loc[t].values.astype(float)
            r = fwd_ret.loc[t].values.astype(float)
            mask = np.isfinite(a) & np.isfinite(r)
            if mask.sum() < min_observations:
                continue
            ic = float(np.corrcoef(a[mask], r[mask])[0, 1])
            ric, _ = spearmanr(a[mask], r[mask])
            if np.isfinite(ic):
                ics.append(ic)
            if np.isfinite(ric):
                rics.append(ric)

        ic_mean = float(np.mean(ics)) if ics else 0.0
        ric_mean = float(np.mean(rics)) if rics else 0.0

        results.append({
            "horizon": h,
            "IC": ic_mean,
            "RankIC": ric_mean,
        })
        logger.info(f"Horizon {h}H: IC={ic_mean:.4f}, RankIC={ric_mean:.4f}")

    return pd.DataFrame(results)


def plot_ic_decay(
    decay_df: pd.DataFrame,
    factor_name: str = "Factor",
    save_path: str | None = None,
):
    """Plot IC decay curve over forward horizons."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(decay_df["horizon"], decay_df["IC"], "o-", label="IC", color="steelblue")
    ax.plot(decay_df["horizon"], decay_df["RankIC"], "s--", label="RankIC", color="coral")

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Forward Horizon (hours)")
    ax.set_ylabel("IC")
    ax.set_title(f"IC Decay: {factor_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"IC decay plot saved to {save_path}")

    return fig
