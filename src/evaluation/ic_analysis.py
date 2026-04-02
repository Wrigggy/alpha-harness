"""Compute IC, ICIR, RankIC, RankICIR for alpha factor evaluation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from loguru import logger


@dataclass
class ICMetrics:
    ic_mean: float
    ic_std: float
    icir: float
    rank_ic_mean: float
    rank_ic_std: float
    rank_icir: float
    ic_series: pd.Series
    rank_ic_series: pd.Series


def evaluate_factor(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    min_observations: int = 10,
) -> ICMetrics:
    """Evaluate a single factor via cross-sectional IC analysis.

    Args:
        factor_values: DataFrame, index=timestamp, columns=symbols.
        forward_returns: DataFrame, same shape, forward returns.
        min_observations: Minimum valid observations per timestamp.

    Returns:
        ICMetrics with all IC statistics.
    """
    # Align indices and columns
    common_idx = factor_values.index.intersection(forward_returns.index)
    common_cols = factor_values.columns.intersection(forward_returns.columns)
    fv = factor_values.loc[common_idx, common_cols]
    fr = forward_returns.loc[common_idx, common_cols]

    ics = []
    rics = []
    timestamps = []

    for t in common_idx:
        a = fv.loc[t].values.astype(float)
        r = fr.loc[t].values.astype(float)

        mask = np.isfinite(a) & np.isfinite(r)
        if mask.sum() < min_observations:
            continue

        ic, _ = pearsonr(a[mask], r[mask])
        ric, _ = spearmanr(a[mask], r[mask])

        if np.isfinite(ic) and np.isfinite(ric):
            ics.append(ic)
            rics.append(ric)
            timestamps.append(t)

    ic_series = pd.Series(ics, index=timestamps, name="IC")
    ric_series = pd.Series(rics, index=timestamps, name="RankIC")

    ic_mean = float(np.mean(ics)) if ics else 0.0
    ic_std = float(np.std(ics)) if ics else 1.0
    ric_mean = float(np.mean(rics)) if rics else 0.0
    ric_std = float(np.std(rics)) if rics else 1.0

    return ICMetrics(
        ic_mean=ic_mean,
        ic_std=ic_std,
        icir=ic_mean / ic_std if ic_std > 0 else 0.0,
        rank_ic_mean=ric_mean,
        rank_ic_std=ric_std,
        rank_icir=ric_mean / ric_std if ric_std > 0 else 0.0,
        ic_series=ic_series,
        rank_ic_series=ric_series,
    )


def evaluate_factor_pool(
    factor_dict: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate all factors in a pool and return a summary table.

    Args:
        factor_dict: dict mapping factor name -> factor values DataFrame.
        forward_returns: forward returns DataFrame.

    Returns:
        Summary DataFrame with columns: IC, ICIR, RankIC, RankICIR.
    """
    rows = []
    for name, fv in factor_dict.items():
        metrics = evaluate_factor(fv, forward_returns)
        rows.append({
            "factor": name,
            "IC": metrics.ic_mean,
            "IC_std": metrics.ic_std,
            "ICIR": metrics.icir,
            "RankIC": metrics.rank_ic_mean,
            "RankIC_std": metrics.rank_ic_std,
            "RankICIR": metrics.rank_icir,
        })
        logger.info(
            f"{name}: IC={metrics.ic_mean:.4f} ICIR={metrics.icir:.4f} "
            f"RankIC={metrics.rank_ic_mean:.4f} RankICIR={metrics.rank_icir:.4f}"
        )

    return pd.DataFrame(rows).set_index("factor")
