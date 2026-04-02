"""Cross-factor correlation analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


def compute_factor_correlation(
    factor_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute pairwise correlation matrix between all factors.

    For each pair of factors, computes the mean cross-sectional correlation
    across all timestamps.

    Args:
        factor_dict: dict mapping factor name -> factor values DataFrame
            (timestamp x symbols).

    Returns:
        Correlation matrix DataFrame (n_factors x n_factors).
    """
    names = list(factor_dict.keys())
    n = len(names)
    corr_matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            fi = factor_dict[names[i]]
            fj = factor_dict[names[j]]

            # Align
            common_idx = fi.index.intersection(fj.index)
            common_cols = fi.columns.intersection(fj.columns)

            a = fi.loc[common_idx, common_cols].values.flatten()
            b = fj.loc[common_idx, common_cols].values.flatten()

            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() > 10:
                corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
            else:
                corr = 0.0

            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

    result = pd.DataFrame(corr_matrix, index=names, columns=names)
    return result


def find_redundant_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
) -> list[tuple[str, str, float]]:
    """Find pairs of factors with correlation above the threshold.

    Returns:
        List of (factor_a, factor_b, correlation) tuples.
    """
    pairs = []
    names = corr_matrix.index.tolist()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                pairs.append((names[i], names[j], corr))
    if pairs:
        logger.warning(f"Found {len(pairs)} redundant factor pairs (|corr| > {threshold})")
    return pairs


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    save_path: str | None = None,
    figsize: tuple = (12, 10),
):
    """Plot and optionally save a correlation heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title("Factor Correlation Matrix")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Correlation heatmap saved to {save_path}")

    return fig
