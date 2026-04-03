"""Top-N long / bottom-N short portfolio backtest."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger

from src.backtest.performance_metrics import compute_metrics, compute_turnover


def long_short_backtest(
    factor_values: pd.DataFrame,
    returns: pd.DataFrame,
    n_long: int = 30,
    n_short: int = 30,
    liquidity_filter: pd.DataFrame | None = None,
    min_liquidity_percentile: float = 0.2,
    rebalance_freq: int = 1,
    transaction_cost_bps: float = 5.0,
) -> dict:
    """Run a simple long-short backtest.

    At each rebalance:
    1. Optionally filter universe by liquidity (quote_volume)
    2. Rank symbols by factor value
    3. Long top n_long (equal weight)
    4. Short bottom n_short (equal weight)
    5. Compute portfolio return net of transaction costs

    Args:
        factor_values: DataFrame (timestamp x symbols), factor signal.
        returns: DataFrame (timestamp x symbols), forward 1H returns.
        n_long: Number of symbols in the long leg.
        n_short: Number of symbols in the short leg.
        liquidity_filter: Optional DataFrame of liquidity proxy (e.g., quote_volume).
        min_liquidity_percentile: Filter out symbols below this percentile.
        rebalance_freq: Rebalance every N bars.
        transaction_cost_bps: One-way transaction cost in basis points.

    Returns:
        Dict with equity_curve, metrics, weights_history.
    """
    # Align data
    common_idx = factor_values.index.intersection(returns.index)
    common_cols = factor_values.columns.intersection(returns.columns)
    fv = factor_values.loc[common_idx, common_cols]
    ret = returns.loc[common_idx, common_cols]

    if liquidity_filter is not None:
        liq = liquidity_filter.loc[common_idx, common_cols]
    else:
        liq = None

    tc = transaction_cost_bps / 10000.0  # Convert bps to decimal

    portfolio_returns = []
    weights_records = []
    prev_weights = pd.Series(0.0, index=common_cols)

    rebalance_dates = common_idx[::rebalance_freq]

    for i, date in enumerate(common_idx):
        if date not in rebalance_dates:
            # Hold existing positions
            if not weights_records:
                continue
            bar_ret = ret.loc[date]
            port_ret = (prev_weights * bar_ret).sum()
            portfolio_returns.append({"date": date, "return": port_ret})
            continue

        # Get factor values for this bar
        fv_bar = fv.loc[date]
        valid_mask = fv_bar.notna()

        # Apply liquidity filter
        if liq is not None:
            liq_bar = liq.loc[date]
            if liq_bar.notna().sum() > 0:
                threshold = liq_bar.quantile(min_liquidity_percentile)
                valid_mask = valid_mask & (liq_bar >= threshold)

        valid_symbols = fv_bar[valid_mask].dropna()
        if len(valid_symbols) < n_long + n_short:
            portfolio_returns.append({"date": date, "return": 0.0})
            continue

        # Rank and select
        ranked = valid_symbols.rank(ascending=False)
        long_symbols = ranked.nsmallest(n_long).index  # Highest factor values
        short_symbols = ranked.nlargest(n_short).index  # Lowest factor values

        # Equal weight
        new_weights = pd.Series(0.0, index=common_cols)
        new_weights[long_symbols] = 1.0 / n_long
        new_weights[short_symbols] = -1.0 / n_short

        # On rebalance bar: use OLD weights for this bar's return (no look-ahead),
        # then switch to new weights for subsequent bars.
        # Transaction cost is deducted on the rebalance bar.
        turnover = (new_weights - prev_weights).abs().sum()
        tc_cost = turnover * tc

        bar_ret = ret.loc[date]
        port_ret = (prev_weights * bar_ret).sum() - tc_cost

        portfolio_returns.append({"date": date, "return": port_ret})
        weights_records.append({"date": date, **new_weights.to_dict()})
        prev_weights = new_weights

    # Build equity curve
    ret_df = pd.DataFrame(portfolio_returns).set_index("date")
    equity_curve = (1 + ret_df["return"]).cumprod()
    equity_curve.name = "equity"

    # Compute metrics
    metrics = compute_metrics(equity_curve)

    # Weights history
    weights_history = pd.DataFrame(weights_records).set_index("date") if weights_records else pd.DataFrame()
    if not weights_history.empty:
        metrics["avg_turnover"] = compute_turnover(weights_history)
    else:
        metrics["avg_turnover"] = 0.0

    logger.info(
        f"Backtest complete: AR={metrics['annual_return']:.2%}, "
        f"Sharpe={metrics['sharpe_ratio']:.2f}, "
        f"MDD={metrics['max_drawdown']:.2%}"
    )

    return {
        "equity_curve": equity_curve,
        "returns": ret_df,
        "metrics": metrics,
        "weights_history": weights_history,
    }


def plot_backtest(
    equity_curve: pd.Series,
    metrics: dict,
    title: str = "Long-Short Backtest",
    save_path: str | None = None,
):
    """Plot backtest equity curve with key metrics."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Equity curve
    ax = axes[0]
    ax.plot(equity_curve.index, equity_curve.values, color="steelblue", linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)

    # Annotate metrics
    text = (
        f"AR: {metrics['annual_return']:.2%}  |  "
        f"Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
        f"MDD: {metrics['max_drawdown']:.2%}  |  "
        f"Calmar: {metrics['calmar_ratio']:.2f}"
    )
    ax.text(0.02, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Drawdown
    ax2 = axes[1]
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="salmon", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Backtest plot saved to {save_path}")

    return fig
