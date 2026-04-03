"""Portfolio performance metrics for backtest evaluation."""

import numpy as np
import pandas as pd

# 1H bars per year (365 * 24)
BARS_PER_YEAR = 8760


def compute_metrics(equity_curve: pd.Series) -> dict:
    """Compute comprehensive performance metrics from an equity curve.

    Args:
        equity_curve: Series of cumulative portfolio values (starting at 1.0).

    Returns:
        Dict of performance metrics.
    """
    returns = equity_curve.pct_change().dropna()

    # Annualized return
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_bars = len(equity_curve)
    annual_return = (1 + total_return) ** (BARS_PER_YEAR / max(n_bars - 1, 1)) - 1

    # Annualized volatility
    annual_vol = returns.std() * np.sqrt(BARS_PER_YEAR)

    # Sharpe ratio (assuming 0 risk-free rate for crypto)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    # Maximum drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = abs(drawdown.min())

    # Calmar ratio
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0

    # Win rate
    win_rate = (returns > 0).mean()

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "n_bars": n_bars,
    }


def compute_turnover(weights_history: pd.DataFrame) -> float:
    """Compute average portfolio turnover per rebalance.

    Args:
        weights_history: DataFrame of portfolio weights (timestamp x symbols).

    Returns:
        Average absolute weight change per rebalance.
    """
    diffs = weights_history.diff().abs()
    turnover_per_bar = diffs.sum(axis=1).dropna()
    return float(turnover_per_bar.mean())
