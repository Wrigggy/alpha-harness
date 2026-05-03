"""Portfolio performance metrics for backtest evaluation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

DEFAULT_BARS_PER_YEAR = 8760
MIN_TURNOVER_FLOOR = 0.125


def _safe_float(value: float | int | np.floating) -> float:
    if value is None or not np.isfinite(value):
        return 0.0
    return float(value)


def _compute_drawdown(curve: pd.Series) -> tuple[pd.Series, float]:
    if curve.empty:
        return pd.Series(dtype=float), 0.0
    cummax = curve.cummax()
    drawdown = curve / cummax - 1.0
    max_drawdown = abs(_safe_float(drawdown.min()))
    return drawdown, max_drawdown


def compute_turnover(weights_history: pd.DataFrame) -> float:
    """Compute average portfolio turnover per rebalance."""
    if weights_history.empty:
        return 0.0
    diffs = weights_history.diff().abs()
    turnover_per_bar = diffs.sum(axis=1).dropna()
    return _safe_float(turnover_per_bar.mean())


def compute_fitness(sharpe_ratio: float, annual_return: float, avg_turnover: float) -> float:
    """Approximate WorldQuant-style fitness score."""
    turnover_floor = max(abs(avg_turnover), MIN_TURNOVER_FLOOR)
    if turnover_floor <= 0:
        return 0.0
    return _safe_float(sharpe_ratio * math.sqrt(abs(annual_return) / turnover_floor))


def compute_margin(total_pnl: float, total_traded: float) -> float:
    """Compute PnL per traded dollar."""
    if total_traded <= 0:
        return 0.0
    return _safe_float(total_pnl / total_traded)


def compute_metrics(
    equity_curve: pd.Series,
    pnl_series: pd.Series | None = None,
    turnover_series: pd.Series | None = None,
    long_count_series: pd.Series | None = None,
    short_count_series: pd.Series | None = None,
    booksize: float = 1.0,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
) -> dict:
    """Compute portfolio metrics from an equity curve and daily proxy series."""
    if equity_curve.empty:
        empty = {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "win_rate": 0.0,
            "avg_turnover": 0.0,
            "fitness": 0.0,
            "margin": 0.0,
            "mean_pnl": 0.0,
            "total_pnl": 0.0,
            "total_traded": 0.0,
            "n_bars": 0,
            "bars_per_year": bars_per_year,
        }
        if long_count_series is not None:
            empty["avg_long_count"] = 0.0
        if short_count_series is not None:
            empty["avg_short_count"] = 0.0
        return empty

    returns = equity_curve.pct_change().dropna()
    total_return = _safe_float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)
    n_bars = len(equity_curve)
    annual_return = (1.0 + total_return) ** (bars_per_year / max(n_bars - 1, 1)) - 1.0
    annual_vol = _safe_float(returns.std() * np.sqrt(bars_per_year))
    sharpe = _safe_float(annual_return / annual_vol) if annual_vol > 0 else 0.0

    drawdown, max_drawdown = _compute_drawdown(equity_curve)
    calmar = _safe_float(annual_return / max_drawdown) if max_drawdown > 0 else 0.0
    win_rate = _safe_float((returns > 0).mean()) if not returns.empty else 0.0

    pnl = pnl_series.reindex(equity_curve.index).fillna(0.0) if pnl_series is not None else returns * booksize
    turnover = (
        turnover_series.reindex(equity_curve.index).fillna(0.0)
        if turnover_series is not None
        else pd.Series(0.0, index=equity_curve.index)
    )
    avg_turnover = _safe_float(turnover.mean())
    total_pnl = _safe_float(pnl.sum())
    total_traded = _safe_float((turnover * booksize).sum())
    margin = compute_margin(total_pnl=total_pnl, total_traded=total_traded)
    fitness = compute_fitness(sharpe_ratio=sharpe, annual_return=annual_return, avg_turnover=avg_turnover)

    metrics = {
        "total_return": total_return,
        "annual_return": _safe_float(annual_return),
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "avg_turnover": avg_turnover,
        "fitness": fitness,
        "margin": margin,
        "mean_pnl": _safe_float(pnl.mean()),
        "total_pnl": total_pnl,
        "total_traded": total_traded,
        "n_bars": n_bars,
        "bars_per_year": bars_per_year,
        "drawdown_series_min": _safe_float(drawdown.min()) if not drawdown.empty else 0.0,
    }

    if long_count_series is not None and not long_count_series.empty:
        metrics["avg_long_count"] = _safe_float(long_count_series.mean())
    if short_count_series is not None and not short_count_series.empty:
        metrics["avg_short_count"] = _safe_float(short_count_series.mean())

    return metrics


def compute_period_metrics(
    period_returns: pd.Series,
    period_turnover: pd.Series,
    period_pnl: pd.Series,
    period_long_count: pd.Series | None = None,
    period_short_count: pd.Series | None = None,
    booksize: float = 1.0,
    bars_per_year: int = DEFAULT_BARS_PER_YEAR,
) -> dict:
    """Compute metrics for an arbitrary sub-period such as one calendar year."""
    period_returns = period_returns.fillna(0.0)
    curve = (1.0 + period_returns).cumprod()
    if not curve.empty and curve.iloc[0] != 0:
        curve = curve / curve.iloc[0]
    metrics = compute_metrics(
        equity_curve=curve,
        pnl_series=period_pnl.fillna(0.0),
        turnover_series=period_turnover.fillna(0.0),
        long_count_series=period_long_count,
        short_count_series=period_short_count,
        booksize=booksize,
        bars_per_year=bars_per_year,
    )
    return metrics
