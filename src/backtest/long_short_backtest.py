"""Top-N long / bottom-N short portfolio backtest."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src.backtest.performance_metrics import compute_metrics, compute_period_metrics


def _empty_result(bars_per_year: int, booksize: float) -> dict:
    empty_index = pd.Index([], name="date")
    daily = pd.DataFrame(
        columns=[
            "return",
            "pnl",
            "turnover",
            "cost",
            "gross_pnl",
            "long_count",
            "short_count",
            "equity",
            "cum_pnl",
            "drawdown",
            "rolling_sharpe",
        ],
        index=empty_index,
    )
    metrics = compute_metrics(pd.Series(dtype=float), booksize=booksize, bars_per_year=bars_per_year)
    yearly = pd.DataFrame(
        columns=[
            "year",
            "sharpe",
            "turnover",
            "fitness",
            "returns",
            "drawdown",
            "margin",
            "long_count",
            "short_count",
        ]
    )
    return {
        "equity_curve": pd.Series(dtype=float, name="equity"),
        "returns": pd.DataFrame(columns=["return"]),
        "metrics": metrics,
        "weights_history": pd.DataFrame(),
        "daily_metrics": daily,
        "yearly_metrics": yearly,
        "aggregate_metrics": {},
    }


def _rolling_sharpe(returns: pd.Series, window: int, bars_per_year: int) -> pd.Series:
    if returns.empty:
        return pd.Series(dtype=float)
    roll_mean = returns.rolling(window=window, min_periods=max(5, window // 5)).mean()
    roll_std = returns.rolling(window=window, min_periods=max(5, window // 5)).std()
    annualized_mean = roll_mean * bars_per_year
    annualized_std = roll_std * np.sqrt(bars_per_year)
    sharpe = annualized_mean / annualized_std.replace(0.0, np.nan)
    return sharpe.replace([np.inf, -np.inf], np.nan)


def _build_yearly_metrics(
    daily_metrics: pd.DataFrame,
    booksize: float,
    bars_per_year: int,
) -> pd.DataFrame:
    if daily_metrics.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "sharpe",
                "turnover",
                "fitness",
                "returns",
                "drawdown",
                "margin",
                "long_count",
                "short_count",
            ]
        )

    rows: list[dict] = []
    grouped = daily_metrics.groupby(daily_metrics.index.year)
    for year, frame in grouped:
        period = compute_period_metrics(
            period_returns=frame["return"],
            period_turnover=frame["turnover"],
            period_pnl=frame["pnl"],
            period_long_count=frame["long_count"],
            period_short_count=frame["short_count"],
            booksize=booksize,
            bars_per_year=bars_per_year,
        )
        rows.append(
            {
                "year": int(year),
                "sharpe": float(period["sharpe_ratio"]),
                "turnover": float(period["avg_turnover"]),
                "fitness": float(period["fitness"]),
                "returns": float(period["annual_return"]),
                "drawdown": float(period["max_drawdown"]),
                "margin": float(period["margin"]),
                "long_count": float(period.get("avg_long_count", 0.0)),
                "short_count": float(period.get("avg_short_count", 0.0)),
            }
        )

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def _build_aggregate_metrics(metrics: dict, yearly_metrics: pd.DataFrame) -> dict:
    aggregate = {
        "Sharpe": float(metrics["sharpe_ratio"]),
        "Turnover": float(metrics["avg_turnover"]),
        "Fitness": float(metrics["fitness"]),
        "Returns": float(metrics["annual_return"]),
        "Drawdown": float(metrics["max_drawdown"]),
        "Margin": float(metrics["margin"]),
    }
    if not yearly_metrics.empty:
        aggregate["Years"] = int(len(yearly_metrics))
    return aggregate


def long_short_backtest(
    factor_values: pd.DataFrame,
    returns: pd.DataFrame,
    n_long: int = 30,
    n_short: int = 30,
    liquidity_filter: pd.DataFrame | None = None,
    min_liquidity_percentile: float = 0.2,
    rebalance_freq: int = 1,
    transaction_cost_bps: float = 5.0,
    bars_per_year: int = 8760,
    booksize: float = 1.0,
) -> dict:
    """Run a long-short backtest with WQ-style proxy diagnostics."""
    common_idx = factor_values.index.intersection(returns.index)
    common_cols = factor_values.columns.intersection(returns.columns)
    fv = factor_values.loc[common_idx, common_cols]
    ret = returns.loc[common_idx, common_cols]

    if fv.empty or ret.empty:
        return _empty_result(bars_per_year=bars_per_year, booksize=booksize)

    liq = None
    if liquidity_filter is not None:
        liq = liquidity_filter.loc[common_idx, common_cols]

    tc = transaction_cost_bps / 10000.0
    rebalance_dates = set(common_idx[::rebalance_freq])

    daily_rows: list[dict] = []
    weights_records: list[dict] = []
    prev_weights = pd.Series(0.0, index=common_cols, dtype=float)
    current_long_count = 0
    current_short_count = 0

    for date in common_idx:
        if date in rebalance_dates:
            fv_bar = fv.loc[date]
            valid_mask = fv_bar.notna()

            if liq is not None:
                liq_bar = liq.loc[date]
                if liq_bar.notna().sum() > 0:
                    threshold = liq_bar.quantile(min_liquidity_percentile)
                    valid_mask = valid_mask & (liq_bar >= threshold)

            valid_symbols = fv_bar[valid_mask].dropna()
            if len(valid_symbols) >= n_long + n_short:
                ranked = valid_symbols.rank(ascending=False, method="first")
                long_symbols = ranked.nsmallest(n_long).index
                short_symbols = ranked.nlargest(n_short).index

                new_weights = pd.Series(0.0, index=common_cols, dtype=float)
                if n_long > 0:
                    new_weights.loc[long_symbols] = 1.0 / n_long
                if n_short > 0:
                    new_weights.loc[short_symbols] = -1.0 / n_short
                current_long_count = int((new_weights > 0).sum())
                current_short_count = int((new_weights < 0).sum())
            else:
                new_weights = prev_weights.copy()

            turnover = float((new_weights - prev_weights).abs().sum())
            tc_cost = turnover * tc
            bar_ret = ret.loc[date].fillna(0.0)
            gross_ret = float((prev_weights * bar_ret).sum())
            net_ret = gross_ret - tc_cost
            pnl = net_ret * booksize
            gross_pnl = gross_ret * booksize

            daily_rows.append(
                {
                    "date": date,
                    "return": net_ret,
                    "pnl": pnl,
                    "turnover": turnover,
                    "cost": tc_cost * booksize,
                    "gross_pnl": gross_pnl,
                    "long_count": current_long_count,
                    "short_count": current_short_count,
                }
            )
            weights_records.append({"date": date, **new_weights.to_dict()})
            prev_weights = new_weights
            continue

        bar_ret = ret.loc[date].fillna(0.0)
        gross_ret = float((prev_weights * bar_ret).sum())
        pnl = gross_ret * booksize
        daily_rows.append(
            {
                "date": date,
                "return": gross_ret,
                "pnl": pnl,
                "turnover": 0.0,
                "cost": 0.0,
                "gross_pnl": pnl,
                "long_count": current_long_count,
                "short_count": current_short_count,
            }
        )

    if not daily_rows:
        return _empty_result(bars_per_year=bars_per_year, booksize=booksize)

    daily_metrics = pd.DataFrame(daily_rows).set_index("date").sort_index()
    daily_metrics.index = pd.to_datetime(daily_metrics.index)
    daily_metrics["equity"] = (1.0 + daily_metrics["return"]).cumprod()
    daily_metrics["cum_pnl"] = daily_metrics["pnl"].cumsum()
    running_max = daily_metrics["equity"].cummax()
    daily_metrics["drawdown"] = daily_metrics["equity"] / running_max - 1.0
    sharpe_window = min(max(20, bars_per_year // 4), max(len(daily_metrics), 20))
    daily_metrics["rolling_sharpe"] = _rolling_sharpe(
        daily_metrics["return"], window=sharpe_window, bars_per_year=bars_per_year
    )

    equity_curve = daily_metrics["equity"].rename("equity")
    weights_history = pd.DataFrame(weights_records).set_index("date") if weights_records else pd.DataFrame()

    metrics = compute_metrics(
        equity_curve=equity_curve,
        pnl_series=daily_metrics["pnl"],
        turnover_series=daily_metrics["turnover"],
        long_count_series=daily_metrics["long_count"],
        short_count_series=daily_metrics["short_count"],
        booksize=booksize,
        bars_per_year=bars_per_year,
    )
    yearly_metrics = _build_yearly_metrics(daily_metrics, booksize=booksize, bars_per_year=bars_per_year)
    aggregate_metrics = _build_aggregate_metrics(metrics, yearly_metrics)

    logger.info(
        "Backtest complete: Returns={:.2%}, Sharpe={:.2f}, Turnover={:.2%}, MDD={:.2%}",
        metrics["annual_return"],
        metrics["sharpe_ratio"],
        metrics["avg_turnover"],
        metrics["max_drawdown"],
    )

    return {
        "equity_curve": equity_curve,
        "returns": daily_metrics[["return"]],
        "metrics": metrics,
        "weights_history": weights_history,
        "daily_metrics": daily_metrics,
        "yearly_metrics": yearly_metrics,
        "aggregate_metrics": aggregate_metrics,
    }


def plot_backtest(
    equity_curve: pd.Series,
    metrics: dict,
    title: str = "Long-Short Backtest",
    save_path: str | None = None,
):
    """Plot backtest equity curve with key metrics."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(equity_curve.index, equity_curve.values, color="steelblue", linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)

    text = (
        f"Returns: {metrics['annual_return']:.2%}  |  "
        f"Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
        f"Turnover: {metrics['avg_turnover']:.2%}  |  "
        f"MDD: {metrics['max_drawdown']:.2%}  |  "
        f"Fitness: {metrics.get('fitness', 0.0):.2f}"
    )
    ax.text(
        0.02,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax2 = axes[1]
    cummax = equity_curve.cummax()
    drawdown = equity_curve / cummax - 1.0
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="salmon", alpha=0.5)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Backtest plot saved to {}", save_path)

    return fig


def plot_worldquant_style_panels(
    daily_metrics: pd.DataFrame,
    title: str = "WorldQuant-Style Backtest Panels",
    save_path: str | None = None,
):
    """Plot PnL, rolling Sharpe, and Turnover time series."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    axes[0].plot(daily_metrics.index, daily_metrics["cum_pnl"], color="teal", linewidth=1.5)
    axes[0].set_title(title)
    axes[0].set_ylabel("Cum PnL")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(daily_metrics.index, daily_metrics["rolling_sharpe"], color="darkorange", linewidth=1.2)
    axes[1].axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    axes[1].set_ylabel("Rolling Sharpe")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(daily_metrics.index, daily_metrics["turnover"], color="slateblue", linewidth=1.2)
    axes[2].set_ylabel("Turnover")
    axes[2].set_xlabel("Date")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("WorldQuant-style panel plot saved to {}", save_path)

    return fig
