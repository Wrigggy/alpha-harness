"""Approximate WorldQuant BRAIN-style single-alpha evaluation on local panels.

This module does not claim to reproduce BRAIN's proprietary scoring. It aligns
local evaluation with common BRAIN constraints:
- cross-sectional daily stock signals
- delay before trading
- liquidity-aware universe filtering
- turnover and coverage diagnostics
- single-alpha candidate scoring instead of only portfolio-level pool scoring
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.backtest.long_short_backtest import long_short_backtest
from src.data_adapter.to_alphagen_format import (
    PanelAlphaCalculator,
    _load_local_panel,
    create_data_splits,
)
from src.evaluation.ic_analysis import evaluate_factor


def _parse_csv_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


@dataclass
class BrainProxyConfig:
    """Configuration for local BRAIN-style proxy evaluation."""

    forward_horizon_bars: int = 1
    delay_bars: int = 1
    signal_rank: bool = True
    signal_demean: bool = True
    signal_standardize: bool = False
    winsorize_quantile: float = 0.01
    cap_neutral_field: str | None = "market_cap"
    industry_neutral_field: str | None = "industry"
    fallback_group_field: str | None = "board"
    liquidity_field: str = "amount"
    universe_field: str | None = None
    sub_universe_field: str | None = None
    min_liquidity_percentile: float = 0.3
    sub_universe_percentile: float = 0.5
    transaction_cost_bps: float = 10.0
    n_long: int = 25
    n_short: int = 25
    rebalance_freq: int = 1
    min_rank_ic: float = 0.01
    min_rank_icir: float = 0.10
    min_coverage: float = 0.60
    max_avg_turnover: float = 0.70
    min_sharpe: float = 0.0
    min_fitness: float = 0.0
    min_sub_universe_sharpe: float = 0.2
    max_weight_concentration: float = 0.10

    @classmethod
    def from_file(cls, path: str = "config/brain_proxy_equity_cn.yaml") -> "BrainProxyConfig":
        cfg_path = Path(path)
        if not cfg_path.exists():
            return cls()
        with cfg_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        signal = raw.get("signal", {})
        backtest = raw.get("backtest", {})
        gates = raw.get("candidate_gates", {})
        return cls(
            forward_horizon_bars=int(raw.get("forward_horizon_bars", 1)),
            delay_bars=int(raw.get("delay_bars", 1)),
            signal_rank=bool(signal.get("rank", True)),
            signal_demean=bool(signal.get("demean", True)),
            signal_standardize=bool(signal.get("standardize", False)),
            winsorize_quantile=float(signal.get("winsorize_quantile", 0.01)),
            cap_neutral_field=signal.get("cap_neutral_field"),
            industry_neutral_field=signal.get("industry_neutral_field"),
            fallback_group_field=signal.get("fallback_group_field", "board"),
            liquidity_field=str(backtest.get("liquidity_field", "amount")),
            universe_field=backtest.get("universe_field"),
            sub_universe_field=backtest.get("sub_universe_field"),
            min_liquidity_percentile=float(backtest.get("min_liquidity_percentile", 0.3)),
            sub_universe_percentile=float(backtest.get("sub_universe_percentile", 0.5)),
            transaction_cost_bps=float(backtest.get("transaction_cost_bps", 10.0)),
            n_long=int(backtest.get("n_long", 25)),
            n_short=int(backtest.get("n_short", 25)),
            rebalance_freq=int(backtest.get("rebalance_freq", 1)),
            min_rank_ic=float(gates.get("min_rank_ic", 0.01)),
            min_rank_icir=float(gates.get("min_rank_icir", 0.10)),
            min_coverage=float(gates.get("min_coverage", 0.60)),
            max_avg_turnover=float(gates.get("max_avg_turnover", 0.70)),
            min_sharpe=float(gates.get("min_sharpe", 0.0)),
            min_fitness=float(gates.get("min_fitness", 0.0)),
            min_sub_universe_sharpe=float(gates.get("min_sub_universe_sharpe", 0.2)),
            max_weight_concentration=float(gates.get("max_weight_concentration", 0.10)),
        )


def build_forward_returns(close_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compute close-to-close forward returns."""
    return close_df.shift(-horizon) / close_df - 1.0


def tensor_to_factor_frame(stock_data, tensor: object, name: str = "factor") -> pd.DataFrame:
    frame = stock_data.make_dataframe(tensor, columns=[name])
    return frame[name].unstack(level=1).astype(float)


def _winsorize_row(values: pd.Series, quantile: float) -> pd.Series:
    if quantile <= 0:
        return values
    finite = values.dropna()
    if finite.empty:
        return values
    lower = finite.quantile(quantile)
    upper = finite.quantile(1.0 - quantile)
    return values.clip(lower=lower, upper=upper)


def _residualize_against_numeric(signal: pd.Series, numeric: pd.Series) -> pd.Series:
    mask = signal.notna() & numeric.notna()
    if mask.sum() < 3:
        return signal
    x = numeric.loc[mask].astype(float).values
    y = signal.loc[mask].astype(float).values
    design = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ beta
    residual = pd.Series(y - fitted, index=signal.loc[mask].index)
    out = signal.copy()
    out.loc[mask] = residual
    return out


def _neutralize_row(
    signal: pd.Series,
    cap_row: pd.Series | None = None,
    industry_row: pd.Series | None = None,
) -> pd.Series:
    out = signal.copy()
    if industry_row is not None:
        group_frame = pd.DataFrame({"signal": out, "industry": industry_row})
        mask = group_frame["signal"].notna() & group_frame["industry"].notna()
        if mask.any():
            group_means = group_frame.loc[mask].groupby("industry")["signal"].transform("mean")
            group_frame.loc[mask, "signal"] = group_frame.loc[mask, "signal"] - group_means
            out = group_frame["signal"]
    if cap_row is not None:
        cap_numeric = pd.to_numeric(cap_row, errors="coerce")
        cap_numeric = np.log1p(cap_numeric.clip(lower=0))
        out = _residualize_against_numeric(out, cap_numeric)
    return out


def prepare_brain_signal(
    factor_df: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    cfg: BrainProxyConfig,
) -> pd.DataFrame:
    """Apply BRAIN-like cross-sectional preprocessing."""
    aligned = factor_df.copy().astype(float)
    cap_df = panel.get(cfg.cap_neutral_field) if cfg.cap_neutral_field else None
    if cap_df is None and cfg.cap_neutral_field == "market_cap" and "market_cap_proxy" in panel:
        cap_df = panel["market_cap_proxy"]
    industry_df = panel.get(cfg.industry_neutral_field) if cfg.industry_neutral_field else None
    if industry_df is None and cfg.fallback_group_field:
        industry_df = panel.get(cfg.fallback_group_field)

    out_rows: list[pd.Series] = []
    for ts in aligned.index:
        row = aligned.loc[ts]
        row = _winsorize_row(row, cfg.winsorize_quantile)

        if cfg.signal_rank:
            row = row.rank(method="average", pct=True)

        cap_row = None
        if cap_df is not None and ts in cap_df.index:
            cap_row = cap_df.loc[ts, aligned.columns]
        industry_row = None
        if industry_df is not None and ts in industry_df.index:
            industry_row = industry_df.loc[ts, aligned.columns]
        row = _neutralize_row(row, cap_row=cap_row, industry_row=industry_row)

        if cfg.signal_demean:
            mean = row.dropna().mean()
            if pd.notna(mean):
                row = row - mean

        if cfg.signal_standardize:
            std = row.dropna().std()
            if pd.notna(std) and std > 1e-12:
                row = row / std

        out_rows.append(row)

    prepared = pd.DataFrame(out_rows, index=aligned.index, columns=aligned.columns)
    if cfg.delay_bars > 0:
        prepared = prepared.shift(cfg.delay_bars)
    return prepared


def _resolve_liquidity_field(panel: dict[str, pd.DataFrame], preferred: str) -> str:
    if preferred in panel:
        return preferred
    fallbacks = ["amount", "turnover", "quote_volume", "volume"]
    for field in fallbacks:
        if field in panel:
            return field
    raise KeyError("No usable liquidity field found in panel")


def _resolve_optional_mask_field(panel: dict[str, pd.DataFrame], field: str | None) -> pd.DataFrame | None:
    if not field:
        return None
    if field not in panel:
        return None
    raw = panel[field]
    if raw.empty:
        return None
    numeric = raw.apply(pd.to_numeric, errors="coerce")
    return numeric.fillna(0.0) > 0.0


def build_proxy_context(
    data_config_path: str,
    split: str,
    brain_config_path: str = "config/brain_proxy_equity_cn.yaml",
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """Load panel, split, returns, and liquidity matrices for proxy evaluation."""
    cfg = BrainProxyConfig.from_file(brain_config_path)
    with open(data_config_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    source_cfg = data_cfg.get("source", {})
    source_name = source_cfg.get("name", "crypto")
    processed_dir = (
        data_cfg["data"]["processed_dir"]
        if source_name == "crypto"
        else source_cfg.get("panel_dir", data_cfg["data"]["processed_dir"])
    )
    panel = _load_local_panel(processed_dir, data_config=data_config_path)
    splits = create_data_splits(
        processed_dir,
        data_config_path,
        device=device,
        max_backtrack_days=100,
        max_future_days=max(5, cfg.forward_horizon_bars),
    )
    stock_data = splits[split]
    target_expr = None
    calculator = PanelAlphaCalculator(stock_data, target_expr)

    date_index = stock_data.make_dataframe(
        torch.zeros((stock_data.n_days, stock_data.n_stocks), device=device),
        ["stub"],
    ).index.levels[0]
    close_df = panel["close"].loc[date_index, stock_data.stock_ids].astype(float)
    forward_returns = build_forward_returns(close_df, cfg.forward_horizon_bars)
    next_bar_returns = build_forward_returns(close_df, 1)

    liquidity_field = _resolve_liquidity_field(panel, cfg.liquidity_field)
    liquidity_df = panel[liquidity_field].loc[close_df.index, close_df.columns].astype(float)
    universe_mask = _resolve_optional_mask_field(panel, cfg.universe_field)
    if universe_mask is not None:
        universe_mask = universe_mask.loc[close_df.index, close_df.columns]
    sub_universe_mask = _resolve_optional_mask_field(panel, cfg.sub_universe_field)
    if sub_universe_mask is not None:
        sub_universe_mask = sub_universe_mask.loc[close_df.index, close_df.columns]

    return {
        "brain_cfg": cfg,
        "panel": panel,
        "stock_data": stock_data,
        "calculator": calculator,
        "close_df": close_df,
        "forward_returns": forward_returns,
        "next_bar_returns": next_bar_returns,
        "liquidity_df": liquidity_df,
        "liquidity_field": liquidity_field,
        "universe_mask": universe_mask,
        "sub_universe_mask": sub_universe_mask,
        "bars_per_year": int(source_cfg.get("bars_per_year", 252)),
        "source_name": source_name,
    }


def _coverage_ratio(signal: pd.DataFrame) -> float:
    if signal.empty:
        return 0.0
    return float(signal.notna().mean(axis=1).mean())


def _weight_concentration(weights_history: pd.DataFrame) -> float:
    if weights_history.empty:
        return 0.0
    per_row = weights_history.abs()
    denom = per_row.sum(axis=1).replace(0.0, np.nan)
    concentration = per_row.max(axis=1) / denom
    concentration = concentration.replace([np.inf, -np.inf], np.nan).dropna()
    return float(concentration.mean()) if not concentration.empty else 0.0


def _sub_universe_mask(liquidity_df: pd.DataFrame, percentile: float = 0.5) -> pd.DataFrame:
    if liquidity_df.empty:
        return pd.DataFrame(False, index=liquidity_df.index, columns=liquidity_df.columns)
    thresholds = liquidity_df.quantile(percentile, axis=1)
    return liquidity_df.ge(thresholds, axis=0) & liquidity_df.notna()


def _run_sub_universe_backtest(
    signal: pd.DataFrame,
    next_bar_returns: pd.DataFrame,
    liquidity_df: pd.DataFrame,
    cfg: BrainProxyConfig,
    bars_per_year: int,
    sub_universe_mask: pd.DataFrame | None = None,
) -> dict:
    mask = (
        sub_universe_mask.reindex(index=signal.index, columns=signal.columns).fillna(False)
        if sub_universe_mask is not None
        else _sub_universe_mask(liquidity_df, percentile=cfg.sub_universe_percentile)
    )
    masked_signal = signal.where(mask)
    masked_returns = next_bar_returns.where(mask)
    available_counts = mask.sum(axis=1)
    median_available = int(available_counts.median()) if not available_counts.empty else 0
    n_leg = max(1, min(cfg.n_long, median_available // 2))
    return long_short_backtest(
        masked_signal,
        masked_returns,
        n_long=n_leg,
        n_short=n_leg,
        liquidity_filter=None,
        min_liquidity_percentile=0.0,
        rebalance_freq=cfg.rebalance_freq,
        transaction_cost_bps=cfg.transaction_cost_bps,
        bars_per_year=bars_per_year,
    )


def _build_readiness_score(metrics: dict[str, float], cfg: BrainProxyConfig) -> float:
    score = 0.0
    score += min(abs(metrics.get("rank_ic", 0.0)) / max(cfg.min_rank_ic, 1e-6), 2.0) * 0.30
    score += min(abs(metrics.get("rank_icir", 0.0)) / max(cfg.min_rank_icir, 1e-6), 2.0) * 0.20
    score += min(metrics.get("coverage", 0.0) / max(cfg.min_coverage, 1e-6), 1.5) * 0.15
    score += min(max(metrics.get("sharpe_ratio", 0.0), 0.0) / max(cfg.min_sharpe or 0.5, 0.5), 2.0) * 0.15
    turnover_headroom = max(0.0, cfg.max_avg_turnover - metrics.get("avg_turnover", cfg.max_avg_turnover))
    score += min(turnover_headroom / max(cfg.max_avg_turnover, 1e-6), 1.0) * 0.10
    score += min(max(metrics.get("liquidity_pass_rate", 0.0), 0.0), 1.0) * 0.10
    return float(score)


def evaluate_brain_candidate(
    expression: str,
    factor_df: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    forward_returns: pd.DataFrame,
    next_bar_returns: pd.DataFrame,
    liquidity_df: pd.DataFrame,
    bars_per_year: int,
    cfg: BrainProxyConfig,
    preprocessed_signal: bool = False,
    universe_mask: pd.DataFrame | None = None,
    sub_universe_mask: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate a single factor as a BRAIN-style candidate."""
    effective_factor = factor_df
    effective_forward_returns = forward_returns
    effective_next_bar_returns = next_bar_returns
    effective_liquidity = liquidity_df

    if universe_mask is not None:
        aligned_mask = universe_mask.reindex(index=factor_df.index, columns=factor_df.columns).fillna(False)
        effective_factor = factor_df.where(aligned_mask)
        effective_forward_returns = forward_returns.where(aligned_mask)
        effective_next_bar_returns = next_bar_returns.where(aligned_mask)
        effective_liquidity = liquidity_df.where(aligned_mask)

    signal = effective_factor if preprocessed_signal else prepare_brain_signal(effective_factor, panel=panel, cfg=cfg)
    ic_metrics = evaluate_factor(signal, effective_forward_returns, min_observations=max(3, min(15, signal.shape[1])))
    backtest = long_short_backtest(
        signal,
        effective_next_bar_returns,
        n_long=min(cfg.n_long, max(1, signal.shape[1] // 2)),
        n_short=min(cfg.n_short, max(1, signal.shape[1] // 2)),
        liquidity_filter=effective_liquidity,
        min_liquidity_percentile=cfg.min_liquidity_percentile,
        rebalance_freq=cfg.rebalance_freq,
        transaction_cost_bps=cfg.transaction_cost_bps,
        bars_per_year=bars_per_year,
    )

    liq_mask = effective_liquidity.notna()
    liq_thresholds = effective_liquidity.quantile(cfg.min_liquidity_percentile, axis=1)
    pass_matrix = effective_liquidity.ge(liq_thresholds, axis=0) & liq_mask
    liquidity_pass_rate = float(pass_matrix.mean(axis=1).mean()) if not pass_matrix.empty else 0.0
    sub_universe_backtest = _run_sub_universe_backtest(
        signal=signal,
        next_bar_returns=effective_next_bar_returns,
        liquidity_df=effective_liquidity,
        cfg=cfg,
        bars_per_year=bars_per_year,
        sub_universe_mask=sub_universe_mask,
    )

    metrics = {
        "rank_ic": ic_metrics.rank_ic_mean,
        "rank_icir": ic_metrics.rank_icir,
        "ic": ic_metrics.ic_mean,
        "icir": ic_metrics.icir,
        "coverage": _coverage_ratio(signal),
        "liquidity_pass_rate": liquidity_pass_rate,
        "weight_concentration": _weight_concentration(backtest["weights_history"]),
        "sub_universe_sharpe": float(sub_universe_backtest["metrics"].get("sharpe_ratio", 0.0)),
        **backtest["metrics"],
    }
    metrics["brain_readiness_score"] = _build_readiness_score(metrics, cfg)
    metrics["passes_proxy_gates"] = bool(
        abs(metrics["rank_ic"]) >= cfg.min_rank_ic
        and abs(metrics["rank_icir"]) >= cfg.min_rank_icir
        and metrics["coverage"] >= cfg.min_coverage
        and metrics["avg_turnover"] <= cfg.max_avg_turnover
        and metrics["sharpe_ratio"] >= cfg.min_sharpe
        and metrics.get("fitness", 0.0) >= cfg.min_fitness
        and metrics.get("sub_universe_sharpe", 0.0) >= cfg.min_sub_universe_sharpe
        and metrics.get("weight_concentration", 1.0) <= cfg.max_weight_concentration
    )
    metrics["passes_submission_gates"] = bool(
        metrics["coverage"] >= cfg.min_coverage
        and metrics["avg_turnover"] <= cfg.max_avg_turnover
        and metrics["sharpe_ratio"] >= cfg.min_sharpe
        and metrics.get("fitness", 0.0) >= cfg.min_fitness
        and metrics.get("sub_universe_sharpe", 0.0) >= cfg.min_sub_universe_sharpe
        and metrics.get("weight_concentration", 1.0) <= cfg.max_weight_concentration
    )

    summary = {
        "expression": expression,
        **metrics,
    }
    return signal, summary


def compute_signal_correlation_matrix(signal_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute flattened pairwise correlation across prepared signals."""
    names = list(signal_map.keys())
    corr = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            left = signal_map[names[i]]
            right = signal_map[names[j]]
            common_idx = left.index.intersection(right.index)
            common_cols = left.columns.intersection(right.columns)
            a = left.loc[common_idx, common_cols].to_numpy().astype(float).ravel()
            b = right.loc[common_idx, common_cols].to_numpy().astype(float).ravel()
            mask = np.isfinite(a) & np.isfinite(b)
            value = 0.0
            if mask.sum() >= 10:
                value = float(np.corrcoef(a[mask], b[mask])[0, 1])
            corr[i, j] = value
            corr[j, i] = value
    return pd.DataFrame(corr, index=names, columns=names)
