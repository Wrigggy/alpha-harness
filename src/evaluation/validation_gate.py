"""Formalized validation gates for candidate alpha factors."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from loguru import logger

from src.evaluation.ic_analysis import evaluate_factor


@dataclass
class ValidationConfig:
    min_rank_ic: float = 0.03
    min_rank_icir: float = 0.5
    max_turnover: float = 0.3
    min_decay_halflife: int = 3  # bars
    max_pool_correlation: float = 0.7
    min_judge_score: float = 0.3


@dataclass
class ValidationResult:
    passed: bool
    metrics: dict  # all computed metrics
    failures: list[str]  # which gates failed


def _compute_turnover(factor_values: pd.DataFrame) -> float:
    """Compute mean absolute change in cross-sectional ranks per period.

    For each timestamp pair (t, t+1), rank the symbols cross-sectionally,
    then compute the mean absolute rank change (normalized to [0, 1]).
    """
    ranks = factor_values.rank(axis=1, pct=True)
    rank_diff = ranks.diff().abs()
    # Mean across symbols, then mean across timestamps
    turnover_per_period = rank_diff.mean(axis=1)
    return float(turnover_per_period.mean())


def _estimate_decay_halflife(
    factor_values: pd.DataFrame,
    forward_returns: pd.DataFrame,
    horizons: list[int] | None = None,
    min_observations: int = 10,
) -> int:
    """Estimate IC decay half-life.

    Computes Rank IC at horizons [1, 2, 4, 8, 16] by shifting forward_returns.
    Finds the first horizon where IC drops below IC_1 / 2.
    Returns the horizon as the half-life estimate, or the max horizon if IC
    never drops below IC_1 / 2.
    """
    if horizons is None:
        horizons = [1, 2, 4, 8, 16]

    common_idx = factor_values.index.intersection(forward_returns.index)
    common_cols = factor_values.columns.intersection(forward_returns.columns)
    fv = factor_values.loc[common_idx, common_cols]

    # Build cumulative forward returns at each horizon from 1-period forward returns
    fr_1 = forward_returns.loc[common_idx, common_cols]

    ic_by_horizon: dict[int, float] = {}
    for h in horizons:
        # Cumulative forward return over h periods: shift fr by 0..h-1 and sum
        cum_ret = pd.DataFrame(0.0, index=common_idx, columns=common_cols)
        for lag in range(h):
            cum_ret += fr_1.shift(-lag)
        # The last h-1 rows will have NaN contributions; that's handled by masking below

        rics: list[float] = []
        for t in common_idx:
            a = fv.loc[t].values.astype(float)
            r = cum_ret.loc[t].values.astype(float)
            mask = np.isfinite(a) & np.isfinite(r)
            if mask.sum() < min_observations:
                continue
            ric, _ = spearmanr(a[mask], r[mask])
            if np.isfinite(ric):
                rics.append(ric)

        ic_by_horizon[h] = float(np.mean(rics)) if rics else 0.0

    if not ic_by_horizon:
        return 0

    ic_1 = ic_by_horizon.get(horizons[0], 0.0)
    half_threshold = abs(ic_1) / 2.0

    for h in horizons:
        if abs(ic_by_horizon[h]) < half_threshold:
            return h

    return horizons[-1]


def _max_pool_correlation(
    factor_values: pd.DataFrame,
    existing_pool: list[pd.DataFrame],
) -> float:
    """Compute the maximum absolute correlation between a candidate and existing pool members."""
    if not existing_pool:
        return 0.0

    candidate = factor_values.values.flatten().astype(float)
    cand_mask_base = np.isfinite(candidate)

    max_corr = 0.0
    for pool_member in existing_pool:
        # Align to common index/columns
        common_idx = factor_values.index.intersection(pool_member.index)
        common_cols = factor_values.columns.intersection(pool_member.columns)
        if len(common_idx) == 0 or len(common_cols) == 0:
            continue

        a = factor_values.loc[common_idx, common_cols].values.flatten().astype(float)
        b = pool_member.loc[common_idx, common_cols].values.flatten().astype(float)
        mask = np.isfinite(a) & np.isfinite(b)

        if mask.sum() < 10:
            continue

        corr = abs(float(np.corrcoef(a[mask], b[mask])[0, 1]))
        if corr > max_corr:
            max_corr = corr

    return max_corr


class ValidationGate:
    """Run a battery of validation gates on a candidate alpha factor.

    Gates:
    - Rank IC magnitude
    - Rank ICIR magnitude
    - Portfolio turnover
    - IC decay half-life
    - Maximum correlation with existing pool
    - LLM judge score (optional)
    """

    def __init__(self, config: ValidationConfig | None = None):
        self.config = config or ValidationConfig()

    def validate(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.DataFrame,
        existing_pool: list[pd.DataFrame] | None = None,
        judge_score: float | None = None,
    ) -> ValidationResult:
        """Run all validation gates on a candidate factor.

        Args:
            factor_values: DataFrame (timestamp x symbols) of factor values.
            forward_returns: DataFrame (timestamp x symbols) of 1-period forward returns.
            existing_pool: List of DataFrames for existing pool members (for correlation check).
            judge_score: Optional LLM judge score in [0, 1].

        Returns:
            ValidationResult with pass/fail, all metrics, and list of failures.
        """
        cfg = self.config
        failures: list[str] = []
        metrics: dict = {}

        # --- IC metrics ---
        ic_metrics = evaluate_factor(factor_values, forward_returns)
        metrics["rank_ic"] = ic_metrics.rank_ic_mean
        metrics["rank_icir"] = ic_metrics.rank_icir
        metrics["ic"] = ic_metrics.ic_mean
        metrics["icir"] = ic_metrics.icir

        if abs(ic_metrics.rank_ic_mean) < cfg.min_rank_ic:
            failures.append(
                f"rank_ic={ic_metrics.rank_ic_mean:.4f} < min={cfg.min_rank_ic}"
            )
        if abs(ic_metrics.rank_icir) < cfg.min_rank_icir:
            failures.append(
                f"rank_icir={ic_metrics.rank_icir:.4f} < min={cfg.min_rank_icir}"
            )

        # --- Turnover ---
        turnover = _compute_turnover(factor_values)
        metrics["turnover"] = turnover
        if turnover > cfg.max_turnover:
            failures.append(
                f"turnover={turnover:.4f} > max={cfg.max_turnover}"
            )

        # --- IC decay half-life ---
        halflife = _estimate_decay_halflife(factor_values, forward_returns)
        metrics["decay_halflife"] = halflife
        if halflife < cfg.min_decay_halflife:
            failures.append(
                f"decay_halflife={halflife} < min={cfg.min_decay_halflife}"
            )

        # --- Pool correlation ---
        if existing_pool:
            max_corr = _max_pool_correlation(factor_values, existing_pool)
            metrics["max_pool_correlation"] = max_corr
            if max_corr > cfg.max_pool_correlation:
                failures.append(
                    f"max_pool_correlation={max_corr:.4f} > max={cfg.max_pool_correlation}"
                )
        else:
            metrics["max_pool_correlation"] = 0.0

        # --- Judge score ---
        if judge_score is not None:
            metrics["judge_score"] = judge_score
            if judge_score < cfg.min_judge_score:
                failures.append(
                    f"judge_score={judge_score:.4f} < min={cfg.min_judge_score}"
                )

        passed = len(failures) == 0

        if passed:
            logger.info("Validation PASSED | metrics={}", metrics)
        else:
            logger.warning("Validation FAILED | failures={} | metrics={}", failures, metrics)

        return ValidationResult(passed=passed, metrics=metrics, failures=failures)
