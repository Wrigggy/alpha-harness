"""Factor combination strategies for building composite alpha signals."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from loguru import logger


class FactorCombiner(ABC):
    """Base class for factor combination strategies.

    Subclasses implement fit() to learn weights from historical data
    and combine() to produce a single composite signal.
    """

    @abstractmethod
    def fit(
        self,
        factor_dict: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> None:
        """Learn combination weights from training data.

        Args:
            factor_dict: Mapping of factor name -> DataFrame (timestamp x symbols).
            forward_returns: DataFrame (timestamp x symbols) of forward returns.
        """
        ...

    @abstractmethod
    def combine(self, factor_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Produce a single combined signal from multiple factors.

        Args:
            factor_dict: Mapping of factor name -> DataFrame (timestamp x symbols).

        Returns:
            Combined signal DataFrame (timestamp x symbols).
        """
        ...


class EqualWeightCombiner(FactorCombiner):
    """Simple average of all factors (no fitting needed)."""

    def __init__(self):
        self.factor_names: list[str] = []

    def fit(
        self,
        factor_dict: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> None:
        self.factor_names = list(factor_dict.keys())
        logger.info(
            "EqualWeightCombiner fit with {} factors (weights are uniform)",
            len(self.factor_names),
        )

    def combine(self, factor_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not factor_dict:
            raise ValueError("factor_dict is empty")

        names = list(factor_dict.keys())
        # Start from the first factor, then iteratively add
        result = factor_dict[names[0]].copy()
        for name in names[1:]:
            result = result.add(factor_dict[name], fill_value=0.0)

        result = result / len(names)
        logger.info("EqualWeightCombiner produced signal with shape {}", result.shape)
        return result


class ICWeightedCombiner(FactorCombiner):
    """Weight factors by their historical mean Rank IC."""

    def __init__(self):
        self.weights: dict[str, float] = {}

    def fit(
        self,
        factor_dict: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> None:
        from scipy.stats import spearmanr

        self.weights = {}

        for name, fv in factor_dict.items():
            common_idx = fv.index.intersection(forward_returns.index)
            common_cols = fv.columns.intersection(forward_returns.columns)
            fv_aligned = fv.loc[common_idx, common_cols]
            fr_aligned = forward_returns.loc[common_idx, common_cols]

            rics: list[float] = []
            for t in common_idx:
                a = fv_aligned.loc[t].values.astype(float)
                r = fr_aligned.loc[t].values.astype(float)
                mask = np.isfinite(a) & np.isfinite(r)
                if mask.sum() < 10:
                    continue
                ric, _ = spearmanr(a[mask], r[mask])
                if np.isfinite(ric):
                    rics.append(ric)

            mean_ic = float(np.mean(rics)) if rics else 0.0
            self.weights[name] = abs(mean_ic)

        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        else:
            n = len(self.weights)
            self.weights = {k: 1.0 / n for k in self.weights} if n > 0 else {}

        logger.info("ICWeightedCombiner weights: {}", self.weights)

    def combine(self, factor_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not self.weights:
            raise ValueError("Must call fit() before combine()")

        result: pd.DataFrame | None = None
        for name, fv in factor_dict.items():
            w = self.weights.get(name, 0.0)
            if w == 0.0:
                continue
            weighted = fv * w
            if result is None:
                result = weighted.copy()
            else:
                result = result.add(weighted, fill_value=0.0)

        if result is None:
            raise ValueError("No factors with nonzero weight")

        logger.info("ICWeightedCombiner produced signal with shape {}", result.shape)
        return result


class RidgeCombiner(FactorCombiner):
    """Ridge regression to learn optimal factor weights.

    Each (timestamp, symbol) pair is treated as an observation. Features are
    the factor values and the target is the forward return.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model: Ridge | None = None
        self.factor_names: list[str] = []

    def fit(
        self,
        factor_dict: dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
    ) -> None:
        self.factor_names = list(factor_dict.keys())

        # Align all factors to common index and columns
        common_idx = forward_returns.index
        common_cols = forward_returns.columns
        for fv in factor_dict.values():
            common_idx = common_idx.intersection(fv.index)
            common_cols = common_cols.intersection(fv.columns)

        # Build feature matrix: each row is a (timestamp, symbol) observation
        n_timestamps = len(common_idx)
        n_symbols = len(common_cols)
        n_obs = n_timestamps * n_symbols
        n_factors = len(self.factor_names)

        X = np.empty((n_obs, n_factors), dtype=np.float64)
        for j, name in enumerate(self.factor_names):
            X[:, j] = factor_dict[name].loc[common_idx, common_cols].values.flatten()

        y = forward_returns.loc[common_idx, common_cols].values.flatten().astype(np.float64)

        # Drop rows with any NaN
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X_clean = X[mask]
        y_clean = y[mask]

        if len(X_clean) < n_factors + 1:
            logger.warning(
                "Too few valid observations ({}) for Ridge fit, falling back to equal weights",
                len(X_clean),
            )
            self.model = None
            return

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_clean, y_clean)

        weight_str = ", ".join(
            f"{name}={w:.4f}"
            for name, w in zip(self.factor_names, self.model.coef_)
        )
        logger.info(
            "RidgeCombiner fit on {} observations | weights: {}",
            len(X_clean),
            weight_str,
        )

    def combine(self, factor_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        if self.model is None:
            # Fallback to equal weight
            logger.warning("RidgeCombiner has no fitted model, using equal weights")
            combiner = EqualWeightCombiner()
            combiner.factor_names = self.factor_names
            return combiner.combine(factor_dict)

        # Align to common index/columns across all factors provided
        names = self.factor_names
        ref = factor_dict[names[0]]
        common_idx = ref.index
        common_cols = ref.columns
        for name in names[1:]:
            common_idx = common_idx.intersection(factor_dict[name].index)
            common_cols = common_cols.intersection(factor_dict[name].columns)

        n_timestamps = len(common_idx)
        n_symbols = len(common_cols)
        n_factors = len(names)

        X = np.empty((n_timestamps * n_symbols, n_factors), dtype=np.float64)
        for j, name in enumerate(names):
            X[:, j] = factor_dict[name].loc[common_idx, common_cols].values.flatten()

        predictions = self.model.predict(X)
        result = pd.DataFrame(
            predictions.reshape(n_timestamps, n_symbols),
            index=common_idx,
            columns=common_cols,
        )

        logger.info("RidgeCombiner produced signal with shape {}", result.shape)
        return result
