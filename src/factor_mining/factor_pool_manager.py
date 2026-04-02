"""Manage, deduplicate, and persist discovered alpha factors."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger


@dataclass
class AlphaFactor:
    """A discovered formulaic alpha factor."""
    expression: str
    ic: float = 0.0
    rank_ic: float = 0.0
    weight: float = 0.0
    source: str = ""  # "alphagen" or "alphaqcm"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AlphaFactor":
        return cls(**d)


class FactorPoolManager:
    """Manage a pool of discovered alpha factors with deduplication.

    Enforces:
    - Maximum pool size
    - Minimum IC threshold
    - Maximum pairwise correlation threshold
    """

    def __init__(
        self,
        pool_size: int = 20,
        ic_threshold: float = 0.02,
        correlation_threshold: float = 0.7,
    ):
        self.pool_size = pool_size
        self.ic_threshold = ic_threshold
        self.correlation_threshold = correlation_threshold

        self.factors: list[AlphaFactor] = []
        self._value_cache: dict[str, torch.Tensor] = {}

    def try_add(
        self,
        factor: AlphaFactor,
        factor_values: torch.Tensor,
    ) -> bool:
        """Try to add a factor to the pool.

        The factor is added if:
        1. Its IC exceeds the threshold
        2. Its correlation with all existing pool members is below the threshold
        3. The pool is not full, OR the new factor is better than the weakest member

        Args:
            factor: The alpha factor to add.
            factor_values: Tensor of shape (n_timestamps, n_symbols).

        Returns:
            True if the factor was added, False otherwise.
        """
        # Check IC threshold
        if abs(factor.ic) < self.ic_threshold:
            return False

        # Check correlation with existing members
        for existing in self.factors:
            if existing.expression in self._value_cache:
                corr = self._compute_correlation(
                    factor_values, self._value_cache[existing.expression]
                )
                if abs(corr) > self.correlation_threshold:
                    logger.debug(
                        f"Factor rejected: corr={corr:.3f} with '{existing.expression}'"
                    )
                    return False

        # Pool not full: just add
        if len(self.factors) < self.pool_size:
            self.factors.append(factor)
            self._value_cache[factor.expression] = factor_values.clone()
            logger.info(f"Factor added to pool ({len(self.factors)}/{self.pool_size}): IC={factor.ic:.4f}")
            return True

        # Pool full: replace weakest if new factor is better
        weakest_idx = min(range(len(self.factors)), key=lambda i: abs(self.factors[i].ic))
        if abs(factor.ic) > abs(self.factors[weakest_idx].ic):
            old = self.factors[weakest_idx]
            self._value_cache.pop(old.expression, None)
            self.factors[weakest_idx] = factor
            self._value_cache[factor.expression] = factor_values.clone()
            logger.info(
                f"Factor replaced in pool: new IC={factor.ic:.4f} > old IC={old.ic:.4f}"
            )
            return True

        return False

    def _compute_correlation(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Compute mean cross-sectional correlation between two factor tensors."""
        a_np = a.detach().cpu().numpy().flatten()
        b_np = b.detach().cpu().numpy().flatten()
        mask = np.isfinite(a_np) & np.isfinite(b_np) & (a_np != 0) & (b_np != 0)
        if mask.sum() < 10:
            return 0.0
        return float(np.corrcoef(a_np[mask], b_np[mask])[0, 1])

    def prune(self):
        """Remove factors whose IC has decayed below the threshold."""
        before = len(self.factors)
        self.factors = [f for f in self.factors if abs(f.ic) >= self.ic_threshold]
        removed = before - len(self.factors)
        if removed:
            logger.info(f"Pruned {removed} factors from pool")

    def get_weights(self) -> list[float]:
        """Get normalized weights for all factors in the pool."""
        if not self.factors:
            return []
        total_ic = sum(abs(f.ic) for f in self.factors)
        if total_ic == 0:
            return [1.0 / len(self.factors)] * len(self.factors)
        return [abs(f.ic) / total_ic for f in self.factors]

    def save(self, path: str):
        """Save the factor pool to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = [f.to_dict() for f in self.factors]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Factor pool saved to {path} ({len(self.factors)} factors)")

    def load(self, path: str):
        """Load a factor pool from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        self.factors = [AlphaFactor.from_dict(d) for d in data]
        logger.info(f"Factor pool loaded from {path} ({len(self.factors)} factors)")

    def summary(self) -> str:
        """Return a summary string of the current pool."""
        if not self.factors:
            return "Empty pool"
        ics = [f.ic for f in self.factors]
        return (
            f"Pool: {len(self.factors)}/{self.pool_size} factors | "
            f"IC: mean={np.mean(ics):.4f}, max={max(ics):.4f}, min={min(ics):.4f}"
        )
