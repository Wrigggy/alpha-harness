"""Utilities for loading and normalizing factor pools."""

from __future__ import annotations

import json
from pathlib import Path


def invert_expression(expression: str) -> str:
    """Wrap an expression with a sign flip."""
    return f"Mul(-1.0,{expression})"


def orient_expression(expression: str, rank_ic: float) -> tuple[str, float]:
    """Flip a factor so downstream use prefers positive Rank IC orientation."""
    if rank_ic < 0:
        return invert_expression(expression), -1.0
    return expression, 1.0


def score_to_weight(rank_ic: float, rank_icir: float) -> float:
    """Convert validation quality into a stable positive combination weight."""
    return max(abs(rank_ic), 0.0) + 0.5 * max(abs(rank_icir), 0.0)


def load_pool(pool_path: str | Path) -> tuple[list[str], list[float]]:
    """Load a factor pool from a supported JSON shape."""
    path = Path(pool_path)
    with path.open(encoding="utf-8") as f:
        pool_data = json.load(f)

    if isinstance(pool_data, list):
        expressions = [str(item.get("expression", "")) for item in pool_data]
        weights = [float(item.get("weight", item.get("ic", 1.0))) for item in pool_data]
        return expressions, weights

    if isinstance(pool_data, dict):
        if "exprs" in pool_data:
            expressions = [str(expr) for expr in pool_data["exprs"]]
            weights = [float(w) for w in pool_data.get("weights", [1.0] * len(expressions))]
            return expressions, weights
        if "expressions" in pool_data:
            expressions = [str(expr) for expr in pool_data["expressions"]]
            weights = [float(w) for w in pool_data.get("weights", [1.0] * len(expressions))]
            return expressions, weights

    raise ValueError(f"Unrecognized pool format: {path}")


def normalize_weights(weights: list[float]) -> list[float]:
    """Normalize weights by absolute exposure."""
    if not weights:
        return []
    total = sum(abs(w) for w in weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]
