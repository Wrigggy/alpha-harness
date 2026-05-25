"""AST tools for AlphaAgent-style structural novelty regularization.

Ports the three core functions from
  https://github.com/RndmVariableQ/AlphaAgent
  alphaagent/components/coder/factor_coder/factor_ast.py

to the alphagen Expression hierarchy used in this repo:

  Feature      → "unique variable"
  Constant     → "free arg"
  DeltaTime    → "free arg" (window param, treated as a numeric leaf)
  UnaryOp / BinaryOp / RollingOp / PairRollingOp → internal nodes

The acceptance rule (from FactorRegulator.is_expression_acceptable) is:

  max_common_subtree_size_vs_zoo ≤ duplication_threshold (default 8)
  free_args / total_nodes        < 0.5
  unique_vars / total_nodes      < 0.5
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "external" / "alphagen"))

from alphagen.data.expression import (  # noqa: E402
    BinaryOperator,
    Constant,
    DeltaTime,
    Expression,
    Feature,
    Operator,
    PairRollingOperator,
    RollingOperator,
    UnaryOperator,
)


# ---------------------------------------------------------------------------
# Tree walking helpers
# ---------------------------------------------------------------------------

def _children(node: Expression) -> tuple[Expression, ...]:
    if isinstance(node, Operator):
        return tuple(node.operands)
    return ()


def count_all_nodes(expr: Expression) -> int:
    return 1 + sum(count_all_nodes(c) for c in _children(expr))


def count_free_args(expr: Expression) -> int:
    """Constants AND DeltaTimes count as free numeric leaves."""
    if isinstance(expr, (Constant, DeltaTime)):
        return 1
    return sum(count_free_args(c) for c in _children(expr))


def count_unique_vars(expr: Expression) -> int:
    seen: set[int] = set()

    def _walk(n: Expression) -> None:
        if isinstance(n, Feature):
            # FeatureType is an IntEnum (OPEN, CLOSE, ...) — use its int value
            seen.add(int(n._feature))
            return
        for c in _children(n):
            _walk(c)

    _walk(expr)
    return len(seen)


# ---------------------------------------------------------------------------
# Subtree equality and largest-common-subtree search
# ---------------------------------------------------------------------------

# Commutative binary operators where (a op b) ≡ (b op a) for the purpose
# of matching textbook alphas. We keep the set tight — Add/Mul/Min/Max/etc.
_COMMUTATIVE_OPS: frozenset[str] = frozenset({"Add", "Mul", "Max", "Min", "Greater", "Less"})


def _nodes_equal(a: Expression, b: Expression) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, Feature):
        return int(a._feature) == int(b._feature)
    if isinstance(a, Constant):
        return a._value == b._value
    if isinstance(a, DeltaTime):
        return a._delta_time == b._delta_time
    # Operators: same class is enough for the node-level check
    return True


def _subtrees_equal(a: Expression, b: Expression) -> bool:
    if not _nodes_equal(a, b):
        return False
    if isinstance(a, (Feature, Constant, DeltaTime)):
        return True
    ca, cb = _children(a), _children(b)
    if len(ca) != len(cb):
        return False
    if isinstance(a, BinaryOperator) and type(a).__name__ in _COMMUTATIVE_OPS:
        l1, r1 = ca
        l2, r2 = cb
        return (_subtrees_equal(l1, l2) and _subtrees_equal(r1, r2)) or \
               (_subtrees_equal(l1, r2) and _subtrees_equal(r1, l2))
    return all(_subtrees_equal(x, y) for x, y in zip(ca, cb))


def _iter_subtrees(node: Expression) -> Iterable[Expression]:
    yield node
    for c in _children(node):
        yield from _iter_subtrees(c)


def largest_common_subtree_size(a: Expression, b: Expression) -> int:
    """Return the size (node count) of the largest matching subtree.

    Quadratic in subtree count — fine for our sub-100-node expressions.
    """
    best = 0
    subs_b = list(_iter_subtrees(b))
    # Precompute sizes once
    sizes_b = [count_all_nodes(s) for s in subs_b]
    for sa in _iter_subtrees(a):
        sa_size = count_all_nodes(sa)
        if sa_size <= best:
            continue
        for sb, sb_size in zip(subs_b, sizes_b):
            if sb_size != sa_size or sb_size <= best:
                continue
            if _subtrees_equal(sa, sb):
                best = sa_size
                break
    return best


def match_zoo(candidate: Expression, zoo: list[Expression]) -> tuple[int, Expression | None]:
    """Return (max_subtree_size, the_matched_zoo_entry_or_None)."""
    best_size = 0
    best_zoo: Expression | None = None
    for entry in zoo:
        try:
            size = largest_common_subtree_size(candidate, entry)
        except Exception:
            continue
        if size > best_size:
            best_size = size
            best_zoo = entry
    return best_size, best_zoo


# ---------------------------------------------------------------------------
# Acceptance gate (port of FactorRegulator.is_expression_acceptable)
# ---------------------------------------------------------------------------


def evaluate_originality(
    candidate: Expression,
    zoo: list[Expression],
    duplication_threshold: int = 8,
    leaf_ratio_threshold: float = 0.5,
) -> dict:
    n_all = count_all_nodes(candidate)
    n_free = count_free_args(candidate)
    n_vars = count_unique_vars(candidate)
    dup_size, matched = match_zoo(candidate, zoo)
    free_ratio = n_free / n_all if n_all else 1.0
    vars_ratio = n_vars / n_all if n_all else 1.0
    accepted = (
        dup_size <= duplication_threshold
        and free_ratio < leaf_ratio_threshold
        and vars_ratio < leaf_ratio_threshold
    )
    reject_reasons: list[str] = []
    if dup_size > duplication_threshold:
        reject_reasons.append(
            f"max_common_subtree_size={dup_size} > threshold={duplication_threshold}"
        )
    if free_ratio >= leaf_ratio_threshold:
        reject_reasons.append(f"free_args_ratio={free_ratio:.2f} >= {leaf_ratio_threshold}")
    if vars_ratio >= leaf_ratio_threshold:
        reject_reasons.append(f"unique_vars_ratio={vars_ratio:.2f} >= {leaf_ratio_threshold}")
    return {
        "n_all_nodes": n_all,
        "n_free_args": n_free,
        "n_unique_vars": n_vars,
        "duplicated_subtree_size": dup_size,
        "matched_zoo_expr": str(matched) if matched is not None else None,
        "free_args_ratio": free_ratio,
        "unique_vars_ratio": vars_ratio,
        "accepted": accepted,
        "reject_reasons": reject_reasons,
    }
