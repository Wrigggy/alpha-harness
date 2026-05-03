"""Rewrite a local alpha pool into stricter WorldQuant submission candidates."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ALPHAGEN = ROOT / "external" / "alphagen"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXTERNAL_ALPHAGEN) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ALPHAGEN))

from alphagen.data.expression import Abs, Add, Constant, Delta, Div, EMA, Expression, Feature, Log, Max, Mean, Med, Min, Mul, Pow, Rank, Ref, Sign, Skew, Std, Sub, Sum, Var, WMA, Corr, Cov
from alphagen.data.parser import parse_expression

from src.brain_proxy.evaluator import (
    build_proxy_context,
    evaluate_brain_candidate,
    prepare_brain_signal,
    tensor_to_factor_frame,
)
from src.brain_proxy.expression_translation import (
    BrainExpressionTranslator,
    build_brain_variants,
    render_brain_submission_template,
)
from src.team_governance.registry import build_alpha_id
from src.utils.pool_io import load_pool


HARD_GATE_DEFAULTS = {
    "min_sharpe": 1.25,
    "min_fitness": 1.0,
    "max_weight_concentration": 0.10,
    "min_sub_universe_sharpe": 0.20,
}


def _is_price_feature(expr: Expression) -> bool:
    return isinstance(expr, Feature) and expr._feature.name.lower() in {"open", "high", "low", "close", "vwap"}


def _clone_with_operands(expr: Expression, operands: list[Expression]) -> Expression:
    kind = type(expr)
    if kind is Abs:
        return Abs(operands[0])
    if kind is Sign:
        return Sign(operands[0])
    if kind is Log:
        return Log(operands[0])
    if kind is Add:
        return Add(operands[0], operands[1])
    if kind is Sub:
        return Sub(operands[0], operands[1])
    if kind is Mul:
        return Mul(operands[0], operands[1])
    if kind is Div:
        return Div(operands[0], operands[1])
    if kind is Pow:
        return Pow(operands[0], operands[1])
    if kind is Ref:
        return Ref(operands[0], operands[1])
    if kind is Mean:
        return Mean(operands[0], operands[1])
    if kind is Sum:
        return Sum(operands[0], operands[1])
    if kind is Std:
        return Std(operands[0], operands[1])
    if kind is Var:
        return Var(operands[0], operands[1])
    if kind is Skew:
        return Skew(operands[0], operands[1])
    if kind is Max:
        return Max(operands[0], operands[1])
    if kind is Min:
        return Min(operands[0], operands[1])
    if kind is Med:
        return Med(operands[0], operands[1])
    if kind is Rank:
        return Rank(operands[0], operands[1])
    if kind is Delta:
        return Delta(operands[0], operands[1])
    if kind is WMA:
        return WMA(operands[0], operands[1])
    if kind is EMA:
        return EMA(operands[0], operands[1])
    if kind is Cov:
        return Cov(operands[0], operands[1], operands[2])
    if kind is Corr:
        return Corr(operands[0], operands[1], operands[2])
    return expr


def _rewrite_local_for_brain(expr: Expression) -> tuple[Expression, list[str]]:
    notes: list[str] = []

    if isinstance(expr, Delta) and isinstance(expr.operands[0], Sub):
        sub_expr = expr.operands[0]
        left, right = sub_expr.operands
        if isinstance(left, Constant) and float(left.value) == -1.0 and _is_price_feature(right):
            notes.append("rewrote unit-mismatch delta on price field")
            return Mul(Constant(-1.0), Delta(right, expr.operands[1])), notes

    if isinstance(expr, Sub) and isinstance(expr.operands[0], Constant) and _is_price_feature(expr.operands[1]):
        notes.append("rewrote price-level subtraction into inverse form")
        return Div(expr.operands[0], expr.operands[1]), notes

    operands = list(getattr(expr, "operands", ()))
    if operands:
        rewritten_ops: list[Expression] = []
        for op in operands:
            new_op, child_notes = _rewrite_local_for_brain(op)
            notes.extend(child_notes)
            rewritten_ops.append(new_op)
        expr = _clone_with_operands(expr, rewritten_ops)
    return expr, notes


def _contains_raw_price_leaf(expr: Expression) -> bool:
    if isinstance(expr, Feature):
        return _is_price_feature(expr)
    operands = getattr(expr, "operands", ())
    return any(_contains_raw_price_leaf(op) for op in operands)


def _is_price_level_structure(expr: Expression) -> bool:
    if isinstance(expr, Feature):
        return _is_price_feature(expr)

    if isinstance(expr, (Rank, Delta, WMA, EMA, Mean, Std, Var, Skew, Max, Min, Med, Ref, Corr, Cov)):
        return False

    if isinstance(expr, (Add, Sub, Mul, Div, Pow)):
        left, right = expr.operands
        left_is_price = isinstance(left, Feature) and _is_price_feature(left)
        right_is_price = isinstance(right, Feature) and _is_price_feature(right)
        left_is_constant = isinstance(left, Constant)
        right_is_constant = isinstance(right, Constant)
        if left_is_price or right_is_price:
            if left_is_constant or right_is_constant:
                return True
            if left_is_price and right_is_price and not isinstance(expr, Div):
                return True
        return False

    return False


def _generate_candidate_expressions(worldquant_expr: str) -> list[tuple[str, str]]:
    return [
        ("base", worldquant_expr),
        ("rank", f"rank({worldquant_expr})"),
        ("rank_decay_5", f"ts_decay_linear(rank({worldquant_expr}), 5)"),
        ("rank_decay", f"ts_decay_linear(rank({worldquant_expr}), 10)"),
        ("rank_decay_15", f"ts_decay_linear(rank({worldquant_expr}), 15)"),
        ("group_neutralize", f"group_neutralize(rank({worldquant_expr}), subindustry)"),
        ("group_neutralize_decay_5", f"ts_decay_linear(group_neutralize(rank({worldquant_expr}), subindustry), 5)"),
        ("group_neutralize_decay", f"ts_decay_linear(group_neutralize(rank({worldquant_expr}), subindustry), 10)"),
        ("group_neutralize_decay_15", f"ts_decay_linear(group_neutralize(rank({worldquant_expr}), subindustry), 15)"),
        ("winsorize_group_decay", f"ts_decay_linear(group_neutralize(rank(winsorize({worldquant_expr}, std=4)), subindustry), 10)"),
        ("winsorize_group_decay_15", f"ts_decay_linear(group_neutralize(rank(winsorize({worldquant_expr}, std=4)), subindustry), 15)"),
    ]


def _rowwise_rank(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rank(axis=1, method="average", pct=True)


def _rowwise_winsorize(frame: pd.DataFrame, quantile: float = 0.01) -> pd.DataFrame:
    out = frame.copy().astype(float)
    for idx in out.index:
        row = out.loc[idx]
        finite = row.dropna()
        if finite.empty:
            continue
        lower = finite.quantile(quantile)
        upper = finite.quantile(1.0 - quantile)
        out.loc[idx] = row.clip(lower=lower, upper=upper)
    return out


def _group_neutralize(frame: pd.DataFrame, group_frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().astype(float)
    common_idx = out.index.intersection(group_frame.index)
    common_cols = out.columns.intersection(group_frame.columns)
    for idx in common_idx:
        row = out.loc[idx, common_cols].astype(float)
        groups = group_frame.loc[idx, common_cols]
        df = pd.DataFrame({"signal": row, "group": groups})
        mask = df["signal"].notna() & df["group"].notna()
        if not mask.any():
            continue
        means = df.loc[mask].groupby("group")["signal"].transform("mean")
        df.loc[mask, "signal"] = df.loc[mask, "signal"] - means
        out.loc[idx, common_cols] = df["signal"]
    return out


def _ts_decay_linear(frame: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    return frame.rolling(window=window, min_periods=max(2, window // 3)).mean()


def _numeric_variant_signal(base_signal: pd.DataFrame, panel: dict[str, pd.DataFrame], variant: str) -> pd.DataFrame:
    board = panel.get("board")
    if variant == "base":
        return base_signal
    if variant == "rank":
        return _rowwise_rank(base_signal)
    if variant == "rank_decay_5":
        return _ts_decay_linear(_rowwise_rank(base_signal), 5)
    if variant == "rank_decay":
        return _ts_decay_linear(_rowwise_rank(base_signal), 10)
    if variant == "rank_decay_15":
        return _ts_decay_linear(_rowwise_rank(base_signal), 15)
    if variant == "group_neutralize":
        if board is None:
            return _rowwise_rank(base_signal)
        return _group_neutralize(_rowwise_rank(base_signal), board)
    if variant == "group_neutralize_decay_5":
        if board is None:
            return _ts_decay_linear(_rowwise_rank(base_signal), 5)
        return _ts_decay_linear(_group_neutralize(_rowwise_rank(base_signal), board), 5)
    if variant == "group_neutralize_decay":
        if board is None:
            return _ts_decay_linear(_rowwise_rank(base_signal), 10)
        return _ts_decay_linear(_group_neutralize(_rowwise_rank(base_signal), board), 10)
    if variant == "group_neutralize_decay_15":
        if board is None:
            return _ts_decay_linear(_rowwise_rank(base_signal), 15)
        return _ts_decay_linear(_group_neutralize(_rowwise_rank(base_signal), board), 15)
    if variant == "winsorize_group_decay":
        wins = _rowwise_winsorize(base_signal, 0.01)
        if board is None:
            return _ts_decay_linear(_rowwise_rank(wins), 10)
        return _ts_decay_linear(_group_neutralize(_rowwise_rank(wins), board), 10)
    if variant == "winsorize_group_decay_15":
        wins = _rowwise_winsorize(base_signal, 0.01)
        if board is None:
            return _ts_decay_linear(_rowwise_rank(wins), 15)
        return _ts_decay_linear(_group_neutralize(_rowwise_rank(wins), board), 15)
    return base_signal


def _metric_filter(row: dict, gates: dict[str, float]) -> bool:
    return bool(
        row.get("coverage", 0.0) >= 0.60
        and row.get("avg_turnover", 1.0) <= 0.70
        and row.get("sharpe_ratio", 0.0) >= gates["min_sharpe"]
        and row.get("fitness", 0.0) >= gates["min_fitness"]
        and row.get("weight_concentration", 1.0) <= gates["max_weight_concentration"]
        and row.get("sub_universe_sharpe", 0.0) >= gates["min_sub_universe_sharpe"]
    )


def _clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for file in path.glob("*"):
        if file.is_file():
            file.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite a local pool into strict WorldQuant candidates")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--data-config", default="config/data_config_equity_brain_top50.yaml")
    parser.add_argument("--brain-config", default="config/brain_proxy_equity_cn.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--min-sharpe", type=float, default=HARD_GATE_DEFAULTS["min_sharpe"])
    parser.add_argument("--min-fitness", type=float, default=HARD_GATE_DEFAULTS["min_fitness"])
    parser.add_argument("--max-weight-concentration", type=float, default=HARD_GATE_DEFAULTS["max_weight_concentration"])
    parser.add_argument("--min-sub-universe-sharpe", type=float, default=HARD_GATE_DEFAULTS["min_sub_universe_sharpe"])
    args = parser.parse_args()

    expressions, _ = load_pool(args.pool)
    context = build_proxy_context(
        data_config_path=args.data_config,
        split=args.split,
        brain_config_path=args.brain_config,
    )
    translator = BrainExpressionTranslator()
    gates = {
        "min_sharpe": args.min_sharpe,
        "min_fitness": args.min_fitness,
        "max_weight_concentration": args.max_weight_concentration,
        "min_sub_universe_sharpe": args.min_sub_universe_sharpe,
    }

    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "out" / "brain_rewrite_pool" / f"{Path(args.pool).stem}_{args.split}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    submit_dir = out_dir / "submit_ready_worldquant"
    _clean_dir(submit_dir)

    rows: list[dict] = []
    accepted_rows: list[dict] = []
    variant_rows: list[dict] = []

    for idx, expression in enumerate(expressions, start=1):
        parsed = parse_expression(expression)
        rewritten_local, rewrite_notes = _rewrite_local_for_brain(parsed)
        if _is_price_level_structure(rewritten_local):
            rewrite_notes.append("dropped price-level structure")
            continue

        translated = translator.translate(rewritten_local)
        if not translated.supported:
            continue

        factor_df = tensor_to_factor_frame(
            context["stock_data"],
            context["calculator"].evaluate_alpha(rewritten_local),
            name=f"factor_{idx:02d}",
        )
        base_signal = prepare_brain_signal(
            factor_df=factor_df,
            panel=context["panel"],
            cfg=context["brain_cfg"],
        )

        for variant_name, candidate_expr in _generate_candidate_expressions(translated.worldquant_expression):
            candidate_factor = _numeric_variant_signal(base_signal, context["panel"], variant_name)
            _, summary = evaluate_brain_candidate(
                expression=candidate_expr,
                factor_df=candidate_factor,
                panel=context["panel"],
                forward_returns=context["forward_returns"],
                next_bar_returns=context["next_bar_returns"],
                liquidity_df=context["liquidity_df"],
                bars_per_year=context["bars_per_year"],
                cfg=context["brain_cfg"],
                preprocessed_signal=True,
            )
            summary["source_expression"] = expression
            summary["candidate_variant"] = variant_name
            summary["alpha_id"] = build_alpha_id(candidate_expr)
            summary["worldquant_expression"] = translated.worldquant_expression
            summary["translation_supported"] = translated.supported
            summary["translation_notes"] = " | ".join([*rewrite_notes, *translated.notes])
            rows.append(summary)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["passes_proxy_gates", "brain_readiness_score", "sharpe_ratio", "fitness"],
            ascending=[False, False, False, False],
        )

    accepted = result[result.apply(lambda row: _metric_filter(row.to_dict(), gates), axis=1)].copy() if not result.empty else result
    accepted_rows = accepted.to_dict(orient="records")

    result.to_csv(out_dir / "brain_rewrite_candidates.csv", index=False)
    accepted.to_csv(out_dir / "accepted_for_submission.csv", index=False)
    with (out_dir / "brain_rewrite_candidates.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    for row in accepted_rows:
        template = render_brain_submission_template(
            alpha_name=row["alpha_id"],
            base_expression=row["worldquant_expression"],
            local_expression=row["source_expression"],
            description=f"Rewritten variant: {row['candidate_variant']}",
        )
        (submit_dir / f"{row['alpha_id']}.txt").write_text(template, encoding="utf-8")
        for variant in build_brain_variants(row["worldquant_expression"]):
            variant_rows.append(
                {
                    "alpha_id": row["alpha_id"],
                    "variant": variant.label,
                    "expression": variant.expression,
                    "notes": " | ".join(variant.notes),
                }
            )

    pd.DataFrame(variant_rows).to_csv(out_dir / "brain_expression_variants.csv", index=False)
    logger.info("Brain rewrite pool written to {}", out_dir)


if __name__ == "__main__":
    main()
