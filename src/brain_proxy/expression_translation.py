"""Translate local AlphaGen expressions into WorldQuant BRAIN-style syntax."""

from __future__ import annotations

from dataclasses import dataclass

from alphagen.data.expression import (
    Abs,
    Add,
    Constant,
    Corr,
    Cov,
    CSRank,
    Delta,
    DeltaTime,
    Div,
    EMA,
    Expression,
    Feature,
    Greater,
    Less,
    Log,
    Mad,
    Max,
    Mean,
    Med,
    Min,
    Mul,
    Pow,
    Rank,
    Ref,
    Sign,
    Skew,
    Std,
    Sub,
    Sum,
    Var,
    WMA,
)
from alphagen.data.parser import parse_expression


FEATURE_NAME_MAP = {
    "amount": "amount",
    "close": "close",
    "high": "high",
    "low": "low",
    "open": "open",
    "volume": "volume",
    "vwap": "vwap",
    "turnover": "turnover",
    "market_cap": "market_cap",
    "industry": "industry",
}


@dataclass
class TranslationResult:
    local_expression: str
    worldquant_expression: str
    supported: bool
    notes: list[str]


@dataclass
class BrainVariant:
    label: str
    expression: str
    notes: list[str]


def _is_price_feature(expr: Expression) -> bool:
    return isinstance(expr, Feature) and expr._feature.name.lower() in {"open", "high", "low", "close", "vwap"}


def _needs_unit_safe_rewrite(expr: Expression) -> bool:
    return isinstance(expr, Delta) and isinstance(expr.operands[0], Sub) and isinstance(expr.operands[0].operands[0], Constant)


def rewrite_local_expression_for_brain(expr: Expression) -> tuple[Expression, list[str]]:
    """Rewrite a local AlphaGen expression into a safer local AST before translation."""
    notes: list[str] = []

    if isinstance(expr, Delta) and isinstance(expr.operands[0], Sub):
        sub_expr = expr.operands[0]
        left = sub_expr.operands[0]
        right = sub_expr.operands[1]
        if isinstance(left, Constant) and float(left.value) == -1.0 and _is_price_feature(right):
            notes.append("Rewrote price-constant delta to reverse(delta(price)) to avoid unit mismatch.")
            return Mul(Constant(-1.0), Delta(right, expr.operands[1])), notes

    if isinstance(expr, Sub) and isinstance(expr.operands[0], Constant) and _is_price_feature(expr.operands[1]):
        notes.append("Rewrote price-constant subtraction into price-relative form.")
        return Div(expr.operands[0], expr.operands[1]), notes

    if hasattr(expr, "operands"):
        operands = list(expr.operands)
        rewritten_ops: list[Expression] = []
        for op in operands:
            new_op, child_notes = rewrite_local_expression_for_brain(op)
            notes.extend(child_notes)
            rewritten_ops.append(new_op)
        if type(expr).__name__ == "Abs":
            return Abs(rewritten_ops[0]), notes
        if type(expr).__name__ == "Sign":
            return Sign(rewritten_ops[0]), notes
        if type(expr).__name__ == "Log":
            return Log(rewritten_ops[0]), notes
        if type(expr).__name__ == "Add":
            return Add(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Sub":
            return Sub(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Mul":
            return Mul(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Div":
            return Div(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Pow":
            return Pow(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Greater":
            return Greater(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Less":
            return Less(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Ref":
            return Ref(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Mean":
            return Mean(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Sum":
            return Sum(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Std":
            return Std(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Var":
            return Var(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Skew":
            return Skew(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Max":
            return Max(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Min":
            return Min(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Rank":
            return Rank(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Delta":
            return Delta(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "WMA":
            return WMA(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "EMA":
            return EMA(rewritten_ops[0], rewritten_ops[1]), notes
        if type(expr).__name__ == "Cov":
            return Cov(rewritten_ops[0], rewritten_ops[1], rewritten_ops[2]), notes
        if type(expr).__name__ == "Corr":
            return Corr(rewritten_ops[0], rewritten_ops[1], rewritten_ops[2]), notes

    return expr, notes


class BrainExpressionTranslator:
    """AST-level translator from local AlphaGen syntax to BRAIN-style syntax."""

    def __init__(self, feature_name_map: dict[str, str] | None = None) -> None:
        self.feature_name_map = dict(FEATURE_NAME_MAP)
        if feature_name_map:
            self.feature_name_map.update(feature_name_map)

    def translate(self, expression: str | Expression) -> TranslationResult:
        expr = parse_expression(expression) if isinstance(expression, str) else expression
        notes: list[str] = []
        translated = self._translate_node(expr, notes)
        return TranslationResult(
            local_expression=str(expression),
            worldquant_expression=translated,
            supported=not any(note.startswith("UNSUPPORTED") for note in notes),
            notes=notes,
        )

    def _translate_node(self, expr: Expression, notes: list[str]) -> str:
        if isinstance(expr, Feature):
            local_name = str(expr).lstrip("$")
            return self.feature_name_map.get(local_name, local_name)
        if isinstance(expr, Constant):
            value = float(expr.value)
            return str(int(value)) if value.is_integer() else str(value)
        if isinstance(expr, DeltaTime):
            return str(expr).rstrip("d")

        operands = getattr(expr, "operands", ())
        rendered = [self._translate_node(op, notes) for op in operands]

        if isinstance(expr, Abs):
            return f"abs({rendered[0]})"
        if isinstance(expr, Sign):
            return f"sign({rendered[0]})"
        if isinstance(expr, Log):
            return f"log({rendered[0]})"
        if isinstance(expr, CSRank):
            return f"rank({rendered[0]})"
        if isinstance(expr, Add):
            return f"add({rendered[0]},{rendered[1]})"
        if isinstance(expr, Sub):
            return f"subtract({rendered[0]},{rendered[1]})"
        if isinstance(expr, Mul):
            return f"multiply({rendered[0]},{rendered[1]})"
        if isinstance(expr, Div):
            return f"divide({rendered[0]},{rendered[1]})"
        if isinstance(expr, Pow):
            return f"power({rendered[0]},{rendered[1]})"
        if isinstance(expr, Greater):
            return f"max({rendered[0]},{rendered[1]})"
        if isinstance(expr, Less):
            return f"min({rendered[0]},{rendered[1]})"
        if isinstance(expr, Ref):
            return f"ts_delay({rendered[0]},{rendered[1]})"
        if isinstance(expr, Mean):
            return f"ts_mean({rendered[0]},{rendered[1]})"
        if isinstance(expr, Sum):
            return f"ts_sum({rendered[0]},{rendered[1]})"
        if isinstance(expr, Std):
            return f"ts_std_dev({rendered[0]},{rendered[1]})"
        if isinstance(expr, Var):
            return f"ts_var({rendered[0]},{rendered[1]})"
        if isinstance(expr, Skew):
            return f"ts_skewness({rendered[0]},{rendered[1]})"
        if isinstance(expr, Max):
            return f"ts_max({rendered[0]},{rendered[1]})"
        if isinstance(expr, Min):
            return f"ts_min({rendered[0]},{rendered[1]})"
        if isinstance(expr, Med):
            notes.append("UNSUPPORTED: Med mapped to ts_median; verify operator availability on your BRAIN account.")
            return f"ts_median({rendered[0]},{rendered[1]})"
        if isinstance(expr, Mad):
            notes.append("UNSUPPORTED: Mad mapped to ts_mean(abs(subtract(x, ts_mean(x,d))), d) proxy.")
            x, d = rendered
            return f"ts_mean(abs(subtract({x},ts_mean({x},{d}))),{d})"
        if isinstance(expr, Rank):
            return f"ts_rank({rendered[0]},{rendered[1]})"
        if type(expr).__name__ == "GroupNeutralize":
            return f"group_neutralize({rendered[0]},{rendered[1]})"
        if type(expr).__name__ == "GroupRank":
            return f"group_rank({rendered[0]},{rendered[1]})"
        if isinstance(expr, Delta):
            operand = expr.operands[0]
            if (
                isinstance(operand, Sub)
                and isinstance(operand.operands[0], Constant)
                and float(operand.operands[0].value) == -1.0
                and isinstance(operand.operands[1], Feature)
                and operand.operands[1]._feature.name.lower() in {"open", "high", "low", "close", "vwap"}
            ):
                field = self.feature_name_map.get(operand.operands[1]._feature.name.lower(), operand.operands[1]._feature.name.lower())
                notes.append("UNSUPPORTED: rewritten price-constant delta to reverse(ts_delta(price,d)) to avoid unit mismatch.")
                return f"reverse(ts_delta({field},{rendered[1]}))"
            return f"ts_delta({rendered[0]},{rendered[1]})"
        if isinstance(expr, WMA):
            notes.append("WMA mapped to ts_decay_linear for WorldQuant-style submission.")
            return f"ts_decay_linear({rendered[0]},{rendered[1]})"
        if isinstance(expr, EMA):
            notes.append("EMA mapped to ts_mean proxy; replace with platform-native smoothing if needed.")
            return f"ts_mean({rendered[0]},{rendered[1]})"
        if isinstance(expr, Cov):
            return f"ts_covariance({rendered[0]},{rendered[1]},{rendered[2]})"
        if isinstance(expr, Corr):
            return f"ts_corr({rendered[0]},{rendered[1]},{rendered[2]})"

        notes.append(f"UNSUPPORTED: No WorldQuant mapping registered for {type(expr).__name__}.")
        return str(expr)


def translate_expression_to_brain(expression: str) -> TranslationResult:
    """Translate one local expression string to BRAIN-style syntax."""
    translator = BrainExpressionTranslator()
    expr = parse_expression(expression)
    rewritten, notes = rewrite_local_expression_for_brain(expr)
    result = translator.translate(rewritten)
    result.notes = notes + result.notes
    result.supported = not any(note.startswith("UNSUPPORTED") for note in result.notes)
    return result


def build_brain_variants(base_expression: str) -> list[BrainVariant]:
    """Generate a few practical BRAIN-style refinement variants."""
    return [
        BrainVariant(
            label="raw",
            expression=base_expression,
            notes=["Direct translation from local AlphaGen syntax."],
        ),
        BrainVariant(
            label="ranked",
            expression=f"rank({base_expression})",
            notes=["Cross-sectional rank for scale control and outlier robustness."],
        ),
        BrainVariant(
            label="ranked_winsorized",
            expression=f"rank(winsorize({base_expression}, std=4))",
            notes=["Add winsorize before rank to reduce extreme values."],
        ),
        BrainVariant(
            label="ranked_group_neutralized",
            expression=f"group_neutralize(rank({base_expression}), subindustry)",
            notes=["Neutralize cross-sectional signal within subindustry groups."],
        ),
        BrainVariant(
            label="ranked_group_neutralized_decay",
            expression=f"ts_decay_linear(group_neutralize(rank({base_expression}), subindustry), 10)",
            notes=["Add linear decay to reduce turnover after group neutralization."],
        ),
    ]


def render_brain_submission_template(
    alpha_name: str,
    base_expression: str,
    local_expression: str,
    description: str | None = None,
) -> str:
    """Render a WorldQuant BRAIN editor template block."""
    note = description or "Auto-generated candidate translated from local AlphaGen expression."
    lines = [
        "/*",
        f"NAME: {alpha_name}",
        "",
        "HYPOTHESIS:",
        note,
        "",
        "LOCAL_EXPRESSION:",
        local_expression,
        "",
        "IMPLEMENTATION:",
        "Base BRAIN-style expression translated from the local research expression.",
        "*/",
        "",
        base_expression,
        "",
    ]
    return "\n".join(lines)
