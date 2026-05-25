"""ExpressionParser ported into our src tree so AlphaQCM's fork can use it.

The QCM fork of alphagen at external/alphaqcm/alphagen/ ships expression.py but
not parser.py or utils/misc.py. We need a parser that constructs QCM-fork
Expression objects (so they're compatible with the QCM AlphaPool). Touching
external/alphaqcm/ is off-limits because external/ is gitignored — the fix
wouldn't survive a re-clone.

Logic is a direct port of external/alphagen/alphagen/data/parser.py with
find_last_if inlined. Must be imported AFTER run_alphaqcm.py has prioritized
QCM's alphagen on sys.path; the relative imports below then resolve to QCM's
expression module.
"""

from __future__ import annotations

import re
from typing import Type, List, Dict, Set, Union, Optional, cast

# These resolve to the QCM-fork modules when imported from run_alphaqcm.py
# (which has already prepended external/alphaqcm to sys.path).
from alphagen.data.expression import (  # type: ignore
    Expression, Operator, Feature, FeatureType, Constant, DeltaTime,
    RollingOperator, PairRollingOperator,
)


_PATTERN = re.compile(r'([+-]?[\d.]+|\W|\w+)')
_NUMERIC = re.compile(r'[+-]?[\d.]+')
_StackItem = Union[List[Type[Operator]], Expression]
_OpMap = Dict[str, List[Type[Operator]]]
_DTLike = Union[float, Constant, DeltaTime]


class ExpressionParsingError(Exception):
    pass


def _find_last_if(lst, predicate) -> int:
    for i in range(len(lst) - 1, -1, -1):
        if predicate(lst[i]):
            return i
    return -1


class ExpressionParser:
    def __init__(
        self,
        operators: List[Type[Operator]],
        ignore_case: bool = False,
        time_deltas_need_suffix: bool = False,
        non_positive_time_deltas_allowed: bool = True,
        feature_need_dollar_sign: bool = False,
        additional_operator_mapping: Optional[_OpMap] = None,
    ):
        self._ignore_case = ignore_case
        self._allow_np_dt = non_positive_time_deltas_allowed
        self._suffix_needed = time_deltas_need_suffix
        self._dollar_needed = feature_need_dollar_sign
        self._features = {f.name.lower(): f for f in FeatureType}
        self._operators: _OpMap = {op.__name__: [op] for op in operators}
        if additional_operator_mapping is not None:
            self._merge_op_mapping(additional_operator_mapping)
        if ignore_case:
            self._operators = {k.lower(): v for k, v in self._operators.items()}
        self._stack: List[_StackItem] = []
        self._tokens: List[str] = []

    def parse(self, expr: str) -> Expression:
        self._stack = []
        self._tokens = [t for t in _PATTERN.findall(expr) if not t.isspace()]
        self._tokens.reverse()
        while self._tokens:
            self._stack.append(self._get_next_item())
            self._process_punctuation()
        if len(self._stack) != 1:
            raise ExpressionParsingError("Multiple items remain in the stack")
        if isinstance(self._stack[0], Expression):
            return self._stack[0]
        raise ExpressionParsingError(f"{self._stack[0]} is not a valid expression")

    def _merge_op_mapping(self, mapping: _OpMap) -> None:
        for name, ops in mapping.items():
            if (old_ops := self._operators.get(name)) is not None:
                self._operators[name] = list(dict.fromkeys(old_ops + ops))
            else:
                self._operators[name] = ops

    def _get_next_item(self) -> _StackItem:
        top = self._pop_token()
        if top == '$':
            top = self._pop_token()
            if (feature := self._features.get(top)) is None:
                raise ExpressionParsingError(f"Can't find the feature {top}")
            return Feature(feature)
        elif self._tokens_eq(top, "Constant"):
            if self._pop_token() != '(':
                raise ExpressionParsingError("\"Constant\" should be followed by a left parenthesis")
            value = self._to_float(self._pop_token())
            if self._pop_token() != ')':
                raise ExpressionParsingError("\"Constant\" should be closed by a right parenthesis")
            return Constant(value)
        elif _NUMERIC.fullmatch(top) is not None:
            value = self._to_float(top)
            if self._peek_token() == 'd':
                self._pop_token()
                return self._as_delta_time(value)
            return Constant(value)
        else:
            if not self._dollar_needed and (feature := self._features.get(top)) is not None:
                return Feature(feature)
            if (ops := self._operators.get(top)) is not None:
                return ops
            raise ExpressionParsingError(f"Cannot find the operator/feature name {top}")

    def _process_punctuation(self) -> None:
        if not self._tokens:
            return
        top = self._pop_token()
        stack_top_is_ops = self._stack and not isinstance(self._stack[-1], Expression)
        if (top == '(') != stack_top_is_ops:
            raise ExpressionParsingError("A left parenthesis should follow an operator name")
        if top in ('(', ','):
            return
        if top == ')':
            self._build_one_subexpr()
            self._process_punctuation()
            return
        raise ExpressionParsingError(f"Unexpected token {top}")

    def _build_one_subexpr(self) -> None:
        op_idx = _find_last_if(self._stack, lambda item: isinstance(item, list))
        if op_idx == -1:
            raise ExpressionParsingError("Unmatched right parenthesis")
        ops = cast(List[Type[Operator]], self._stack[op_idx])
        operands = self._stack[op_idx + 1:]
        self._stack = self._stack[:op_idx]
        if any(not isinstance(item, Expression) for item in operands):
            raise ExpressionParsingError("An operator name cannot be used as an operand")
        operands = cast(List[Expression], operands)
        dt_operands = operands
        if (not self._suffix_needed and
                isinstance(operands[-1], Constant) and
                (dt := self._as_delta_time(operands[-1], noexcept=True)) is not None):
            dt_operands = operands.copy()
            dt_operands[-1] = dt
        # QCM-fork operators don't expose validate_parameters — fall back to n_args
        # check + direct construction (raise on failure).
        msgs: Set[str] = set()
        for op in ops:
            used = operands
            if issubclass(op, (RollingOperator, PairRollingOperator)):
                used = dt_operands
            try:
                expected = op.n_args()
            except Exception:
                expected = len(used)
            if expected != len(used):
                msgs.add(f"{op.__name__} expects {expected} args, got {len(used)}")
                continue
            try:
                self._stack.append(op(*used))
                return
            except Exception as ex:
                msgs.add(f"{op.__name__}: {ex}")
        raise ExpressionParsingError("; ".join(msgs) or "no operator matched")

    def _tokens_eq(self, lhs: str, rhs: str) -> bool:
        return lhs.lower() == rhs.lower() if self._ignore_case else lhs == rhs

    @classmethod
    def _to_float(cls, token: str) -> float:
        try:
            return float(token)
        except Exception:
            raise ExpressionParsingError(f"{token} can't be converted to float")

    def _pop_token(self) -> str:
        if not self._tokens:
            raise ExpressionParsingError("No more tokens left")
        top = self._tokens.pop()
        return top.lower() if self._ignore_case else top

    def _peek_token(self) -> Optional[str]:
        return self._tokens[-1] if self._tokens else None

    def _as_delta_time(self, value: _DTLike, noexcept: bool = False):
        def maybe_raise(message: str) -> None:
            if not noexcept:
                raise ExpressionParsingError(message)

        if isinstance(value, DeltaTime):
            return value
        if isinstance(value, Constant):
            # QCM fork stores it as `_value`, upstream as `value`.
            const_val = getattr(value, "_value", getattr(value, "value", None))
            if const_val is None:
                maybe_raise(f"Cannot read Constant value from {value!r}")
                return None
            value = const_val
        if not float(value).is_integer():
            maybe_raise(f"A DeltaTime should be integral, but {value} is not")
            return None
        if int(value) <= 0 and not self._allow_np_dt:
            maybe_raise(f"A DeltaTime should refer to a positive time difference, but got {int(value)}d")
            return None
        return DeltaTime(int(value))
