"""Hypothesis-driven idea agent for AlphaGen warm-start.

Generates candidate factor expressions via Claude (claude_agent_sdk), validates
them against AlphaGen's parser, scores single-factor IC on the train calculator,
and returns the top-k seeds for `pool.force_load_exprs(...)`.

CLI usage:
    python -m src.factor_mining.idea_agent \
        --seed 42 \
        --n-generate 50 \
        --top-k 10 \
        --out data/factors/warm_seeds_seed42.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import yaml
from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "external" / "alphagen"))

from alphagen.config import OPERATORS  # noqa: E402
from alphagen.data.expression import Expression, Feature, FeatureType, Ref  # noqa: E402
from alphagen.data.parser import ExpressionParser, ExpressionParsingError  # noqa: E402

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query  # noqa: E402

from src.data_adapter.to_alphagen_format import create_data_splits, CryptoAlphaCalculator  # noqa: E402
from src.utils.device import get_device  # noqa: E402


def _build_parser() -> ExpressionParser:
    return ExpressionParser(
        operators=OPERATORS,
        ignore_case=False,
        time_deltas_need_suffix=True,
        non_positive_time_deltas_allowed=False,
        feature_need_dollar_sign=True,
    )


def _load_prompt(n_seeds: int) -> str:
    p = _ROOT / "prompts" / "idea_agent.txt"
    template = p.read_text(encoding="utf-8")
    return template.replace("{N_SEEDS}", str(n_seeds))


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    """Pull the first top-level JSON array out of a response that may contain
    markdown fences or trailing prose."""
    fence = re.search(r"```(?:json)?\s*\n?(\[.*?\])\s*\n?```", text, re.DOTALL)
    blob = fence.group(1) if fence else text
    start = blob.find("[")
    if start < 0:
        raise ValueError("No JSON array found in response")
    depth = 0
    end = -1
    for i, ch in enumerate(blob[start:], start=start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError("Unbalanced JSON array in response")
    return json.loads(blob[start:end])


async def _call_claude(prompt: str, model: str, seed_hint: int) -> str:
    full_prompt = f"{prompt}\n\n[generation_seed_hint={seed_hint}]"
    result = ""
    async for message in query(
        prompt=full_prompt,
        options=ClaudeAgentOptions(
            model=model,
            allowed_tools=[],
            max_turns=1,
        ),
    ):
        if isinstance(message, ResultMessage):
            result = message.result
    if not result:
        raise RuntimeError("Empty response from Claude")
    return result


def generate_raw_seeds(n_generate: int, model: str, seed: int) -> list[dict[str, Any]]:
    prompt = _load_prompt(n_generate)
    raw = anyio.run(_call_claude, prompt, model, seed)
    items = _extract_json_array(raw)
    logger.info(f"LLM returned {len(items)} raw items")
    return items


def validate_and_parse(
    items: list[dict[str, Any]],
    parser: ExpressionParser,
) -> list[dict[str, Any]]:
    valid = []
    seen_exprs: set[str] = set()
    for item in items:
        expr_str = item.get("expr", "").strip()
        if not expr_str or expr_str in seen_exprs:
            continue
        try:
            parsed = parser.parse(expr_str)
        except (ExpressionParsingError, AssertionError, ValueError, KeyError) as e:
            logger.debug(f"Parse failed for {expr_str!r}: {e}")
            continue
        seen_exprs.add(expr_str)
        valid.append({
            "family": item.get("family", ""),
            "hypothesis": item.get("hypothesis", ""),
            "expr_str": expr_str,
            "expression": parsed,
        })
    logger.info(f"Parsed {len(valid)}/{len(items)} expressions successfully")
    return valid


def score_seeds(
    valid: list[dict[str, Any]],
    calculator: CryptoAlphaCalculator,
    min_abs_ic: float = 0.01,
) -> list[dict[str, Any]]:
    """Compute single-factor IC on the given calculator, attach to each item.

    Items with non-finite IC or |IC| < min_abs_ic are dropped.
    """
    scored = []
    for item in valid:
        try:
            ic = float(calculator.calc_single_IC_ret(item["expression"]))
        except Exception as e:
            logger.debug(f"IC failed for {item['expr_str']!r}: {e}")
            continue
        if not np.isfinite(ic) or abs(ic) < min_abs_ic:
            continue
        item["train_ic"] = ic
        scored.append(item)
    scored.sort(key=lambda x: abs(x["train_ic"]), reverse=True)
    logger.info(f"{len(scored)} seeds passed |IC| ≥ {min_abs_ic}")
    return scored


def run(
    seed: int,
    n_generate: int,
    top_k: int,
    out_path: Path,
    model: str = "claude-opus-4-7",
    data_config_path: str = "config/data_config.yaml",
    min_abs_ic: float = 0.01,
) -> None:
    np.random.seed(seed)

    parser = _build_parser()
    raw_items = generate_raw_seeds(n_generate, model, seed)

    valid = validate_and_parse(raw_items, parser)
    if not valid:
        raise RuntimeError("No valid expressions parsed from LLM response")

    with open(data_config_path, encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    processed_dir = data_cfg["data"]["processed_dir"]
    device = get_device("auto")
    splits = create_data_splits(
        processed_dir, data_config_path, device=device,
        max_backtrack_days=100, max_future_days=10,
    )
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -8) / close - 1
    train_calc = CryptoAlphaCalculator(splits["train"], target)

    scored = score_seeds(valid, train_calc, min_abs_ic=min_abs_ic)
    selected = scored[:top_k]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "model": model,
        "n_generate": n_generate,
        "n_raw": len(raw_items),
        "n_parsed": len(valid),
        "n_scored": len(scored),
        "n_selected": len(selected),
        "min_abs_ic": min_abs_ic,
        "seeds": [
            {
                "family": s["family"],
                "hypothesis": s["hypothesis"],
                "expr": s["expr_str"],
                "train_ic": s["train_ic"],
            }
            for s in selected
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Wrote {len(selected)} seeds to {out_path}")
    for s in selected:
        logger.info(f"  IC={s['train_ic']:+.4f}  [{s['family']}]  {s['expr_str']}")


def load_seed_expressions(path: Path) -> list[Expression]:
    """Re-parse the saved warm-seed JSON into Expression objects for `force_load_exprs`."""
    parser = _build_parser()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    exprs = []
    for s in payload["seeds"]:
        try:
            exprs.append(parser.parse(s["expr"]))
        except Exception as e:
            logger.warning(f"Re-parse failed for {s['expr']!r}: {e}")
    return exprs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-generate", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--data-config", default="config/data_config.yaml")
    ap.add_argument("--min-abs-ic", type=float, default=0.01)
    args = ap.parse_args()
    run(
        seed=args.seed,
        n_generate=args.n_generate,
        top_k=args.top_k,
        out_path=args.out,
        model=args.model,
        data_config_path=args.data_config,
        min_abs_ic=args.min_abs_ic,
    )
