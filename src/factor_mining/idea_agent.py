"""LLM idea-agent: pick warm-start factors from a curated 80-factor library.

The LLM is given (a) the library and (b) 10 market hypotheses, and returns 1-2
factor IDs per hypothesis. We then look up the expressions, score single-factor
IC on the chosen data source's train calculator, and emit a top-k seed JSON.

Usage:
    python -m src.factor_mining.idea_agent \\
        --seed 42 --top-k 10 \\
        --data-source cn \\
        --out data/factors/warm_seeds_cn_seed42.json

    # crypto smoke test
    python -m src.factor_mining.idea_agent \\
        --seed 42 --top-k 5 \\
        --data-source crypto \\
        --out data/factors/warm_seeds_smoke.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "external" / "alphagen"))

from alphagen.config import OPERATORS  # noqa: E402
from alphagen.data.expression import Expression  # noqa: E402
from alphagen.data.parser import ExpressionParser, ExpressionParsingError  # noqa: E402


LIBRARY_PATH = _ROOT / "data" / "factor_library.json"
PROMPT_PATH = _ROOT / "prompts" / "idea_agent_pick.txt"


def _build_parser() -> ExpressionParser:
    return ExpressionParser(
        operators=OPERATORS,
        ignore_case=False,
        time_deltas_need_suffix=True,
        non_positive_time_deltas_allowed=False,
        feature_need_dollar_sign=True,
    )


def _load_library() -> list[dict[str, Any]]:
    lib = json.loads(LIBRARY_PATH.read_text(encoding="utf-8"))
    if not lib:
        raise RuntimeError(f"Empty factor library at {LIBRARY_PATH}")
    return lib


def _render_library_block(lib: list[dict]) -> str:
    """Compact library table for the prompt. Hides full RPN expression to keep
    tokens down and to force the LLM to commit by ID rather than by expression."""
    rows = []
    for f in lib:
        rows.append(f"{f['id']:>10s} | {f['family']:>14s} | {f['description']} | {f['source']}")
    return "\n".join(rows)


def _load_prompt(lib: list[dict]) -> str:
    template = PROMPT_PATH.read_text(encoding="utf-8")
    return template.replace("{LIBRARY}", _render_library_block(lib))


def _extract_json_array(text: str) -> list[dict[str, Any]]:
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


def pick_factors(lib: list[dict], model: str, seed: int) -> list[dict[str, Any]]:
    """Returns the raw LLM output (list of {hypothesis, chosen_ids, reasoning})."""
    import anyio
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    async def _call(prompt: str) -> str:
        full = f"{prompt}\n\n[generation_seed_hint={seed}]"
        result = ""
        async for message in query(
            prompt=full,
            options=ClaudeAgentOptions(model=model, allowed_tools=[], max_turns=1),
        ):
            if isinstance(message, ResultMessage):
                result = message.result
        if not result:
            raise RuntimeError("Empty response from Claude")
        return result

    prompt = _load_prompt(lib)
    raw = anyio.run(_call, prompt)
    items = _extract_json_array(raw)
    logger.info(f"LLM returned picks for {len(items)} hypotheses")
    return items


def resolve_picks(
    picks: list[dict[str, Any]],
    lib: list[dict],
) -> list[dict[str, Any]]:
    """Look up chosen IDs in the library, dedupe, attach hypothesis tag.

    Returns one entry per unique factor:
        {id, family, expr, description, hypotheses[], reasonings[]}
    Picks that reference an unknown ID are logged and skipped.
    """
    by_id = {f["id"]: f for f in lib}
    bucket: dict[str, dict] = {}
    for p in picks:
        hyp = p.get("hypothesis", "")
        reasoning = p.get("reasoning", "")
        for fid in p.get("chosen_ids", []):
            if fid not in by_id:
                logger.warning(f"LLM picked unknown ID {fid!r} for {hyp!r}")
                continue
            if fid not in bucket:
                f = by_id[fid]
                bucket[fid] = {
                    "id": fid,
                    "family": f["family"],
                    "expr": f["expr"],
                    "description": f["description"],
                    "source": f["source"],
                    "hypotheses": [hyp],
                    "reasonings": [reasoning],
                }
            else:
                bucket[fid]["hypotheses"].append(hyp)
                bucket[fid]["reasonings"].append(reasoning)
    return list(bucket.values())


def score_picks(
    resolved: list[dict[str, Any]],
    train_calc,
    parser: ExpressionParser,
    min_abs_ic: float = 0.005,
) -> list[dict[str, Any]]:
    scored = []
    for item in resolved:
        try:
            expr = parser.parse(item["expr"])
        except ExpressionParsingError as e:
            logger.warning(f"Library expr failed to parse for {item['id']}: {e}")
            continue
        try:
            ic = float(train_calc.calc_single_IC_ret(expr))
        except Exception as e:
            logger.warning(f"IC compute failed for {item['id']}: {e}")
            continue
        if not np.isfinite(ic) or abs(ic) < min_abs_ic:
            logger.debug(f"{item['id']} dropped: |IC|={ic:.4f} < {min_abs_ic}")
            continue
        item["train_ic"] = ic
        scored.append(item)
    scored.sort(key=lambda x: abs(x["train_ic"]), reverse=True)
    return scored


def run(
    seed: int,
    top_k: int,
    out_path: Path,
    data_source: str,
    model: str,
    data_config_path: str,
    min_abs_ic: float,
) -> None:
    np.random.seed(seed)

    lib = _load_library()
    logger.info(f"Loaded {len(lib)} factors from library")

    picks = pick_factors(lib, model=model, seed=seed)
    resolved = resolve_picks(picks, lib)
    logger.info(f"Resolved to {len(resolved)} unique factors")
    if not resolved:
        raise RuntimeError("LLM returned no valid picks")

    parser = _build_parser()
    from src.factor_mining._calc_factory import build_calculators
    logger.info(f"Building {data_source} train calculator...")
    calcs = build_calculators(
        data_source=data_source,
        data_config_path=data_config_path,
        splits_to_load=("train",),
    )
    train_calc = calcs["train"]

    scored = score_picks(resolved, train_calc, parser, min_abs_ic=min_abs_ic)
    logger.info(f"{len(scored)} factors passed |IC| ≥ {min_abs_ic}")

    selected = scored[:top_k]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "model": model,
        "data_source": data_source,
        "mode": "pick",
        "n_library": len(lib),
        "n_picks_raw": sum(len(p.get("chosen_ids", [])) for p in picks),
        "n_resolved": len(resolved),
        "n_scored": len(scored),
        "n_selected": len(selected),
        "min_abs_ic": min_abs_ic,
        "raw_picks": picks,
        "seeds": [
            {
                "id": s["id"],
                "family": s["family"],
                "expr": s["expr"],
                "description": s["description"],
                "source": s["source"],
                "hypotheses": s["hypotheses"],
                "train_ic": s["train_ic"],
            }
            for s in selected
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Wrote {len(selected)} seeds to {out_path}")
    for s in selected:
        hyps = ",".join(s["hypotheses"][:2])
        logger.info(f"  IC={s['train_ic']:+.4f}  [{s['family']:>14s}]  {s['id']:>8s}  {s['expr']}  ({hyps})")


def load_seed_expressions(path: Path) -> list[Expression]:
    """Re-parse the saved warm-seed JSON into Expression objects for `force_load_exprs`."""
    parser = _build_parser()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    exprs = []
    for s in payload["seeds"]:
        try:
            exprs.append(parser.parse(s["expr"]))
        except Exception as e:
            logger.warning(f"Re-parse failed for {s.get('id', s.get('expr'))!r}: {e}")
    return exprs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-source", choices=["crypto", "cn"], default="cn",
                    help="Which calculator to use for IC scoring (default: cn)")
    ap.add_argument("--model", default="claude-opus-4-7")
    ap.add_argument("--data-config", default="config/data_config.yaml")
    ap.add_argument("--min-abs-ic", type=float, default=0.005)
    args = ap.parse_args()
    run(
        seed=args.seed,
        top_k=args.top_k,
        out_path=args.out,
        data_source=args.data_source,
        model=args.model,
        data_config_path=args.data_config,
        min_abs_ic=args.min_abs_ic,
    )
