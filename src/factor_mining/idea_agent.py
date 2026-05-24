"""LLM idea-agent: warm-start factors for the RL pool.

Two modes:
  pick    — LLM picks factor IDs from the library; we load library expressions
            directly. (Baseline; known to over-saturate the incremental-IC reward.)
  compose — LLM composes NEW expressions using library IDs as primitives,
            conditioned on a regime summary. Resolver expands the F-IDs into
            full RPN expressions. (Default — leaves headroom for the RL agent.)

Usage:
    python -m src.factor_mining.idea_agent --mode compose \\
        --seed 42 --top-k 10 \\
        --data-source cn \\
        --out data/factors/warm_seeds_cn_compose_seed42.json
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

from src.llm_client import get_llm_client  # noqa: E402


LIBRARY_PATH = _ROOT / "data" / "factor_library.json"
PROMPT_PICK_PATH = _ROOT / "prompts" / "idea_agent_pick.txt"
PROMPT_COMPOSE_PATH = _ROOT / "prompts" / "idea_agent_compose.txt"

# Library IDs follow `<prefix>_<3 digits>` (e.g., m_008, cp_003)
ID_TOKEN_RE = re.compile(r"\b([a-z]+_\d{3})\b")


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
    rows = []
    for f in lib:
        rows.append(f"{f['id']:>10s} | {f['family']:>14s} | {f['description']} | {f['source']}")
    return "\n".join(rows)


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


# ---------------------------------------------------------------------------
# Pick mode (baseline — LLM returns ID lists)
# ---------------------------------------------------------------------------

def pick_factors(lib: list[dict], client, seed: int) -> list[dict[str, Any]]:
    template = PROMPT_PICK_PATH.read_text(encoding="utf-8")
    rendered = template.replace("{LIBRARY}", _render_library_block(lib))
    raw = client.complete(rendered, seed=seed)
    items = _extract_json_array(raw)
    logger.info(f"LLM returned picks for {len(items)} hypotheses")
    return items


def resolve_picks(picks: list[dict[str, Any]], lib: list[dict]) -> list[dict[str, Any]]:
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


# ---------------------------------------------------------------------------
# Compose mode (LLM emits new expressions with F-IDs as primitives)
# ---------------------------------------------------------------------------

def expand_template(template: str, by_id: dict[str, dict]) -> tuple[str, list[str]]:
    """Substitute every library F-ID token with its canonical RPN expression.
    Returns (expanded_expr, list_of_used_ids_in_order).
    """
    flat = re.sub(r"\s+", "", template)
    used: list[str] = []

    def _sub(match: re.Match) -> str:
        fid = match.group(1)
        if fid in by_id:
            used.append(fid)
            return by_id[fid]["expr"]
        return match.group(0)

    expanded = ID_TOKEN_RE.sub(_sub, flat)
    return expanded, used


def compose_factors(lib: list[dict], regime: str, client, seed: int) -> list[dict[str, Any]]:
    template = PROMPT_COMPOSE_PATH.read_text(encoding="utf-8")
    rendered = (
        template
        .replace("{LIBRARY}", _render_library_block(lib))
        .replace("{REGIME}", regime)
    )
    raw = client.complete(rendered, seed=seed)
    items = _extract_json_array(raw)
    logger.info(f"LLM returned compositions for {len(items)} hypotheses")
    return items


def resolve_composed(items: list[dict[str, Any]], lib: list[dict]) -> list[dict[str, Any]]:
    """Flatten composed templates → one entry per composition.

    Drops compositions that are a bare ID (pick-mode in disguise) or that
    reference unknown F-IDs after expansion.
    """
    by_id = {f["id"]: f for f in lib}
    known_ids = set(by_id.keys())
    bare_id = re.compile(r"^[a-z]+_\d{3}$")
    out: list[dict[str, Any]] = []
    for entry in items:
        hyp = entry.get("hypothesis", "")
        regime_rel = entry.get("regime_relevance", "")
        for comp in entry.get("compositions", []):
            template = (comp.get("template") or "").strip()
            if not template:
                continue
            flat = re.sub(r"\s+", "", template)
            if bare_id.match(flat):
                logger.warning(f"compose: bare ID template {template!r} ignored (pick-mode disguise)")
                continue
            expanded, used = expand_template(template, by_id)
            unknown = set(ID_TOKEN_RE.findall(expanded)) - known_ids
            if unknown:
                logger.warning(f"compose: unknown IDs {sorted(unknown)} in template {template!r} (dropped)")
                continue
            out.append({
                "id": f"composed_{len(out):03d}",
                "family": "composed",
                "expr": expanded,
                "template": template,
                "description": comp.get("rationale", ""),
                "source": "compose",
                "hypotheses": [hyp],
                "regime_relevances": [regime_rel],
                "reasonings": [comp.get("rationale", "")],
                "used_library_ids": sorted(set(used)),
            })
    return out


# ---------------------------------------------------------------------------
# Shared scoring + emission pipeline
# ---------------------------------------------------------------------------

def score_picks(
    resolved: list[dict[str, Any]],
    train_calc,
    parser: ExpressionParser,
    min_abs_ic: float = 0.005,
) -> list[dict[str, Any]]:
    """Parse, IC-score on train, sign-flip negatives so all emitted seeds have IC > 0."""
    scored = []
    for item in resolved:
        try:
            expr = parser.parse(item["expr"])
        except ExpressionParsingError as e:
            logger.warning(f"Expr failed to parse for {item['id']}: {e}")
            continue
        try:
            ic = float(train_calc.calc_single_IC_ret(expr))
        except Exception as e:
            logger.warning(f"IC compute failed for {item['id']}: {e}")
            continue
        if not np.isfinite(ic) or abs(ic) < min_abs_ic:
            logger.debug(f"{item['id']} dropped: |IC|={ic:.4f} < {min_abs_ic}")
            continue
        if ic < 0:
            flipped = f"Mul(-1.0,{item['expr']})"
            try:
                _ = parser.parse(flipped)
                item["expr"] = flipped
                item["sign_flipped"] = True
                ic = -ic
            except ExpressionParsingError as e:
                logger.warning(f"{item['id']} sign-flip failed: {e} (using raw expr)")
        item["train_ic"] = ic
        scored.append(item)
    scored.sort(key=lambda x: x["train_ic"], reverse=True)
    return scored


def run(
    seed: int,
    top_k: int,
    out_path: Path,
    data_source: str,
    model: str | None,
    data_config_path: str,
    min_abs_ic: float,
    mode: str,
    llm_backend: str | None,
    provider: str | None = None,
) -> None:
    np.random.seed(seed)

    extra: dict = {}
    if provider:
        extra["provider"] = provider
    client = get_llm_client(backend=llm_backend, model=model, **extra)
    logger.info(
        f"LLM client: backend={client.backend} model={client.model}"
        + (f" provider={provider}" if provider else "")
    )

    lib = _load_library()
    logger.info(f"Loaded {len(lib)} factors from library | mode={mode}")

    from src.factor_mining._calc_factory import build_calculators
    logger.info(f"Building {data_source} train calculator...")
    calcs = build_calculators(
        data_source=data_source,
        data_config_path=data_config_path,
        splits_to_load=("train",),
    )
    train_calc = calcs["train"]
    parser = _build_parser()

    if mode == "pick":
        picks = pick_factors(lib, client=client, seed=seed)
        resolved = resolve_picks(picks, lib)
        raw_payload: Any = picks
        regime: str | None = None
    elif mode == "compose":
        from src.factor_mining._regime import compute_regime_summary
        regime = compute_regime_summary(train_calc)
        logger.info(f"Regime: {regime}")
        items = compose_factors(lib, regime=regime, client=client, seed=seed)
        resolved = resolve_composed(items, lib)
        raw_payload = {"regime": regime, "items": items}
    else:
        raise ValueError(f"Unknown mode {mode!r} (expected pick | compose)")

    logger.info(f"Resolved to {len(resolved)} candidate {mode} items")
    if not resolved:
        raise RuntimeError(f"LLM returned no valid {mode} items")

    scored = score_picks(resolved, train_calc, parser, min_abs_ic=min_abs_ic)
    logger.info(f"{len(scored)} items passed |IC| ≥ {min_abs_ic}")

    selected = scored[:top_k]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seeds_out = []
    for s in selected:
        rec = {
            "id": s["id"],
            "family": s["family"],
            "expr": s["expr"],
            "description": s["description"],
            "source": s["source"],
            "hypotheses": s.get("hypotheses", []),
            "train_ic": s["train_ic"],
        }
        if "template" in s:
            rec["template"] = s["template"]
            rec["used_library_ids"] = s.get("used_library_ids", [])
        if s.get("sign_flipped"):
            rec["sign_flipped"] = True
        seeds_out.append(rec)

    payload = {
        "seed": seed,
        "llm_backend": client.backend,
        "model": client.model,
        "data_source": data_source,
        "mode": mode,
        "regime": regime,
        "n_library": len(lib),
        "n_resolved": len(resolved),
        "n_scored": len(scored),
        "n_selected": len(selected),
        "min_abs_ic": min_abs_ic,
        "raw_llm_output": raw_payload,
        "seeds": seeds_out,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Wrote {len(selected)} {mode} seeds to {out_path}")
    for s in selected:
        hyps = ",".join(s.get("hypotheses", [])[:2])
        tag = "[composed]" if s.get("source") == "compose" else f"[{s['family']:>10s}]"
        logger.info(f"  IC={s['train_ic']:+.4f}  {tag}  {s['id']:>11s}  {s['expr']}  ({hyps})")


def load_seed_expressions(path: Path) -> list[Expression]:
    """Re-parse warm-seed JSON into Expression objects for `force_load_exprs`.

    Works for both pick-mode and compose-mode emitted files — the `expr` field
    is always a final RPN string, regardless of mode.
    """
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
    ap.add_argument("--mode", choices=["pick", "compose"], default="compose",
                    help="pick=LLM returns library IDs; compose=LLM emits new exprs (default)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-source", choices=["crypto", "cn"], default="cn")
    ap.add_argument(
        "--llm-backend",
        choices=["claude_code", "openrouter", "anthropic"],
        default=None,
        help="LLM backend. Falls back to $LLM_BACKEND, then 'claude_code'.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model name; backend-specific default if omitted "
             "(claude_code=claude-opus-4-7, openrouter=deepseek/deepseek-chat).",
    )
    ap.add_argument("--data-config", default="config/data_config.yaml")
    ap.add_argument("--min-abs-ic", type=float, default=0.005)
    ap.add_argument(
        "--provider",
        default=None,
        help="OpenRouter provider routing — e.g. 'DeepSeek' to force the "
             "official DeepSeek endpoint. Falls back to $OPENROUTER_PROVIDER.",
    )
    args = ap.parse_args()
    run(
        seed=args.seed,
        top_k=args.top_k,
        out_path=args.out,
        data_source=args.data_source,
        model=args.model,
        data_config_path=args.data_config,
        min_abs_ic=args.min_abs_ic,
        mode=args.mode,
        llm_backend=args.llm_backend,
        provider=args.provider,
    )
