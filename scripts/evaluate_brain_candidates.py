"""Evaluate pool expressions as single-alpha BRAIN-style local candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ALPHAGEN = ROOT / "external" / "alphagen"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(EXTERNAL_ALPHAGEN) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ALPHAGEN))

from alphagen.data.parser import parse_expression

from src.brain_proxy.evaluator import (
    build_proxy_context,
    evaluate_brain_candidate,
    tensor_to_factor_frame,
)
from src.brain_proxy.expression_translation import BrainExpressionTranslator
from src.brain_proxy.expression_translation import build_brain_variants, render_brain_submission_template
from src.team_governance.registry import (
    build_alpha_id,
    load_registry,
    save_registry,
    update_proxy_metrics,
    utc_now_iso,
)
from src.utils.pool_io import load_pool


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a pool with BRAIN-style proxy metrics")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--data-config", default="config/data_config_equity_cn.yaml")
    parser.add_argument("--brain-config", default="config/brain_proxy_equity_cn.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--registry", default=None, help="Optional registry path to update proxy metrics")
    args = parser.parse_args()

    expressions, _ = load_pool(args.pool)
    context = build_proxy_context(
        data_config_path=args.data_config,
        split=args.split,
        brain_config_path=args.brain_config,
    )

    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "out" / "brain_proxy" / f"{Path(args.pool).stem}_{args.split}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    submit_dir = out_dir / "submit_ready_worldquant"
    submit_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    variant_rows: list[dict] = []
    registry = load_registry(args.registry) if args.registry else None
    evaluated_at = utc_now_iso()
    translator = BrainExpressionTranslator()

    for idx, expression in enumerate(expressions, start=1):
        parsed = parse_expression(expression)
        translation = translator.translate(parsed)
        factor_df = tensor_to_factor_frame(
            context["stock_data"],
            context["calculator"].evaluate_alpha(parsed),
            name=f"factor_{idx:02d}",
        )
        _, summary = evaluate_brain_candidate(
            expression=expression,
            factor_df=factor_df,
            panel=context["panel"],
            forward_returns=context["forward_returns"],
            next_bar_returns=context["next_bar_returns"],
            liquidity_df=context["liquidity_df"],
            bars_per_year=context["bars_per_year"],
            cfg=context["brain_cfg"],
        )
        summary["alpha_id"] = build_alpha_id(expression)
        summary["worldquant_expression"] = translation.worldquant_expression
        summary["translation_supported"] = translation.supported
        summary["translation_notes"] = " | ".join(translation.notes)
        rows.append(summary)

        template = render_brain_submission_template(
            alpha_name=summary["alpha_id"],
            base_expression=translation.worldquant_expression,
            local_expression=expression,
        )
        (submit_dir / f"{summary['alpha_id']}.txt").write_text(template, encoding="utf-8")
        for variant in build_brain_variants(translation.worldquant_expression):
            variant_rows.append(
                {
                    "alpha_id": summary["alpha_id"],
                    "variant": variant.label,
                    "expression": variant.expression,
                    "notes": " | ".join(variant.notes),
                }
            )

        if registry is not None:
            try:
                update_proxy_metrics(registry, summary["alpha_id"], summary, evaluated_at=evaluated_at)
            except KeyError:
                logger.warning("alpha_id {} not found in registry; skipping metric update", summary["alpha_id"])

    result = pd.DataFrame(rows).sort_values(["passes_proxy_gates", "brain_readiness_score"], ascending=[False, False])
    result.to_csv(out_dir / "brain_candidate_metrics.csv", index=False)
    result[
        [
            "alpha_id",
            "expression",
            "worldquant_expression",
            "translation_supported",
            "translation_notes",
        ]
    ].to_csv(out_dir / "worldquant_expression_candidates.csv", index=False)
    pd.DataFrame(variant_rows).to_csv(out_dir / "brain_expression_variants.csv", index=False)

    accepted = result[
        (result["passes_submission_gates"])
        & (result["weight_concentration"] <= 0.10)
        & (result["sub_universe_sharpe"] >= 0.2)
        & (result["sharpe_ratio"] >= 1.25)
        & (result["fitness"] >= 1.0)
    ].copy()
    accepted.to_csv(out_dir / "accepted_for_submission.csv", index=False)
    if submit_dir.exists():
        keep_names = set(accepted["alpha_id"].tolist())
        for txt in submit_dir.glob("*.txt"):
            if txt.stem not in keep_names:
                txt.unlink()
        for txt in submit_dir.glob("*.txt"):
            if txt.stem not in keep_names:
                txt.unlink()

    with (out_dir / "brain_candidate_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    if registry is not None and args.registry:
        save_registry(registry, args.registry)

    logger.info("BRAIN proxy candidate report written to {}", out_dir)


if __name__ == "__main__":
    main()
