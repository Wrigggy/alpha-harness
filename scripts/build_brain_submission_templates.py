"""Build BRAIN submission templates and refinement variants from a local pool."""

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

from src.brain_proxy.expression_translation import (
    BrainExpressionTranslator,
    build_brain_variants,
    render_brain_submission_template,
)
from src.utils.pool_io import load_pool, normalize_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Build WorldQuant BRAIN submission templates from a local pool")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    expressions, raw_weights = load_pool(args.pool)
    weights = normalize_weights(raw_weights)
    translator = BrainExpressionTranslator()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "out" / "brain_submission_templates" / Path(args.pool).stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    submit_dir = out_dir / "submit_ready_worldquant"
    submit_dir.mkdir(parents=True, exist_ok=True)

    variant_rows: list[dict] = []
    template_rows: list[dict] = []

    for idx, (expression, weight) in enumerate(zip(expressions, weights), start=1):
        name = f"alpha_{idx:02d}"
        result = translator.translate(expression)
        template = render_brain_submission_template(
            alpha_name=name,
            base_expression=result.worldquant_expression,
            local_expression=expression,
        )
        (submit_dir / f"{name}.txt").write_text(template, encoding="utf-8")
        template_rows.append(
            {
                "name": name,
                "expression": expression,
                "worldquant_expression": result.worldquant_expression,
                "weight": weight,
                "translation_supported": result.supported,
                "translation_notes": " | ".join(result.notes),
                "template_file": f"submit_ready_worldquant/{name}.txt",
            }
        )

        for variant in build_brain_variants(result.worldquant_expression):
            variant_rows.append(
                {
                    "name": name,
                    "variant": variant.label,
                    "expression": variant.expression,
                    "notes": " | ".join(variant.notes),
                }
            )

    pd.DataFrame(template_rows).to_csv(out_dir / "brain_submission_templates.csv", index=False)
    pd.DataFrame(variant_rows).to_csv(out_dir / "brain_expression_variants.csv", index=False)
    with (out_dir / "brain_submission_templates.json").open("w", encoding="utf-8") as f:
        json.dump(template_rows, f, indent=2, ensure_ascii=False)

    logger.info("BRAIN submission templates written to {}", out_dir)


if __name__ == "__main__":
    main()
