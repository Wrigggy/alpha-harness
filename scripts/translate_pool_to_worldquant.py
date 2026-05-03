"""Translate a local factor pool into WorldQuant BRAIN-style expressions."""

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

from src.brain_proxy.expression_translation import BrainExpressionTranslator
from src.utils.pool_io import load_pool, normalize_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate local alpha pool expressions to WorldQuant syntax")
    parser.add_argument("--pool", required=True, help="Pool JSON path")
    parser.add_argument("--output-dir", default=None, help="Optional output directory")
    args = parser.parse_args()

    expressions, raw_weights = load_pool(args.pool)
    weights = normalize_weights(raw_weights)
    translator = BrainExpressionTranslator()

    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "out" / "worldquant_translation" / Path(args.pool).stem
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for idx, (expression, weight) in enumerate(zip(expressions, weights), start=1):
        result = translator.translate(expression)
        rows.append(
            {
                "name": f"factor_{idx:02d}",
                "expression": expression,
                "worldquant_expression": result.worldquant_expression,
                "weight": weight,
                "translation_supported": result.supported,
                "translation_notes": " | ".join(result.notes),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "worldquant_expression_map.csv", index=False)
    with (out_dir / "worldquant_expression_map.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    logger.info("WorldQuant translation written to {}", out_dir)


if __name__ == "__main__":
    main()
