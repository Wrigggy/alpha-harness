"""Build pairwise combo candidates from a filtered alpha pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ALPHAGEN = ROOT / "external" / "alphagen"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pairwise combo alpha pool")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if str(ROOT) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(ROOT))
    if str(EXTERNAL_ALPHAGEN) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(EXTERNAL_ALPHAGEN))

    from alphagen.data.parser import parse_expression

    with Path(args.input).open(encoding="utf-8") as f:
        payload = json.load(f)

    exprs = list(dict.fromkeys(payload.get("exprs", [])))
    weights = payload.get("weights", [1.0] * len(exprs))

    combos: list[dict] = []
    for i, left in enumerate(exprs):
        for j, right in enumerate(exprs):
            if j <= i:
                continue
            w_left = float(weights[i] if i < len(weights) else 1.0)
            w_right = float(weights[j] if j < len(weights) else 1.0)
            combos.append(
                {
                    "name": f"sum_{i+1:02d}_{j+1:02d}",
                    "expr": f"Add({left},{right})",
                    "weight": (w_left + w_right) / 2.0,
                }
            )
            combos.append(
                {
                    "name": f"diff_{i+1:02d}_{j+1:02d}",
                    "expr": f"Sub({left},{right})",
                    "weight": (w_left + w_right) / 2.0,
                }
            )
            combos.append(
                {
                    "name": f"mix_{i+1:02d}_{j+1:02d}",
                    "expr": f"Mul({left},{right})",
                    "weight": (w_left + w_right) / 2.0,
                }
            )
            combos.append(
                {
                    "name": f"decay_sum_{i+1:02d}_{j+1:02d}",
                    "expr": f"WMA(Add({left},{right}),10d)",
                    "weight": (w_left + w_right) / 2.0,
                }
            )

    for item in combos:
        parse_expression(item["expr"])

    out = {
        "exprs": [item["expr"] for item in combos],
        "weights": [item["weight"] for item in combos],
        "templates": combos,
        "source": args.input,
        "notes": "Pairwise alpha combos generated from accepted seed factors.",
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Combo alpha pool written to {}", out_path)


if __name__ == "__main__":
    main()
