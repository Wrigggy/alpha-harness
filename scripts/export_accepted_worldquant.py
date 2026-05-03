"""Export only fully accepted WorldQuant-style expressions into a dedicated folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export accepted WorldQuant expressions")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for file in out_dir.glob("*"):
        if file.is_file():
            file.unlink()

    accepted = df[
        (df["passes_submission_gates"] == True)
        & (df["sharpe_ratio"] >= 1.25)
        & (df["fitness"] >= 1.0)
        & (df["sub_universe_sharpe"] >= 0.2)
        & (df["weight_concentration"] <= 0.10)
    ].copy()

    accepted.to_csv(out_dir / "accepted_for_submission.csv", index=False)
    for row in accepted.to_dict(orient="records"):
        alpha_id = str(row["alpha_id"])
        expression = str(row["worldquant_expression"])
        payload = [
            f"/* NAME: {alpha_id}",
            f"LOCAL_EXPRESSION: {row['expression']}",
            "IMPLEMENTATION: Accepted by local hard-gate proxy.",
            "*/",
            "",
            expression,
            "",
        ]
        (out_dir / f"{alpha_id}.txt").write_text("\n".join(payload), encoding="utf-8")

    logger.info("Exported {} accepted expressions to {}", len(accepted), out_dir)


if __name__ == "__main__":
    main()
