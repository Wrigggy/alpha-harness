"""Compose-mode ablation summary: 5 pools (2 vanilla/warm RL + 2 judge-filtered).

Reads data/factors/{A_qcm, B_compose_alphagen, B_compose_qcm,
C_compose_alphagen_judge, C_compose_qcm_judge}_cn_seed{SEED}_pool.json and
prints a table of pool_size / val_ic / test_ic / test_ric. Also writes
out/compose_summary_seed{SEED}.json for downstream notebooks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACT_DIR = ROOT / "data" / "factors"
OUT_DIR = ROOT / "out"


def _pool_path(cond: str, seed: int) -> Path:
    """Map condition name to actual pool filename.

    The judge filter writes <prefix>_judge_pool.json so the suffix order is
    different from the vanilla/warm pools.
    """
    if cond.startswith("C_compose"):
        engine = cond.split("_")[2]  # alphagen | qcm
        return FACT_DIR / f"C_compose_{engine}_cn_seed{seed}_judge_pool.json"
    return FACT_DIR / f"{cond}_cn_seed{seed}_pool.json"


CONDITIONS = [
    "A_qcm",
    "B_compose_alphagen",
    "B_compose_qcm",
    "C_compose_alphagen_judge",
    "C_compose_qcm_judge",
]


def _format_metric(v):
    if v is None:
        return "  n/a "
    try:
        return f"{float(v):+.4f}"
    except (TypeError, ValueError):
        return str(v)


def main(seed: int) -> None:
    rows = []
    for cond in CONDITIONS:
        path = _pool_path(cond, seed)
        if not path.exists():
            rows.append({"condition": cond, "status": "MISSING"})
            continue
        pool = json.loads(path.read_text())
        rows.append({
            "condition": cond,
            "status": "OK",
            "pool_size": len(pool.get("exprs", [])),
            "val_ic": pool.get("val_ic"),
            "val_ric": pool.get("val_ric"),
            "test_ic": pool.get("test_ic"),
            "test_ric": pool.get("test_ric"),
            "n_steps": pool.get("n_steps"),
        })

    print()
    print("=" * 88)
    print(f"Compose-mode ablation summary  (seed={seed})")
    print("=" * 88)
    header = f"{'condition':<32} {'size':>4}  {'val_IC':>8}  {'val_RIC':>8}  {'test_IC':>8}  {'test_RIC':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r["status"] != "OK":
            print(f"{r['condition']:<32}   --  ----   MISSING")
            continue
        print(
            f"{r['condition']:<32} "
            f"{r.get('pool_size','?'):>4}  "
            f"{_format_metric(r.get('val_ic')):>8}  "
            f"{_format_metric(r.get('val_ric')):>8}  "
            f"{_format_metric(r.get('test_ic')):>8}  "
            f"{_format_metric(r.get('test_ric')):>8}"
        )
    print("=" * 88)
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"compose_summary_seed{seed}.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"Summary written to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args.seed)
