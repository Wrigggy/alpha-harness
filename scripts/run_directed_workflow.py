"""Run the directed local-search workflow end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger


ROOT = Path(__file__).resolve().parents[1]


def run_step(args: list[str]) -> None:
    logger.info("Running: {}", " ".join(args))
    subprocess.run(args, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run directed alpha workflow")
    parser.add_argument("--data-config", default="config/data_config_equity_latest_top47_baostock.yaml")
    parser.add_argument("--brain-config", default="config/brain_proxy_equity_cn_weekly.yaml")
    parser.add_argument("--output-root", default="out/directed_workflow_top47_weekly")
    args = parser.parse_args()

    python_exe = sys.executable
    out_root = ROOT / args.output_root
    build_dir = out_root / "build"
    filter_dir = out_root / "filters"
    audit_dir = out_root / "audit"
    rewrite_dir = out_root / "rewrite"
    eval_dir = out_root / "eval"
    for path in [build_dir, filter_dir, audit_dir, rewrite_dir, eval_dir]:
        path.mkdir(parents=True, exist_ok=True)

    directed_pool = build_dir / "directed_alpha_pool.json"
    filtered_pool = filter_dir / "directed_val_soft_filtered.json"

    run_step([
        python_exe,
        str(ROOT / "scripts" / "build_directed_alpha_pool.py"),
        "--output",
        str(directed_pool),
    ])

    run_step([
        python_exe,
        str(ROOT / "scripts" / "soft_filter_alpha_pool.py"),
        "--pool",
        str(directed_pool),
        "--data-config",
        args.data_config,
        "--split",
        "val",
        "--output",
        str(filtered_pool),
        "--min-rank-ic",
        "0.003",
        "--min-rank-icir",
        "0.03",
        "--max-turnover",
        "0.70",
        "--min-decay-halflife",
        "1",
        "--max-correlation",
        "0.80",
        "--keep-top-k",
        "12",
    ])

    run_step([
        python_exe,
        str(ROOT / "scripts" / "robust_factor_audit.py"),
        "--pool",
        str(filtered_pool),
        "--data-config",
        args.data_config,
        "--split",
        "test",
        "--output-dir",
        str(audit_dir),
        "--min-rank-ic",
        "0.003",
        "--min-rank-icir",
        "0.03",
        "--max-correlation",
        "0.80",
        "--noise-sigma",
        "0.02",
    ])

    run_step([
        python_exe,
        str(ROOT / "scripts" / "rewrite_brain_pool.py"),
        "--pool",
        str(filtered_pool),
        "--data-config",
        args.data_config,
        "--brain-config",
        args.brain_config,
        "--split",
        "test",
        "--output-dir",
        str(rewrite_dir),
    ])

    run_step([
        python_exe,
        str(ROOT / "scripts" / "evaluate_brain_candidates.py"),
        "--pool",
        str(filtered_pool),
        "--data-config",
        args.data_config,
        "--brain-config",
        args.brain_config,
        "--split",
        "test",
        "--output-dir",
        str(eval_dir),
    ])

    logger.info("Directed workflow complete under {}", out_root)


if __name__ == "__main__":
    main()
