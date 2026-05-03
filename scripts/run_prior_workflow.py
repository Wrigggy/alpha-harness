"""Run prior-guided filtering, robustness audit, and BRAIN rewrite end-to-end."""

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
    parser = argparse.ArgumentParser(description="Run prior-guided alpha workflow")
    parser.add_argument("--pool", default="data/factors/prior_alpha_pool.json")
    parser.add_argument("--data-config", default="config/data_config_equity_brain_top50.yaml")
    parser.add_argument("--brain-config", default="config/brain_proxy_equity_cn.yaml")
    parser.add_argument("--owner", default="local")
    parser.add_argument("--family", default="prior")
    parser.add_argument("--output-root", default="out/prior_workflow")
    args = parser.parse_args()

    python_exe = sys.executable
    out_root = ROOT / args.output_root
    filters_dir = out_root / "filters"
    audit_dir = out_root / "audit"
    rewrite_dir = out_root / "rewrite"
    report_dir = out_root / "report"
    filters_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    rewrite_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    filtered_pool = filters_dir / "prior_val_soft_filtered.json"

    run_step([
        python_exe,
        str(ROOT / "scripts" / "soft_filter_alpha_pool.py"),
        "--pool", args.pool,
        "--data-config", args.data_config,
        "--split", "val",
        "--output", str(filtered_pool),
        "--min-rank-ic", "0.015",
        "--min-rank-icir", "0.15",
        "--max-turnover", "0.50",
        "--min-decay-halflife", "2",
        "--max-correlation", "0.65",
        "--keep-top-k", "5",
    ])

    run_step([
        python_exe,
        str(ROOT / "scripts" / "robust_factor_audit.py"),
        "--pool", str(filtered_pool),
        "--data-config", args.data_config,
        "--split", "test",
        "--output-dir", str(audit_dir),
        "--min-rank-ic", "0.02",
        "--min-rank-icir", "0.20",
        "--max-correlation", "0.65",
        "--noise-sigma", "0.02",
    ])

    run_step([
        python_exe,
        str(ROOT / "scripts" / "rewrite_brain_pool.py"),
        "--pool", str(filtered_pool),
        "--data-config", args.data_config,
        "--brain-config", args.brain_config,
        "--split", "test",
        "--output-dir", str(rewrite_dir),
    ])

    run_step([
        python_exe,
        str(ROOT / "scripts" / "generate_alpha_report.py"),
        "--pool", str(filtered_pool),
        "--data-config", args.data_config,
        "--split", "test",
        "--backtest-years", "5",
        "--output-dir", str(report_dir),
    ])

    logger.info("Prior workflow complete under {}", out_root)


if __name__ == "__main__":
    main()
