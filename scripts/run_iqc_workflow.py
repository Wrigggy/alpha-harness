"""Run the local IQC workflow end-to-end for one factor pool."""

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
    parser = argparse.ArgumentParser(description="Run register -> proxy evaluate -> governance dashboard")
    parser.add_argument("--pool", required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--source-model", default="alphagen")
    parser.add_argument("--registry", default="data/governance/team_registry.json")
    parser.add_argument("--data-config", default="config/data_config_equity_brain_top50.yaml")
    parser.add_argument("--brain-config", default="config/brain_proxy_equity_cn.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tags", default="")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--status", default="drafted")
    parser.add_argument("--overwrite-metadata", action="store_true")
    parser.add_argument("--skip-register", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--output-root", default="out")
    args = parser.parse_args()

    python_exe = sys.executable
    pool_stem = Path(args.pool).stem
    brain_out = Path(args.output_root) / "brain_proxy" / f"{pool_stem}_{args.split}"
    gov_out = Path(args.output_root) / "governance" / f"{pool_stem}_{args.split}"

    if not args.skip_register:
        cmd = [
            python_exe,
            str(ROOT / "scripts" / "register_alpha_pool.py"),
            "--pool", args.pool,
            "--registry", args.registry,
            "--owner", args.owner,
            "--family", args.family,
            "--source-model", args.source_model,
            "--status", args.status,
            "--tags", args.tags,
        ]
        if args.notes is not None:
            cmd += ["--notes", args.notes]
        if args.overwrite_metadata:
            cmd.append("--overwrite-metadata")
        run_step(cmd)

    if not args.skip_evaluate:
        cmd = [
            python_exe,
            str(ROOT / "scripts" / "evaluate_brain_candidates.py"),
            "--pool", args.pool,
            "--data-config", args.data_config,
            "--brain-config", args.brain_config,
            "--split", args.split,
            "--registry", args.registry,
            "--output-dir", str(brain_out),
        ]
        run_step(cmd)

    if not args.skip_dashboard:
        cmd = [
            python_exe,
            str(ROOT / "scripts" / "analyze_team_registry.py"),
            "--registry", args.registry,
            "--data-config", args.data_config,
            "--brain-config", args.brain_config,
            "--split", args.split,
            "--output-dir", str(gov_out),
        ]
        run_step(cmd)

    logger.info("IQC workflow complete. proxy_out={} governance_out={}", brain_out, gov_out)


if __name__ == "__main__":
    main()
