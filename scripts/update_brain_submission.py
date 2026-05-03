"""Update BRAIN submission status for a registered alpha."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.team_governance.registry import load_registry, save_registry, update_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Track a BRAIN submission event in the team registry")
    parser.add_argument("--registry", default="data/governance/team_registry.json")
    parser.add_argument("--alpha-id", required=True)
    parser.add_argument("--brain-status", required=True, help="submitted / pass / fail / resubmitted / unknown")
    parser.add_argument("--submitted-at", default=None, help="ISO timestamp; defaults to now")
    parser.add_argument("--brain-alpha-name", default=None)
    parser.add_argument("--brain-alpha-id", default=None)
    parser.add_argument("--status", default=None, help="Optional local status override")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()

    registry = load_registry(args.registry)
    record = update_submission(
        registry,
        alpha_id=args.alpha_id,
        brain_status=args.brain_status,
        submitted_at=args.submitted_at,
        brain_alpha_name=args.brain_alpha_name,
        brain_alpha_id=args.brain_alpha_id,
        notes=args.notes,
        status=args.status,
    )
    save_registry(registry, args.registry)
    logger.info("Submission updated for {} -> {}", record.alpha_id, record.brain_status)


if __name__ == "__main__":
    main()
