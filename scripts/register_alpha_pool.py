"""Register a pool of alpha expressions into the team governance registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.team_governance.registry import load_registry, make_record, save_registry, upsert_record
from src.utils.pool_io import load_pool, normalize_weights


def parse_tags(text: str | None) -> list[str]:
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a factor pool into the team alpha registry")
    parser.add_argument("--pool", required=True, help="Path to pool JSON")
    parser.add_argument("--registry", default="data/governance/team_registry.json")
    parser.add_argument("--owner", required=True, help="Team member who owns these candidates")
    parser.add_argument("--family", required=True, help="Alpha family label, e.g. momentum/liquidity")
    parser.add_argument("--source-model", default="alphagen", help="alphagen / alphaqcm / manual / other")
    parser.add_argument("--status", default="drafted", help="drafted / ready / submitted / retired")
    parser.add_argument("--brain-status", default="not_submitted", help="not_submitted / submitted / pass / fail")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--overwrite-metadata", action="store_true")
    args = parser.parse_args()

    expressions, raw_weights = load_pool(args.pool)
    weights = normalize_weights(raw_weights)
    registry = load_registry(args.registry)
    created = 0
    updated = 0
    tags = parse_tags(args.tags)

    for expression, weight in zip(expressions, weights):
        record = make_record(
            expression=expression,
            owner=args.owner,
            family=args.family,
            source_pool=args.pool,
            source_model=args.source_model,
            weight=weight,
            tags=tags,
            status=args.status,
            brain_status=args.brain_status,
            notes=args.notes,
        )
        _, action = upsert_record(registry, record, overwrite_metadata=args.overwrite_metadata)
        if action == "created":
            created += 1
        else:
            updated += 1

    save_registry(registry, args.registry)
    logger.info("Registry updated: created={}, updated={}, path={}", created, updated, args.registry)


if __name__ == "__main__":
    main()
