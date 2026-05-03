"""Registry for team-owned alpha candidates and BRAIN submission tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_expression(expression: str) -> str:
    return " ".join(str(expression).split())


def expression_fingerprint(expression: str) -> str:
    normalized = normalize_expression(expression).encode("utf-8")
    return hashlib.sha1(normalized).hexdigest()


@dataclass
class SubmissionEvent:
    submitted_at: str
    brain_status: str
    brain_alpha_name: str | None = None
    brain_alpha_id: str | None = None
    notes: str | None = None


@dataclass
class AlphaRecord:
    alpha_id: str
    expression: str
    fingerprint: str
    owner: str
    family: str
    source_pool: str = ""
    source_model: str = ""
    weight: float = 0.0
    tags: list[str] = field(default_factory=list)
    status: str = "drafted"
    brain_status: str = "not_submitted"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    submitted_at: str | None = None
    last_evaluated_at: str | None = None
    brain_alpha_name: str | None = None
    brain_alpha_id: str | None = None
    notes: str | None = None
    proxy_metrics: dict[str, Any] = field(default_factory=dict)
    submission_history: list[SubmissionEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["submission_history"] = [asdict(event) for event in self.submission_history]
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "AlphaRecord":
        history = [SubmissionEvent(**item) for item in raw.get("submission_history", [])]
        data = dict(raw)
        data["submission_history"] = history
        return cls(**data)


@dataclass
class GovernanceRegistry:
    schema_version: int = 1
    updated_at: str = field(default_factory=utc_now_iso)
    records: list[AlphaRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "updated_at": self.updated_at,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GovernanceRegistry":
        return cls(
            schema_version=int(raw.get("schema_version", 1)),
            updated_at=raw.get("updated_at", utc_now_iso()),
            records=[AlphaRecord.from_dict(item) for item in raw.get("records", [])],
        )


def load_registry(path: str | Path) -> GovernanceRegistry:
    registry_path = Path(path)
    if not registry_path.exists():
        return GovernanceRegistry()
    with registry_path.open(encoding="utf-8") as f:
        return GovernanceRegistry.from_dict(json.load(f))


def save_registry(registry: GovernanceRegistry, path: str | Path) -> None:
    registry.updated_at = utc_now_iso()
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry.to_dict(), f, indent=2, ensure_ascii=False)


def build_alpha_id(expression: str) -> str:
    return expression_fingerprint(expression)[:12]


def make_record(
    expression: str,
    owner: str,
    family: str,
    source_pool: str = "",
    source_model: str = "",
    weight: float = 0.0,
    tags: list[str] | None = None,
    status: str = "drafted",
    brain_status: str = "not_submitted",
    notes: str | None = None,
) -> AlphaRecord:
    normalized = normalize_expression(expression)
    return AlphaRecord(
        alpha_id=build_alpha_id(normalized),
        expression=normalized,
        fingerprint=expression_fingerprint(normalized),
        owner=owner,
        family=family,
        source_pool=source_pool,
        source_model=source_model,
        weight=weight,
        tags=tags or [],
        status=status,
        brain_status=brain_status,
        notes=notes,
    )


def upsert_record(
    registry: GovernanceRegistry,
    record: AlphaRecord,
    overwrite_metadata: bool = False,
) -> tuple[AlphaRecord, str]:
    registry.updated_at = utc_now_iso()
    for idx, existing in enumerate(registry.records):
        if existing.alpha_id != record.alpha_id:
            continue
        existing.updated_at = utc_now_iso()
        if overwrite_metadata:
            existing.owner = record.owner
            existing.family = record.family
            existing.source_pool = record.source_pool
            existing.source_model = record.source_model
            existing.weight = record.weight
            existing.tags = record.tags
            existing.status = record.status
            existing.notes = record.notes
        else:
            existing.owner = existing.owner or record.owner
            existing.family = existing.family or record.family
            existing.source_pool = existing.source_pool or record.source_pool
            existing.source_model = existing.source_model or record.source_model
            existing.weight = existing.weight or record.weight
            existing.tags = sorted(set(existing.tags).union(record.tags))
            existing.notes = existing.notes or record.notes
        registry.records[idx] = existing
        return existing, "updated"

    registry.records.append(record)
    return record, "created"


def update_submission(
    registry: GovernanceRegistry,
    alpha_id: str,
    brain_status: str,
    submitted_at: str | None = None,
    brain_alpha_name: str | None = None,
    brain_alpha_id: str | None = None,
    notes: str | None = None,
    status: str | None = None,
) -> AlphaRecord:
    submit_time = submitted_at or utc_now_iso()
    for record in registry.records:
        if record.alpha_id != alpha_id:
            continue
        record.updated_at = utc_now_iso()
        record.submitted_at = submit_time
        record.brain_status = brain_status
        record.brain_alpha_name = brain_alpha_name or record.brain_alpha_name
        record.brain_alpha_id = brain_alpha_id or record.brain_alpha_id
        record.notes = notes or record.notes
        if status is not None:
            record.status = status
        record.submission_history.append(
            SubmissionEvent(
                submitted_at=submit_time,
                brain_status=brain_status,
                brain_alpha_name=brain_alpha_name,
                brain_alpha_id=brain_alpha_id,
                notes=notes,
            )
        )
        return record
    raise KeyError(f"alpha_id not found in registry: {alpha_id}")


def update_proxy_metrics(
    registry: GovernanceRegistry,
    alpha_id: str,
    metrics: dict[str, Any],
    evaluated_at: str | None = None,
) -> AlphaRecord:
    stamp = evaluated_at or utc_now_iso()
    for record in registry.records:
        if record.alpha_id != alpha_id:
            continue
        record.updated_at = utc_now_iso()
        record.last_evaluated_at = stamp
        record.proxy_metrics = metrics
        return record
    raise KeyError(f"alpha_id not found in registry: {alpha_id}")


def registry_to_frame(registry: GovernanceRegistry) -> pd.DataFrame:
    rows = []
    for record in registry.records:
        row = {
            "alpha_id": record.alpha_id,
            "owner": record.owner,
            "family": record.family,
            "status": record.status,
            "brain_status": record.brain_status,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "submitted_at": record.submitted_at,
            "last_evaluated_at": record.last_evaluated_at,
            "source_pool": record.source_pool,
            "source_model": record.source_model,
            "weight": record.weight,
            "brain_alpha_name": record.brain_alpha_name,
            "brain_alpha_id": record.brain_alpha_id,
            "tags": ",".join(record.tags),
            "expression": record.expression,
            "notes": record.notes,
        }
        for key, value in record.proxy_metrics.items():
            row[f"proxy_{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)
