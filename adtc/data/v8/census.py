"""No-generation v8 supply census.

The census counts only rows that have already passed deterministic hard gates in
metadata inventory files. Unpinned or unapproved sources contribute zero.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .sources import DEFAULT_SOURCES, SourceEntry, SourcesManifest, load_sources
from .vocab import BEHAVIORS, CELL_MINIMUM, CELL_TARGET, SUBJECTS

ADTC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ADTC_ROOT / "docs" / "artifacts" / "v8" / "supply_census.json"


class InventoryRecord(BaseModel):
    """Metadata-only importer output consumed by the census."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source_id: str
    family_id: str = Field(min_length=1)
    original_id: str = Field(min_length=1)
    source_split: str
    subject: Literal["math", "physics", "chemistry", "biology", "earth_science"]
    eligible_behaviors: list[Literal["solve", "explain", "hint", "diagnose"]]
    grade_band: str
    grounding_available: bool
    license_class: str
    hard_gate_pass: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    template_fingerprint: str | None = None


def _read_inventory(path: Path) -> list[InventoryRecord]:
    records: list[InventoryRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(InventoryRecord.model_validate(payload))
            except Exception as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def _resolve_inventory(source: SourceEntry, project_root: Path) -> Path | None:
    if not source.inventory_path:
        return None
    path = Path(source.inventory_path)
    return path if path.is_absolute() else project_root / path


def _remediation(subject: str) -> list[str]:
    if subject == "biology":
        return [
            "approve a licensed grounding-only biology reference corpus",
            "add solver-backed generators for genetics and population models",
            "revise the matrix through a separate go/no-go decision",
        ]
    if subject == "earth_science":
        return [
            "approve a licensed grounding-only Earth-science reference corpus",
            "add solver-backed generators for half-life, lapse rate, and plate motion",
            "revise the matrix through a separate go/no-go decision",
        ]
    return [
        "pin and approve declared sources",
        "produce hard-gate metadata inventories",
        "add valid structured sources or generators before model spend",
    ]


def build_census(
    manifest: SourcesManifest,
    *,
    project_root: Path = ADTC_ROOT,
    extra_inventory_paths: list[Path] | None = None,
) -> dict[str, Any]:
    source_by_id = {source.id: source for source in manifest.sources}
    inventories: list[InventoryRecord] = []
    source_reports: list[dict[str, Any]] = []

    for source in manifest.sources:
        path = _resolve_inventory(source, project_root)
        loaded: list[InventoryRecord] = []
        status = "not_configured"
        if path is not None:
            if path.is_file():
                loaded = _read_inventory(path)
                status = "loaded"
            else:
                status = "missing"
        inventories.extend(loaded)
        source_reports.append(
            {
                "source_id": source.id,
                "allowed_use": source.allowed_use,
                "review_status": source.review_status,
                "revision_pinned": bool(source.revision),
                "inventory_path": str(path) if path else None,
                "inventory_status": status,
                "inventory_rows": len(loaded),
                "declared_capabilities": source.capabilities.model_dump(),
                "counted_families": 0,
                "rejected_rows": 0,
            }
        )

    for path in extra_inventory_paths or []:
        inventories.extend(_read_inventory(path))

    source_report_by_id = {report["source_id"]: report for report in source_reports}
    families_by_cell: dict[tuple[str, str], set[str]] = defaultdict(set)
    sources_by_cell: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    counted_by_source: dict[str, set[str]] = defaultdict(set)

    for record in inventories:
        source = source_by_id.get(record.source_id)
        report = source_report_by_id.get(record.source_id)
        if source is None:
            raise ValueError(f"inventory references unknown source {record.source_id!r}")
        rejection_reasons: list[str] = list(record.rejection_reasons)
        if not source.allowed_use:
            rejection_reasons.append("source_not_approved")
        if not source.revision:
            rejection_reasons.append("revision_not_pinned")
        if record.source_split != source.imported_split:
            rejection_reasons.append("split_mismatch")
        if record.grade_band != "secondary_9_12":
            rejection_reasons.append("outside_target_grade")
        if record.license_class != source.license_class:
            rejection_reasons.append("license_class_mismatch")
        if not record.hard_gate_pass:
            rejection_reasons.append("hard_gate_failed")

        if rejection_reasons:
            if report:
                report["rejected_rows"] += 1
            continue

        counted_by_source[record.source_id].add(record.family_id)
        for behavior in record.eligible_behaviors:
            if behavior in {"explain", "hint", "diagnose"} and not record.grounding_available:
                continue
            cell = (record.subject, behavior)
            families_by_cell[cell].add(record.family_id)
            sources_by_cell[cell][record.source_id].add(record.family_id)

    for source_id, families in counted_by_source.items():
        source_report_by_id[source_id]["counted_families"] = len(families)

    cells: list[dict[str, Any]] = []
    all_ready = True
    for subject in SUBJECTS:
        for behavior in BEHAVIORS:
            key = (subject, behavior)
            count = len(families_by_cell[key])
            ready = count >= CELL_MINIMUM
            all_ready = all_ready and ready
            cells.append(
                {
                    "subject": subject,
                    "behavior": behavior,
                    "post_dedup_family_lower_bound": count,
                    "preferred_target": CELL_TARGET,
                    "minimum_for_full_build": CELL_MINIMUM,
                    "gap_to_minimum": max(0, CELL_MINIMUM - count),
                    "status": "ready" if ready else "short",
                    "by_source": {
                        source_id: len(families)
                        for source_id, families in sorted(sources_by_cell[key].items())
                    },
                    "remediation": [] if ready else _remediation(subject),
                }
            )

    technical_gate = all(
        cell["status"] in {"ready", "short"} for cell in cells
    ) and len(cells) == len(SUBJECTS) * len(BEHAVIORS)
    return {
        "report_version": "tebeb-v8-supply-census-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_calls": 0,
        "manifest_version": manifest.manifest_version,
        "policy": {
            "counts_only_approved_pinned_sources": True,
            "requires_secondary_9_12": True,
            "grounding_required_for": ["explain", "hint", "diagnose"],
            "family_deduplication": "global family_id",
        },
        "sources": source_reports,
        "cells": cells,
        "summary": {
            "inventory_rows_seen": len(inventories),
            "approved_sources": sum(source.allowed_use for source in manifest.sources),
            "ready_cells": sum(cell["status"] == "ready" for cell in cells),
            "short_cells": sum(cell["status"] == "short" for cell in cells),
            "all_cells_ready_for_model_spend": all_ready,
        },
        "gate_a": {
            "tooling_complete": technical_gate,
            "supply_decision": "go" if all_ready else "blocked_short_cells",
            "may_start_bulk_model_calls": all_ready,
        },
    }


def run_census(
    *,
    sources_path: Path = DEFAULT_SOURCES,
    output_path: Path = DEFAULT_OUTPUT,
    extra_inventory_paths: list[Path] | None = None,
) -> dict[str, Any]:
    manifest = load_sources(sources_path)
    report = build_census(
        manifest,
        extra_inventory_paths=extra_inventory_paths,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
