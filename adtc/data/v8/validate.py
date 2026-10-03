"""Canonical-record and source-artifact validation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from pydantic import ValidationError

from .schema import CanonicalRecord

SOURCE_ARTIFACT_RE = re.compile(
    r"(?:<think>|</think>|####|<<[^>]*>>|<\|(?:assistant|user|system)\|>)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationIssue:
    location: str
    message: str
    kind: str


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    records_seen: int
    records_valid: int
    issues: list[ValidationIssue]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "records_seen": self.records_seen,
            "records_valid": self.records_valid,
            "issues": [asdict(issue) for issue in self.issues],
        }


def find_source_artifacts(record: CanonicalRecord) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    content_fields = {
        "grounding.problem": record.grounding.problem,
        **{
            f"messages[{index}].content": message.content
            for index, message in enumerate(record.messages)
        },
    }
    if record.student_state:
        content_fields["student_state.attempt"] = record.student_state.attempt
    for location, content in content_fields.items():
        match = SOURCE_ARTIFACT_RE.search(content)
        if match:
            issues.append(
                ValidationIssue(
                    location=location,
                    message=f"banned source artifact {match.group(0)!r}",
                    kind="source_artifact",
                )
            )
    return issues


def validate_record(payload: dict[str, Any]) -> tuple[CanonicalRecord | None, list[ValidationIssue]]:
    try:
        record = CanonicalRecord.model_validate(payload)
    except ValidationError as error:
        issues = [
            ValidationIssue(
                location=".".join(str(part) for part in item["loc"]),
                message=item["msg"],
                kind=item["type"],
            )
            for item in error.errors()
        ]
        return None, issues
    issues = find_source_artifacts(record)
    return (record if not issues else None), issues


def validate_records(payloads: Iterable[dict[str, Any]]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    seen = 0
    valid = 0
    for index, payload in enumerate(payloads, start=1):
        seen += 1
        record, row_issues = validate_record(payload)
        if record is not None:
            valid += 1
        issues.extend(
            ValidationIssue(
                location=f"record[{index}].{issue.location}",
                message=issue.message,
                kind=issue.kind,
            )
            for issue in row_issues
        )
    return ValidationReport(
        valid=not issues,
        records_seen=seen,
        records_valid=valid,
        issues=issues,
    )


def validate_jsonl(path: Path) -> ValidationReport:
    payloads: list[dict[str, Any]] = []
    parse_issues: list[ValidationIssue] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("record must be a JSON object")
                payloads.append(value)
            except (json.JSONDecodeError, TypeError) as error:
                parse_issues.append(
                    ValidationIssue(
                        location=f"line[{line_number}]",
                        message=str(error),
                        kind="json_parse",
                    )
                )
    report = validate_records(payloads)
    all_issues = parse_issues + report.issues
    return ValidationReport(
        valid=not all_issues,
        records_seen=report.records_seen + len(parse_issues),
        records_valid=report.records_valid,
        issues=all_issues,
    )
