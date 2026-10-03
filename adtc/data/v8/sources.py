"""Versioned source-manifest loading and fail-closed provenance checks."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_SOURCES = (
    Path(__file__).resolve().parents[1] / "manifests" / "v8" / "sources.yaml"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SourceCapabilities(StrictModel):
    subjects: list[
        Literal["math", "physics", "chemistry", "biology", "earth_science"]
    ]
    behaviors: list[Literal["solve", "explain", "hint", "diagnose"]]
    grade_bands: list[str]
    has_grounding: bool


class SourceEntry(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    name: str
    dataset: str
    url: str
    revision: str | None
    config: str
    imported_split: str
    license: str
    license_class: Literal[
        "permissive",
        "share_alike",
        "noncommercial",
        "custom_restricted",
        "unknown",
        "excluded",
    ]
    attribution: str
    allowed_use: bool
    export_profile: Literal[
        "permissive", "sharealike", "noncommercial", "internal_only", "excluded"
    ]
    review_status: Literal["approved", "pending_pin", "pending_license", "excluded"]
    reviewed_at: str | None
    inventory_path: str | None = None
    capabilities: SourceCapabilities
    notes: str = ""

    @model_validator(mode="after")
    def fail_closed(self) -> "SourceEntry":
        if self.allowed_use:
            if not self.revision:
                raise ValueError("allowed sources require a pinned revision")
            if self.review_status != "approved":
                raise ValueError("allowed sources must have review_status=approved")
            if self.license_class in {"unknown", "excluded"}:
                raise ValueError("unknown/excluded licenses cannot be allowed")
            if self.export_profile == "excluded":
                raise ValueError("allowed sources cannot use export_profile=excluded")
        return self


class SourcesManifest(StrictModel):
    manifest_version: Literal["tebeb-sources-v8.1"]
    generated_at: str
    policy: str
    sources: list[SourceEntry]

    @model_validator(mode="after")
    def unique_ids(self) -> "SourcesManifest":
        ids = [source.id for source in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("source IDs must be unique")
        return self


def load_sources(path: Path = DEFAULT_SOURCES) -> SourcesManifest:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"source manifest must be a mapping: {path}")
    return SourcesManifest.model_validate(payload)
