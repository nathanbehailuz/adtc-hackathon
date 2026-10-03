"""Stable identities and SQLite uniqueness registry for v8 builds."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from .normalize import sha256_id

DEFAULT_REGISTRY = (
    Path(__file__).resolve().parents[1] / "manifests" / "v8" / "identity_registry.sqlite"
)

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS source_records (
    source_key TEXT PRIMARY KEY,
    dataset TEXT NOT NULL,
    revision TEXT NOT NULL,
    source_split TEXT NOT NULL,
    original_id TEXT NOT NULL,
    raw_record_hash TEXT,
    status TEXT NOT NULL DEFAULT 'candidate',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_rows (
    row_id TEXT PRIMARY KEY,
    family_id TEXT NOT NULL,
    source_key TEXT NOT NULL REFERENCES source_records(source_key),
    selected_behavior TEXT NOT NULL,
    constraint_signature TEXT NOT NULL,
    renderer_version TEXT NOT NULL,
    canonical_content_hash TEXT NOT NULL,
    template_fingerprint TEXT NOT NULL,
    semantic_cluster_id TEXT,
    status TEXT NOT NULL,
    duplicate_of TEXT REFERENCES training_rows(row_id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(family_id, selected_behavior, constraint_signature, renderer_version)
);

CREATE INDEX IF NOT EXISTS idx_training_family ON training_rows(family_id);
CREATE INDEX IF NOT EXISTS idx_training_template ON training_rows(template_fingerprint);

CREATE TABLE IF NOT EXISTS build_membership (
    build_name TEXT NOT NULL,
    row_id TEXT NOT NULL REFERENCES training_rows(row_id),
    membership TEXT NOT NULL CHECK (
        membership IN ('qa_pilot', 'mini_train', 'full', 'evaluation')
    ),
    exposure_count INTEGER NOT NULL DEFAULT 0 CHECK (exposure_count >= 0),
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(build_name, row_id, membership)
);
"""


def make_source_key(
    dataset: str, revision: str, source_split: str, original_id: str
) -> str:
    return f"{dataset}@{revision}/{source_split}/{original_id}"


def make_family_id(
    *,
    source_key: str | None = None,
    generator_template_version: str | None = None,
    surface_context: str | None = None,
) -> str:
    """Create a source family or a generator template+scenario family ID."""

    if source_key:
        if generator_template_version or surface_context:
            raise ValueError("source and generator family inputs are mutually exclusive")
        return sha256_id(source_key, namespace="source-family")
    if generator_template_version and surface_context:
        return sha256_id(
            generator_template_version,
            surface_context,
            namespace="generator-family",
        )
    raise ValueError(
        "provide source_key or both generator_template_version and surface_context"
    )


def make_row_id(
    family_id: str,
    selected_behavior: str,
    constraint_signature: str,
    renderer_version: str,
) -> str:
    return sha256_id(
        family_id,
        selected_behavior,
        constraint_signature,
        renderer_version,
        namespace="training-row",
    )


class IdentityRegistry:
    """Transactional interface to the cross-build identity registry."""

    def __init__(self, path: Path = DEFAULT_REGISTRY) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")

    def __enter__(self) -> "IdentityRegistry":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()

    def initialize(self) -> None:
        self.connection.executescript(SCHEMA_SQL)
        self.connection.commit()

    def register_source(
        self,
        *,
        source_key: str,
        dataset: str,
        revision: str,
        source_split: str,
        original_id: str,
        raw_record_hash: str | None = None,
        status: str = "candidate",
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO source_records (
                source_key, dataset, revision, source_split, original_id,
                raw_record_hash, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_key,
                dataset,
                revision,
                source_split,
                original_id,
                raw_record_hash,
                status,
            ),
        )

    def register_training_row(
        self,
        *,
        row_id: str,
        family_id: str,
        source_key: str,
        selected_behavior: str,
        constraint_signature: str,
        renderer_version: str,
        canonical_content_hash: str,
        template_fingerprint: str,
        status: str,
        semantic_cluster_id: str | None = None,
        duplicate_of: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO training_rows (
                row_id, family_id, source_key, selected_behavior,
                constraint_signature, renderer_version, canonical_content_hash,
                template_fingerprint, semantic_cluster_id, status, duplicate_of
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row_id,
                family_id,
                source_key,
                selected_behavior,
                constraint_signature,
                renderer_version,
                canonical_content_hash,
                template_fingerprint,
                semantic_cluster_id,
                status,
                duplicate_of,
            ),
        )

    def add_membership(
        self,
        *,
        build_name: str,
        row_id: str,
        membership: Literal["qa_pilot", "mini_train", "full", "evaluation"],
        exposure_count: int = 0,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO build_membership (
                build_name, row_id, membership, exposure_count
            ) VALUES (?, ?, ?, ?)
            """,
            (build_name, row_id, membership, exposure_count),
        )

    def counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for table in ("source_records", "training_rows", "build_membership"):
            result[table] = int(
                self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
        return result


def initialize_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    with IdentityRegistry(path) as registry:
        registry.initialize()
        return {"path": str(path), "counts": registry.counts()}
