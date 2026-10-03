"""Phase A tests; run with ``python -m unittest data.v8.test_phase_a``."""

from __future__ import annotations

import copy
import sqlite3
import tempfile
import unittest
from pathlib import Path

from .census import build_census
from .fixtures import valid_diagnose_record
from .identity import IdentityRegistry
from .normalize import content_hash, normalize_text
from .schema import CanonicalRecord, canonical_json_schema
from .sources import load_sources
from .validate import validate_record


class SchemaTests(unittest.TestCase):
    def test_valid_fixture(self) -> None:
        record, issues = validate_record(valid_diagnose_record())
        self.assertIsNotNone(record)
        self.assertEqual([], issues)

    def test_move_intent_rule(self) -> None:
        payload = valid_diagnose_record()
        payload["pedagogy"]["teacher_move"] = "focus"
        record, issues = validate_record(payload)
        self.assertIsNone(record)
        self.assertTrue(any("invalid for move" in issue.message for issue in issues))

    def test_correct_attempt_contract(self) -> None:
        payload = valid_diagnose_record()
        state = payload["student_state"]
        state.update(
            {
                "first_error_step": None,
                "error_type": "none",
                "error_class": None,
                "error_class_rationale": "Every displayed step is valid.",
                "misconception": "No misconception observed.",
            }
        )
        CanonicalRecord.model_validate(payload)
        state["first_error_step"] = 1
        with self.assertRaises(ValueError):
            CanonicalRecord.model_validate(payload)

    def test_source_artifact_is_rejected(self) -> None:
        payload = valid_diagnose_record()
        payload["messages"][2]["content"] += " #### 33"
        record, issues = validate_record(payload)
        self.assertIsNone(record)
        self.assertTrue(any(issue.kind == "source_artifact" for issue in issues))

    def test_json_schema_exports(self) -> None:
        schema = canonical_json_schema()
        self.assertEqual(
            "https://tebeb.ai/schemas/tebeb-v8.1.schema.json", schema["$id"]
        )


class IdentityTests(unittest.TestCase):
    def test_normalization_is_stable(self) -> None:
        self.assertEqual(normalize_text(" 3 × 4 = 12 "), normalize_text("3*4=12"))
        self.assertEqual(content_hash("3 × 4 = 12"), content_hash(" 3*4=12 "))

    def test_registry_enforces_source_and_logical_uniqueness(self) -> None:
        payload = valid_diagnose_record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.sqlite"
            with IdentityRegistry(path) as registry:
                registry.initialize()
                source = payload["source"]
                identity = payload["identity"]
                dedup = payload["dedup"]
                selection = payload["quality"]["selection"]
                registry.register_source(
                    source_key=identity["source_key"],
                    dataset=source["dataset"],
                    revision=source["revision"],
                    source_split=source["source_split"],
                    original_id=source["original_id"],
                )
                with self.assertRaises(sqlite3.IntegrityError):
                    registry.register_source(
                        source_key=identity["source_key"],
                        dataset=source["dataset"],
                        revision=source["revision"],
                        source_split=source["source_split"],
                        original_id=source["original_id"],
                    )
                registry.register_training_row(
                    row_id=identity["row_id"],
                    family_id=identity["family_id"],
                    source_key=identity["source_key"],
                    selected_behavior=selection["selected_behavior"],
                    constraint_signature=dedup["constraint_signature"],
                    renderer_version="renderer-v1",
                    canonical_content_hash=dedup["canonical_content_hash"],
                    template_fingerprint=dedup["template_fingerprint"],
                    status="accepted",
                )
                duplicate = copy.deepcopy(identity)
                duplicate["row_id"] = "sha256:" + "f" * 64
                with self.assertRaises(sqlite3.IntegrityError):
                    registry.register_training_row(
                        row_id=duplicate["row_id"],
                        family_id=identity["family_id"],
                        source_key=identity["source_key"],
                        selected_behavior=selection["selected_behavior"],
                        constraint_signature=dedup["constraint_signature"],
                        renderer_version="renderer-v1",
                        canonical_content_hash=dedup["canonical_content_hash"],
                        template_fingerprint=dedup["template_fingerprint"],
                        status="accepted",
                    )


class CensusTests(unittest.TestCase):
    def test_empty_inventory_is_explicitly_blocked(self) -> None:
        manifest = load_sources()
        report = build_census(manifest)
        self.assertEqual(20, len(report["cells"]))
        self.assertEqual(20, report["summary"]["short_cells"])
        self.assertFalse(report["gate_a"]["may_start_bulk_model_calls"])
        self.assertEqual("blocked_short_cells", report["gate_a"]["supply_decision"])


if __name__ == "__main__":
    unittest.main()
