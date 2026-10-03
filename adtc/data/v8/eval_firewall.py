"""Frozen-evaluation hashing and training-candidate firewall."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .normalize import NORMALIZATION_VERSION, content_hash, template_fingerprint

ADTC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ADTC_ROOT / "data" / "manifests" / "v8" / "frozen_eval_hashes.json"
DEFAULT_EVAL_FILES = (
    ADTC_ROOT / "data" / "eval" / "afrimgsm_amh_test_v0.jsonl",
    ADTC_ROOT / "data" / "eval" / "afrimgsm_eng_test_v0.jsonl",
    ADTC_ROOT / "data" / "eval" / "afrimmlu_amh_test_v0.jsonl",
    ADTC_ROOT / "data" / "eval" / "afrixnli_amh_test_v0.jsonl",
    ADTC_ROOT / "data" / "eval" / "en_stem_holdout_v0.jsonl",
    ADTC_ROOT / "data" / "eval" / "custom_tutoring_v0.jsonl",
    ADTC_ROOT / "data" / "eval" / "fertility_parallel_v0.jsonl",
    ADTC_ROOT / "data" / "eval" / "judge_smoke_v7.jsonl",
    ADTC_ROOT / "data" / "eval" / "v8" / "judge_replay_v8.jsonl",
)

EXTERNAL_RESERVATIONS = (
    {
        "dataset": "MRBench",
        "splits": ["all"],
        "status": "pending_download",
        "upstream_families": ["MathDial", "Bridge"],
    },
    {
        "dataset": "BEA-2025-pedagogical-ability",
        "splits": ["all"],
        "status": "pending_download",
        "upstream_families": ["MathDial", "Bridge"],
    },
    {
        "dataset": "MathTutorBench",
        "splits": ["all"],
        "status": "pending_download",
        "upstream_families": ["MathDial", "Bridge", "GSM8K"],
    },
    {
        "dataset": "SceMQA",
        "splits": ["all"],
        "status": "pending_download",
        "upstream_families": [],
    },
    {
        "dataset": "MMLU-high-school",
        "splits": ["auxiliary_train", "dev", "validation", "test"],
        "status": "pending_download",
        "upstream_families": [],
    },
    {
        "dataset": "stemdataset/STEM",
        "splits": ["validation", "test"],
        "status": "pending_source_inspection",
        "upstream_families": [],
    },
    {
        "dataset": "EXAMS",
        "splits": ["dev", "test"],
        "status": "pending_source_inspection",
        "upstream_families": [],
    },
    {
        "dataset": "MATH",
        "splits": ["test"],
        "status": "pending_source_inspection",
        "upstream_families": [],
    },
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def extract_prompt(record: dict[str, Any]) -> str | None:
    messages = record.get("messages")
    if isinstance(messages, list):
        users = [
            message.get("content")
            for message in messages
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        ]
        if users:
            return "\n".join(users)
    example = record.get("example")
    if isinstance(example, dict):
        question = example.get("question")
        if isinstance(question, str) and question.strip():
            choices = example.get("choices")
            return f"{question}\n{choices}" if choices else question
        premise = example.get("premise")
        hypothesis = example.get("hypothesis")
        if isinstance(premise, str) and isinstance(hypothesis, str):
            return f"{premise}\n{hypothesis}"
    english = record.get("en")
    amharic = record.get("am")
    if isinstance(english, str) and isinstance(amharic, str):
        return f"{english}\n{amharic}"
    for key in ("prompt", "question", "text", "input", "premise"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            hypothesis = record.get("hypothesis")
            if key == "premise" and isinstance(hypothesis, str):
                return f"{value}\n{hypothesis}"
            return value
    return None


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, payload


def build_frozen_manifest(paths: Iterable[Path] = DEFAULT_EVAL_FILES) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    exact_hashes: set[str] = set()
    template_hashes: set[str] = set()

    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"frozen eval file is missing: {path}")
        row_count = 0
        prompt_count = 0
        for line_number, record in _iter_jsonl(path):
            row_count += 1
            prompt = extract_prompt(record)
            if not prompt:
                continue
            prompt_count += 1
            exact = content_hash(prompt)
            template = template_fingerprint(prompt)
            exact_hashes.add(exact)
            template_hashes.add(template)
            entries.append(
                {
                    "file": str(path.relative_to(ADTC_ROOT)),
                    "line": line_number,
                    "row_id": str(record.get("id", f"line-{line_number}")),
                    "exact_prompt_hash": exact,
                    "template_fingerprint": template,
                }
            )
        files.append(
            {
                "path": str(path.relative_to(ADTC_ROOT)),
                "sha256": file_sha256(path),
                "rows": row_count,
                "prompts_hashed": prompt_count,
            }
        )

    return {
        "manifest_version": "tebeb-frozen-eval-v8.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "normalization_version": NORMALIZATION_VERSION,
        "policy": {
            "exact_match": "reject",
            "template_match": "reject_pending_adjudication",
            "semantic_match": "Phase C+ embedding firewall",
            "all_training_importers_must_check": True,
        },
        "files": files,
        "entries": entries,
        "exact_prompt_hashes": sorted(exact_hashes),
        "template_fingerprints": sorted(template_hashes),
        "external_reservations": list(EXTERNAL_RESERVATIONS),
    }


def freeze_evaluation(
    *,
    output_path: Path = DEFAULT_MANIFEST,
    paths: Iterable[Path] = DEFAULT_EVAL_FILES,
) -> dict[str, Any]:
    manifest = build_frozen_manifest(paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


@dataclass(frozen=True)
class FirewallDecision:
    allowed: bool
    exact_match: bool
    template_match: bool
    reasons: tuple[str, ...]


class FrozenEvalIndex:
    def __init__(self, manifest_path: Path = DEFAULT_MANIFEST) -> None:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.manifest_version = payload["manifest_version"]
        self.exact_hashes = frozenset(payload["exact_prompt_hashes"])
        self.template_hashes = frozenset(payload["template_fingerprints"])

    def check(self, text: str) -> FirewallDecision:
        exact = content_hash(text) in self.exact_hashes
        template = template_fingerprint(text) in self.template_hashes
        reasons: list[str] = []
        if exact:
            reasons.append("exact_eval_overlap")
        if template:
            reasons.append("template_eval_overlap")
        return FirewallDecision(
            allowed=not reasons,
            exact_match=exact,
            template_match=template,
            reasons=tuple(reasons),
        )

    def assert_allowed(self, text: str) -> None:
        decision = self.check(text)
        if not decision.allowed:
            raise ValueError(
                "training candidate overlaps frozen evaluation: "
                + ", ".join(decision.reasons)
            )


def validate_gate_b(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    eval_root: Path = ADTC_ROOT / "data" / "eval" / "v8",
) -> dict[str, Any]:
    judge_path = eval_root / "judge_replay_v8.jsonl"
    rubric_path = eval_root / "rubrics_v8.json"
    scaffold_path = eval_root / "balanced_tutoring_v8.jsonl"

    judge_rows = [record for _, record in _iter_jsonl(judge_path)]
    judge_ids = [record["id"] for record in judge_rows]
    prompts = [extract_prompt(record) for record in judge_rows]
    if len(judge_rows) != 13 or len(set(judge_ids)) != 13:
        raise ValueError("judge replay must contain exactly 13 unique row IDs")
    if None in prompts or len({content_hash(prompt or "") for prompt in prompts}) != 13:
        raise ValueError("judge replay must contain exactly 13 unique prompts")

    rubrics = json.loads(rubric_path.read_text(encoding="utf-8"))
    defined_checks = set(rubrics["atomic_checks"])
    used_checks = {check for record in judge_rows for check in record["checks"]}
    missing_checks = sorted(used_checks - defined_checks)
    if missing_checks:
        raise ValueError(f"judge checks missing from rubric: {missing_checks}")

    scaffold_rows = [record for _, record in _iter_jsonl(scaffold_path)]
    cells = {(record["subject"], record["behavior"]) for record in scaffold_rows}
    if len(scaffold_rows) != 20 or len(cells) != 20:
        raise ValueError("balanced eval scaffold must contain all 20 unique cells")
    if any(record["minimum_examples"] != 50 for record in scaffold_rows):
        raise ValueError("each balanced eval cell must target 50 examples")

    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_judge = [
        item
        for item in frozen["files"]
        if item["path"] == "data/eval/v8/judge_replay_v8.jsonl"
    ]
    if len(frozen_judge) != 1 or frozen_judge[0]["prompts_hashed"] != 13:
        raise ValueError("judge replay is not fully represented in frozen hashes")
    if not frozen["external_reservations"]:
        raise ValueError("external benchmark reservations are missing")

    return {
        "gate": "B",
        "passed": True,
        "judge_replay_rows": len(judge_rows),
        "rubric_checks": len(defined_checks),
        "balanced_cells_planned": len(cells),
        "frozen_files": len(frozen["files"]),
        "frozen_prompt_hashes": len(frozen["exact_prompt_hashes"]),
        "external_reservations": len(frozen["external_reservations"]),
        "phase_c_imports_may_start": True,
    }
