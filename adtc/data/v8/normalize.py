"""Versioned normalization and hashing primitives for v8 identity."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

NORMALIZATION_VERSION = "norm-v1"
_NUMBER_RE = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])"
)
_SPACE_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"\s*([,.;:!?=+\-*/^(){}\[\]])\s*")
_CAPITALIZED_TOKEN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def normalize_text(text: str) -> str:
    """Normalize text for exact-content comparisons.

    ``norm-v1`` deliberately preserves mathematical symbols and word order while
    normalizing Unicode, case, whitespace, punctuation spacing, and comma-separated
    numeric literals.
    """

    value = unicodedata.normalize("NFKC", text)
    value = value.replace("\u2212", "-").replace("\u00d7", "*").replace("\u00f7", "/")
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)
    value = _PUNCT_SPACE_RE.sub(r"\1", value)
    value = _SPACE_RE.sub(" ", value).strip().casefold()
    return value


def template_normalize(text: str) -> str:
    """Replace surface constants and likely names while preserving structure."""

    value = unicodedata.normalize("NFKC", text)
    value = _CAPITALIZED_TOKEN_RE.sub("<name>", value)
    value = _NUMBER_RE.sub("<number>", value)
    return normalize_text(value)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_id(*parts: str, namespace: str) -> str:
    payload = "\x1f".join((namespace, NORMALIZATION_VERSION, *parts))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def content_hash(text: str) -> str:
    return sha256_id(normalize_text(text), namespace="canonical-content")


def message_hash(content: str, *, role: str) -> str:
    return sha256_id(role, normalize_text(content), namespace="message")


def template_fingerprint(text: str) -> str:
    return sha256_id(template_normalize(text), namespace="template")


def constraint_signature(items: Iterable[Mapping[str, Any]]) -> str:
    normalized = sorted(
        (
            {
                "id": str(item["id"]),
                "parameters": item.get("parameters", {}),
            }
            for item in items
        ),
        key=lambda item: item["id"],
    )
    return sha256_id(stable_json(normalized), namespace="constraints")
