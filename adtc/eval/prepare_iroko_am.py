#!/usr/bin/env python3
"""Prepare frozen Iroko / Masakhane Amharic (+ English) eval JSONL.

Does not mix into training. Writes under adtc/data/eval/ and a SHA256 manifest.
Requires: pip install datasets
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "eval"
RAW_DIR = ROOT / "data" / "raw"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_afrimgsm(example: dict, lang: str, idx: int) -> dict:
    # Fields vary slightly by release; keep common ones.
    question = example.get("question") or example.get("query") or ""
    answer = example.get("answer") or example.get("target") or example.get("response") or ""
    return {
        "id": f"afrimgsm_{lang}_{idx:04d}",
        "suite": "afrimgsm",
        "lang": lang,
        "split": "test",
        "question": question,
        "answer": answer,
        "raw": {k: example[k] for k in example if k not in ("question", "answer")},
    }


def prepare_afrimgsm(langs: list[str], limit: int | None) -> list[Path]:
    from datasets import load_dataset

    written: list[Path] = []
    for lang in langs:
        ds = load_dataset("masakhane/afrimgsm", lang, split="test", cache_dir=str(RAW_DIR / "hf"))
        rows = []
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            rows.append(_normalize_afrimgsm(dict(ex), lang, i))
        out = OUT_DIR / f"afrimgsm_{lang}_test_v0.jsonl"
        _write_jsonl(out, rows)
        written.append(out)
        print(f"wrote {out} ({len(rows)} rows)")
    return written


def prepare_optional(name: str, config: str, limit: int | None) -> Path | None:
    """Best-effort AfriMMLU / AfriXNLI; skip if config missing."""
    from datasets import load_dataset

    try:
        ds = load_dataset(name, config, split="test", cache_dir=str(RAW_DIR / "hf"))
    except Exception as e:  # noqa: BLE001 — surface missing configs clearly
        print(f"skip {name}/{config}: {e}")
        return None
    rows = []
    for i, ex in enumerate(ds):
        if limit is not None and i >= limit:
            break
        rows.append(
            {
                "id": f"{name.split('/')[-1]}_{config}_{i:04d}",
                "suite": name,
                "lang": config,
                "split": "test",
                "example": dict(ex),
            }
        )
    out = OUT_DIR / f"{name.split('/')[-1]}_{config}_test_v0.jsonl"
    _write_jsonl(out, rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return out


def write_manifest(paths: list[Path]) -> Path:
    manifest = {
        "version": "v0",
        "files": [
            {"path": str(p.relative_to(ROOT)), "sha256": _sha256_file(p), "n_lines": sum(1 for _ in p.open())}
            for p in paths
            if p is not None and p.exists()
        ],
    }
    out = OUT_DIR / "eval_manifest_v0.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Cap rows per split (smoke)")
    parser.add_argument("--skip-optional", action="store_true", help="Only AfriMGSM amh+eng")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    paths.extend(prepare_afrimgsm(["amh", "eng"], args.limit))
    if not args.skip_optional:
        for name, cfg in (
            ("masakhane/afrimmlu", "amh"),
            ("masakhane/afrixnli", "amh"),
        ):
            p = prepare_optional(name, cfg, args.limit)
            if p:
                paths.append(p)
    write_manifest(paths)


if __name__ == "__main__":
    main()
