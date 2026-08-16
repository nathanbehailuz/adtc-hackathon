#!/usr/bin/env python3
"""Build English STEM held-out eval from GSM8K test (never train).

Writes adtc/data/eval/en_stem_holdout_v0.jsonl
Requires: pip install datasets
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval" / "en_stem_holdout_v0.jsonl"
RAW = ROOT / "data" / "raw" / "hf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100, help="Holdout size (default 100)")
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test", cache_dir=str(RAW))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with OUT.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i >= args.limit:
                break
            row = {
                "id": f"en_stem_gsm8k_test_{i:04d}",
                "suite": "en_stem_holdout",
                "lang": "en",
                "split": "test",
                "question": ex["question"],
                "answer": ex["answer"],
                "behavior": "solve",
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {OUT} ({n} rows)")


if __name__ == "__main__":
    main()
