#!/usr/bin/env python3
"""Build English STEM tutoring SFT JSONL from GSM8K train (+ optional SciQ).

Never uses GSM8K test / AfriMGSM. Output schema matches DATASETS.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hf"
OUT_DEFAULT = ROOT / "data" / "train" / "en_stem_sft_v0.jsonl"

TEMPLATES = [
    (
        "solve",
        "Solve the following problem step by step.\n\n{q}",
        "{a}",
    ),
    (
        "explain",
        "Explain how to solve this problem clearly for a student.\n\n{q}",
        "{a}",
    ),
    (
        "hint",
        "A student is stuck on this problem. Give one helpful hint without revealing the final numeric answer.\n\n{q}",
        "Focus on the first operation you need to undo, and write what that step achieves before computing the final value.",
    ),
]


def emit(row_id: str, behavior: str, user: str, assistant: str, source: str) -> dict:
    return {
        "id": row_id,
        "direction": "en_en",
        "behavior": behavior,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--include-sciq", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    from datasets import load_dataset

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        gsm = load_dataset("openai/gsm8k", "main", split="train", cache_dir=str(RAW))
        for i, ex in enumerate(gsm):
            if n >= args.limit:
                break
            q, a = ex["question"], ex["answer"]
            behavior, user_t, asst_t = TEMPLATES[i % len(TEMPLATES)]
            row = emit(
                f"en_en_{behavior}_gsm8k_{i:05d}",
                behavior,
                user_t.format(q=q),
                asst_t.format(a=a, q=q),
                "gsm8k_train",
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

        if args.include_sciq and n < args.limit:
            sciq = load_dataset("allenai/sciq", split="train", cache_dir=str(RAW))
            for j, ex in enumerate(sciq):
                if n >= args.limit:
                    break
                q = ex["question"]
                a = ex.get("correct_answer") or ""
                support = ex.get("support") or ""
                assistant = a if not support else f"{a}\n\nExplanation: {support}"
                row = emit(
                    f"en_en_explain_sciq_{j:05d}",
                    "explain",
                    f"Answer and briefly explain:\n\n{q}",
                    assistant,
                    "sciq_train",
                )
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1

    print(f"wrote {args.out} ({n} rows)")


if __name__ == "__main__":
    main()
