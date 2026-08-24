#!/usr/bin/env python3
"""Build English-only SFT mix v6: full GSM8K train tutoring + SciQ.

Outputs:
  data/train/sft_mix_v6.jsonl
  docs/artifacts/v6/sft_mix_v6_counts.json
  docs/artifacts/v6/sft_mix_v6_report.md

Dedup against frozen EN eval sets. Does not touch v0–v5 mixes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

for _k in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "4")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hf"
OUT_DEFAULT = ROOT / "data" / "train" / "sft_mix_v6.jsonl"
COUNTS_DEFAULT = ROOT / "docs" / "artifacts" / "v6" / "sft_mix_v6_counts.json"
REPORT_DEFAULT = ROOT / "docs" / "artifacts" / "v6" / "sft_mix_v6_report.md"

FINAL_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
CALC_RE = re.compile(r"<<([^>=]+)=([^>]+)>>")

EVAL_PATHS = [
    ROOT / "data" / "eval" / "en_stem_holdout_v0.jsonl",
    ROOT / "data" / "eval" / "afrimgsm_eng_test_v0.jsonl",
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


def final_answer(answer: str) -> str | None:
    m = FINAL_RE.search(answer or "")
    return m.group(1) if m else None


def hint_from_gsm8k(question: str, answer: str) -> str:
    calcs = CALC_RE.findall(answer or "")
    if calcs:
        expr, _val = calcs[0]
        return (
            f"Hint: start by computing ({expr.strip()}). Write that intermediate result, "
            f"then continue — do not jump to the final total yet."
        )
    nums = re.findall(r"\d+(?:\.\d+)?", question or "")
    if len(nums) >= 2:
        return (
            f"Hint: use the numbers {nums[0]} and {nums[1]} first to form one clear "
            f"intermediate quantity, then decide what remains. Do not state the final answer."
        )
    return (
        "Hint: identify the first operation you must perform, write what it achieves, "
        "and stop before the final numeric answer."
    )


def first_error_from_gsm8k(question: str, answer: str) -> tuple[str, str] | None:
    gold = final_answer(answer)
    if gold is None:
        return None
    try:
        g = float(gold) if "." in gold else int(gold)
    except ValueError:
        return None
    wrong = g + 1 if g != 0 else g - 1
    user = (
        f"A student works on this problem and concludes the answer is {wrong}.\n\n"
        f"{question}\n\n"
        "Identify the first likely mistake, then give one hint. "
        "Do not reveal the correct final answer."
    )
    asst = (
        f"First mistake: the student landed on {wrong}, which means an earlier arithmetic "
        f"or counting step is off. Hint: recompute the first intermediate carefully. "
        f"Do not state the correct final number."
    )
    return user, asst


def build_gsm8k(limit: int | None) -> list[dict]:
    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main", split="train", cache_dir=str(RAW))
    rows: list[dict] = []
    behaviors = ("solve", "explain", "hint", "first_error")
    for i, ex in enumerate(gsm):
        if limit is not None and len(rows) >= limit:
            break
        q, a = ex["question"], ex["answer"]
        behavior = behaviors[i % len(behaviors)]
        if behavior == "solve":
            user, asst = f"Solve the following problem step by step.\n\n{q}", a
        elif behavior == "explain":
            user, asst = f"Explain how to solve this problem clearly for a student.\n\n{q}", a
        elif behavior == "hint":
            user = (
                "A student is stuck on this problem. Give one helpful hint without "
                f"revealing the final numeric answer.\n\n{q}"
            )
            asst = hint_from_gsm8k(q, a)
        else:
            pair = first_error_from_gsm8k(q, a)
            if not pair:
                user, asst, behavior = f"Solve the following problem step by step.\n\n{q}", a, "solve"
            else:
                user, asst = pair
        rows.append(
            emit(f"en_en_{behavior}_gsm8k_v6_{i:05d}", behavior, user, asst, "gsm8k_train_v6")
        )
    return rows


def build_sciq(limit: int) -> list[dict]:
    from datasets import load_dataset

    sciq = load_dataset("allenai/sciq", split="train", cache_dir=str(RAW))
    rows: list[dict] = []
    for j, ex in enumerate(sciq):
        if len(rows) >= limit:
            break
        q = ex["question"]
        a = ex.get("correct_answer") or ""
        support = (ex.get("support") or "").strip()
        if not a:
            continue
        if j % 2 == 0:
            behavior = "solve"
            user = f"Answer the following science question.\n\n{q}"
            asst = a if not support else f"{a}\n\nExplanation: {support}"
        else:
            behavior = "explain"
            user = f"Answer and briefly explain for a student:\n\n{q}"
            asst = a if not support else f"{a}\n\nExplanation: {support}"
        rows.append(
            emit(f"en_en_{behavior}_sciq_v6_{j:05d}", behavior, user, asst, "sciq_train_v6")
        )
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gsm-limit", type=int, default=None, help="Cap GSM8K rows (default: all train)")
    ap.add_argument("--sciq-limit", type=int, default=3000)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--counts-out", type=Path, default=COUNTS_DEFAULT)
    ap.add_argument("--report-out", type=Path, default=REPORT_DEFAULT)
    ap.add_argument("--skip-dedup", action="store_true")
    args = ap.parse_args()

    print("[mix_v6] building GSM8K tutoring rows …")
    rows = build_gsm8k(args.gsm_limit)
    print(f"[mix_v6] gsm8k={len(rows)}")

    print(f"[mix_v6] building SciQ (limit={args.sciq_limit}) …")
    sciq_rows = build_sciq(args.sciq_limit)
    rows.extend(sciq_rows)
    print(f"[mix_v6] sciq={len(sciq_rows)} total_pre_dedup={len(rows)}")

    write_jsonl(args.out, rows)

    if not args.skip_dedup:
        eval_ok = [p for p in EVAL_PATHS if p.is_file()]
        if eval_ok:
            import sys

            sys.path.insert(0, str(ROOT))
            from eval.dedup_against_eval import extract_prompt, load_eval_hashes, text_hash

            banned = load_eval_hashes(eval_ok)
            kept: list[dict] = []
            dropped = 0
            for r in rows:
                if text_hash(extract_prompt(r)) in banned:
                    dropped += 1
                    continue
                kept.append(r)
            rows = kept
            write_jsonl(args.out, rows)
            print(f"[mix_v6] dedup dropped={dropped} kept={len(rows)}")
        else:
            print("[mix_v6] WARN no eval files for dedup; skipping")

    by_src = Counter(r["source"] for r in rows)
    by_beh = Counter(r["behavior"] for r in rows)
    by_dir = Counter(r["direction"] for r in rows)
    counts = {
        "n": len(rows),
        "by_source": dict(by_src),
        "by_behavior": dict(by_beh),
        "by_direction": dict(by_dir),
        "out": str(args.out),
    }
    args.counts_out.parent.mkdir(parents=True, exist_ok=True)
    args.counts_out.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")

    md = [
        "# SFT mix v6 (English-only)",
        "",
        f"**n={len(rows)}** — Qwen3-1.7B EN STEM tutoring track.",
        "",
        "## By source",
        "",
        "| source | n |",
        "|--------|--:|",
    ]
    for k, v in sorted(by_src.items()):
        md.append(f"| {k} | {v} |")
    md += [
        "",
        "## By behavior",
        "",
        "| behavior | n |",
        "|----------|--:|",
    ]
    for k, v in sorted(by_beh.items()):
        md.append(f"| {k} | {v} |")
    md += ["", f"Wrote `{args.out}`", ""]
    args.report_out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {args.out} ({len(rows)})")
    print(f"wrote {args.counts_out}")
    print(f"wrote {args.report_out}")


if __name__ == "__main__":
    main()
