#!/usr/bin/env python3
"""Build English-only SFT mix v7: cleaned GSM8K + SciQ + authored tutoring.

Key v7 changes vs v6:
  - Strip GSM8K #### and <<expr=val>> markup from all assistant targets
  - Richer hint / first_error / explain templates (why + remediation + check Q)
  - Mix in authored multi-constraint tutoring (judge-aligned patterns)

Outputs:
  data/train/sft_mix_v7.jsonl
  docs/artifacts/v7/sft_mix_v7_counts.json
  docs/artifacts/v7/sft_mix_v7_report.md
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
OUT_DEFAULT = ROOT / "data" / "train" / "sft_mix_v7.jsonl"
COUNTS_DEFAULT = ROOT / "docs" / "artifacts" / "v7" / "sft_mix_v7_counts.json"
REPORT_DEFAULT = ROOT / "docs" / "artifacts" / "v7" / "sft_mix_v7_report.md"
AUTHORED_DEFAULT = ROOT / "data" / "authored_tutoring_v7.jsonl"

FINAL_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
CALC_RE = re.compile(r"<<([^>=]+)=([^>]+)>>")
MARKUP_BAN_RE = re.compile(r"(####|<<[^>]*>>)")

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


def clean_gsm8k_solution(answer: str, *, include_final: bool = True) -> str:
    """Turn GSM8K chain-of-thought into plain tutor prose without #### / <<>>."""
    text = answer or ""
    gold = final_answer(text)

    # Typical: "16 - 3 - 4 = <<16-3-4=9>>9" → "16 - 3 - 4 = 9"
    # Also: "9 * 2 = $<<9*2=18>>18" → "9 * 2 = $18"
    def _repl_eq(m: re.Match[str]) -> str:
        prefix = m.group(1) or ""
        val = m.group(3).strip()
        trail = m.group(4)
        if trail is not None and trail.strip() == val:
            return f"= {prefix}{val}"
        if trail is not None:
            return f"= {prefix}{val} ({trail.strip()})"
        return f"= {prefix}{val}"

    text = re.sub(
        r"=\s*(\$)?<<([^>=]+)=([^>]+)>>(\s*-?\d+(?:\.\d+)?)?",
        _repl_eq,
        text,
    )
    # Any remaining <<expr=val>> → expr = val
    text = CALC_RE.sub(
        lambda m: f"{m.group(1).strip()} = {m.group(2).strip()}",
        text,
    )
    text = FINAL_RE.sub("", text)
    text = re.sub(r"=(\S)", r"= \1", text)  # "3+2=5" → "3+2= 5" then normalize
    text = re.sub(r"=\s+", "= ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    numbered: list[str] = []
    for i, ln in enumerate(lines, 1):
        if re.match(r"^\d+[\.\)]\s", ln):
            numbered.append(ln)
        else:
            numbered.append(f"{i}. {ln}")
    body = "\n".join(numbered)
    if include_final and gold is not None:
        body = f"{body}\n\nFinal answer: {gold}"
    return body


def assert_no_markup(rows: list[dict]) -> None:
    bad = []
    for r in rows:
        asst = r["messages"][1]["content"]
        if MARKUP_BAN_RE.search(asst):
            bad.append(r["id"])
    if bad:
        raise SystemExit(
            f"markup leak in {len(bad)} rows (showing up to 5): {bad[:5]}"
        )


def hint_from_gsm8k(question: str, answer: str) -> str:
    calcs = CALC_RE.findall(answer or "")
    if calcs:
        expr, val = calcs[0]
        return (
            f"Hint: start by computing {expr.strip()}. "
            f"Why: that isolates the first quantity the problem depends on "
            f"(an intermediate result near {val.strip()}) before later steps. "
            f"Check question: after you write that intermediate, what operation "
            f"should come next? Do not state the final answer."
        )
    nums = re.findall(r"\d+(?:\.\d+)?", question or "")
    if len(nums) >= 2:
        return (
            f"Hint: use the numbers {nums[0]} and {nums[1]} first to form one clear "
            f"intermediate quantity. Why: multi-step word problems usually need a "
            f"defined part before the remainder or total. "
            f"Check question: what does that intermediate represent in the story? "
            f"Do not state the final answer."
        )
    return (
        "Hint: identify the first operation you must perform and write what it "
        "achieves. Why: naming the first undo/operation prevents jumping to a "
        "guessed total. Check question: after that step, which quantity is still "
        "unknown? Do not state the final numeric answer."
    )


def first_error_from_gsm8k(question: str, answer: str) -> tuple[str, str] | None:
    gold = final_answer(answer)
    calcs = CALC_RE.findall(answer or "")
    if gold is None:
        return None
    try:
        g = float(gold) if "." in gold else int(gold)
    except ValueError:
        return None
    wrong = g + 1 if g != 0 else g - 1
    if calcs:
        expr, val = calcs[0]
        diagnosis = (
            f"The student likely mishandled the first intermediate "
            f"({expr.strip()} should be {val.strip()}) or carried a wrong value "
            f"forward, which is why they landed on {wrong}."
        )
        hint = (
            f"Recompute {expr.strip()} carefully, write that result, then redo "
            f"only the next step from that correct intermediate."
        )
    else:
        diagnosis = (
            f"Landing on {wrong} instead of the correct total usually means an "
            f"early arithmetic or counting step was off, not just the last line."
        )
        hint = (
            "Recompute the first intermediate from the given numbers, then continue "
            "from that corrected value."
        )
    user = (
        f"A student works on this problem and concludes the answer is {wrong}.\n\n"
        f"{question}\n\n"
        "Identify the exact reasoning or arithmetic mistake, explain why that step "
        "is wrong, give one corrective hint, and ask one short follow-up question. "
        "Do not reveal the correct final answer."
    )
    asst = (
        f"Exact mistake: {diagnosis}\n"
        f"Why it is wrong: a wrong early quantity makes later steps look tidy but "
        f"still produce an incorrect final number.\n"
        f"Corrective hint: {hint}\n"
        f"Follow-up: Which quantity did you compute first, and does it match the "
        f"units or part the problem asks for?\n"
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
            user = f"Solve the following problem step by step.\n\n{q}"
            asst = clean_gsm8k_solution(a, include_final=True)
        elif behavior == "explain":
            user = (
                "Explain how to solve this problem clearly for a secondary-school "
                f"student. Justify each step.\n\n{q}"
            )
            asst = (
                "Here is a student-friendly walkthrough:\n\n"
                + clean_gsm8k_solution(a, include_final=True)
            )
        elif behavior == "hint":
            user = (
                "A student is stuck on this problem. Give one helpful hint: name the "
                "next step, explain why it is valid, and ask one check question. "
                f"Do not reveal the final numeric answer.\n\n{q}"
            )
            asst = hint_from_gsm8k(q, a)
        else:
            pair = first_error_from_gsm8k(q, a)
            if not pair:
                user = f"Solve the following problem step by step.\n\n{q}"
                asst = clean_gsm8k_solution(a, include_final=True)
                behavior = "solve"
            else:
                user, asst = pair
        rows.append(
            emit(f"en_en_{behavior}_gsm8k_v7_{i:05d}", behavior, user, asst, "gsm8k_train_v7")
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
        a = (ex.get("correct_answer") or "").strip()
        support = (ex.get("support") or "").strip()
        if not a:
            continue
        if j % 3 == 0:
            behavior = "solve"
            user = f"Answer the following science question clearly.\n\n{q}"
            asst = (
                f"{a}\n\nExplanation: {support}"
                if support
                else f"{a}\n\nExplanation: State the key idea in one short sentence for a student."
            )
        elif j % 3 == 1:
            behavior = "explain"
            user = (
                "Answer and explain for a 15-year-old student. Define key terms, "
                "correct one common misconception if relevant, and end with one "
                f"check question.\n\n{q}"
            )
            body = support if support else f"The answer is {a}."
            asst = (
                f"Answer: {a}\n\n"
                f"Explanation: {body}\n\n"
                f"Common misconception: memorizing the label without the mechanism — "
                f"focus on cause and effect, not just the name.\n\n"
                f"Check question: In your own words, what evidence or definition "
                f"supports this answer?"
            )
        else:
            behavior = "hint"
            user = (
                "A student is stuck on this science question. Give one hint with a "
                f"brief why, and a check question. Do not reveal the final answer.\n\n{q}"
            )
            asst = (
                "Hint: re-read the question and underline the process or quantity "
                "being asked for (cause, definition, or comparison). Why: science "
                "items often hinge on one precise term rather than a long paragraph. "
                "Check question: are you being asked for a name, a cause, or a "
                "comparison? Do not state the final answer."
            )
        rows.append(
            emit(f"en_en_{behavior}_sciq_v7_{j:05d}", behavior, user, asst, "sciq_train_v7")
        )
    return rows


def load_authored(path: Path) -> list[dict]:
    if not path.is_file():
        print(f"[mix_v7] WARN authored missing: {path}")
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gsm-limit", type=int, default=None)
    ap.add_argument("--sciq-limit", type=int, default=3000)
    ap.add_argument("--authored", type=Path, default=AUTHORED_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--counts-out", type=Path, default=COUNTS_DEFAULT)
    ap.add_argument("--report-out", type=Path, default=REPORT_DEFAULT)
    ap.add_argument("--skip-dedup", action="store_true")
    args = ap.parse_args()

    print("[mix_v7] building GSM8K tutoring rows …")
    rows = build_gsm8k(args.gsm_limit)
    print(f"[mix_v7] gsm8k={len(rows)}")

    print(f"[mix_v7] building SciQ (limit={args.sciq_limit}) …")
    sciq_rows = build_sciq(args.sciq_limit)
    rows.extend(sciq_rows)
    print(f"[mix_v7] sciq={len(sciq_rows)}")

    authored = load_authored(args.authored)
    rows.extend(authored)
    print(f"[mix_v7] authored={len(authored)} total_pre_dedup={len(rows)}")

    assert_no_markup(rows)

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
            print(f"[mix_v7] dedup dropped={dropped} kept={len(rows)}")
        else:
            print("[mix_v7] WARN no eval files for dedup; skipping")

    assert_no_markup(rows)
    write_jsonl(args.out, rows)

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
        "# SFT mix v7 (English-only)",
        "",
        f"**n={len(rows)}** — cleaned GSM8K markup + richer tutoring + authored bank.",
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
