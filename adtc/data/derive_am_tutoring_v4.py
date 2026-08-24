#!/usr/bin/env python3
"""Derive Amharic hint / first_error / explain tutoring rows from AfriqueLLM solve JSONL.

Does not invent new word problems — only wraps existing am_am solve rows with tutoring stems.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ETHIOPIC_RE = re.compile(r"[\u1200-\u137F]")
FINAL_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def user_asst(row: dict) -> tuple[str, str]:
    msgs = row.get("messages") or []
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
    return user or "", asst or ""


def final_num(asst: str) -> str | None:
    m = FINAL_RE.search(asst or "")
    return m.group(1) if m else None


def emit(row_id: str, behavior: str, user: str, asst: str) -> dict:
    return {
        "id": row_id,
        "direction": "am_am",
        "behavior": behavior,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": asst},
        ],
        "source": "am_tutoring_derived_v4",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="inp",
        type=Path,
        default=ROOT / "data" / "train" / "sources" / "afriquellm_gsm8k_am_sft_v0.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "train" / "am_tutoring_derived_v4.jsonl",
    )
    ap.add_argument("--max-rows", type=int, default=1200, help="Max derived tutoring rows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.inp.exists():
        raise SystemExit(f"missing AM solve pool: {args.inp}")

    rng = random.Random(args.seed)
    solves = []
    for r in read_jsonl(args.inp):
        u, a = user_asst(r)
        if not ETHIOPIC_RE.search(u):
            continue
        if not FINAL_RE.search(a):
            continue
        solves.append((r.get("id", "row"), u, a))
    rng.shuffle(solves)

    out_rows: list[dict] = []
    behaviors = ("hint", "first_error", "explain")
    for i, (sid, u, a) in enumerate(solves):
        if len(out_rows) >= args.max_rows:
            break
        behavior = behaviors[i % len(behaviors)]
        gold = final_num(a)
        # Strip leading "Solve..." style English if any; keep problem body
        problem = u
        for prefix in (
            "Solve the following problem step by step.\n\n",
            "Explain how to solve this problem clearly for a student.\n\n",
        ):
            if problem.startswith(prefix):
                problem = problem[len(prefix) :]
        if behavior == "hint":
            user = (
                f"ተማሪው በዚህ ችግር ላይ ተጣብቋል። መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።\n\n{problem}"
            )
            asst = (
                "ፍንጭ፡ የመጀመሪያውን ስሌት / መካከለኛ መጠን በጥንቃቄ አስላ፣ ከዚያ ቀጥል። "
                "የመጨረሻውን ቁጥር አትናገር።"
            )
        elif behavior == "first_error":
            try:
                g = float(gold) if gold and "." in gold else int(gold)  # type: ignore[arg-type]
                wrong = g + 1 if g != 0 else g - 1
            except (TypeError, ValueError):
                wrong = "ስህተት"
            user = (
                f"አንድ ተማሪ ይህን ችግር ፈትቶ መልሱ {wrong} አለ። "
                f"የመጀመሪያውን ስህተት ጠቁም፣ ከዚያም መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።\n\n{problem}"
            )
            asst = (
                f"የመጀመሪያ ስህተት፡ ተማሪው በ {wrong} ላይ ደርሷል ማለት ቀደም ያለ ስሌት ተሳስቷል። "
                f"ፍንጭ፡ የመጀመሪያውን መካከለኛ ቁጥር እንደገና አስላ። ትክክለኛውን መልስ አትናገር። "
                f"ጥያቄውን አትድገም።"
            )
        else:
            user = f"ይህን ችግር ለተማሪ በአጭሩ በአማርኛ አብራራ (መልሱን ጨምር)።\n\n{problem}"
            asst = a
        out_rows.append(emit(f"derived_v4_{behavior}_{i:05d}_{sid}", behavior, user, asst))

    write_jsonl(args.out, out_rows)
    print(f"wrote {args.out} ({len(out_rows)}) from {len(solves)} solve candidates")


if __name__ == "__main__":
    main()
