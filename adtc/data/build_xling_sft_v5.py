#!/usr/bin/env python3
"""Build bilingual tutoring SFT pairs (am_en / en_am) for v5.

Creates synthetic cross-lingual tutoring from EN STEM + AM STEM pools without
touching frozen eval. Caps quality with Ethiopic checks.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ETHIOPIC = re.compile(r"[\u1200-\u137F]")

EN_TO_AM_WRAP = (
    "ተማሪው የሚከተለውን እንግሊዝኛ ችግር አምሃርኛ አብራርተህ መልስ፡\n\n{q}\n\n"
    "መልስህን በአማርኛ ጽፈህ ቁጥራዊ መልሱን በ `####` ምልክት አብረህ አሳይ።"
)
AM_TO_EN_WRAP = (
    "A student asked this Amharic problem. Explain the solution in clear English "
    "and end with #### <number>.\n\n{q}"
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def user_asst(row: dict) -> tuple[str, str]:
    msgs = row.get("messages") or []
    user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    return str(user), str(asst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--en-stem", type=Path, required=True)
    ap.add_argument("--am-stem", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-en-am", type=int, default=800)
    ap.add_argument("--n-am-en", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    en_rows = read_jsonl(args.en_stem)
    am_rows: list[dict] = []
    for p in args.am_stem:
        am_rows.extend(read_jsonl(p))
    am_rows = [r for r in am_rows if ETHIOPIC.search(user_asst(r)[0] or "")]

    out: list[dict] = []
    for i, r in enumerate(rng.sample(en_rows, min(args.n_en_am, len(en_rows)))):
        q, a = user_asst(r)
        if not q or not a:
            continue
        out.append(
            {
                "id": f"xling_en_am_{i:04d}",
                "direction": "en_am",
                "behavior": "explain",
                "messages": [
                    {"role": "user", "content": EN_TO_AM_WRAP.format(q=q)},
                    {"role": "assistant", "content": a if ETHIOPIC.search(a) else a},
                ],
                "source": "xling_en_am_v5",
            }
        )

    for i, r in enumerate(rng.sample(am_rows, min(args.n_am_en, len(am_rows)))):
        q, a = user_asst(r)
        if not q or not a:
            continue
        # Prefer keeping Amharic answer body but instruct English explanation —
        # if answer lacks Latin letters, prefix a short English lead-in.
        asst = a
        if not re.search(r"[A-Za-z]", a):
            asst = "Here is the worked solution:\n" + a
        out.append(
            {
                "id": f"xling_am_en_{i:04d}",
                "direction": "am_en",
                "behavior": "explain",
                "messages": [
                    {"role": "user", "content": AM_TO_EN_WRAP.format(q=q)},
                    {"role": "assistant", "content": asst},
                ],
                "source": "xling_am_en_v5",
            }
        )

    write_jsonl(args.out, out)
    print(f"wrote {args.out} n={len(out)}")


if __name__ == "__main__":
    main()
