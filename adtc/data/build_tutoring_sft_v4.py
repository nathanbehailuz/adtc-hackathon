#!/usr/bin/env python3
"""Build EN + Amharic tutoring SFT for mix v4 (GSM8K train + expanded authored AM).

Outputs:
  data/train/en_stem_sft_v4.jsonl
  data/train/am_tutoring_sft_v4.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
EN_OUT = ROOT / "data" / "train" / "en_stem_sft_v4.jsonl"
AM_OUT = ROOT / "data" / "train" / "am_tutoring_sft_v4.jsonl"

FINAL_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
CALC_RE = re.compile(r"<<([^>=]+)=([^>]+)>>")


def emit(row_id: str, direction: str, behavior: str, user: str, assistant: str, source: str) -> dict:
    return {
        "id": row_id,
        "direction": direction,
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_en_stem(limit: int) -> list[dict]:
    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main", split="train", cache_dir=str(RAW))
    rows: list[dict] = []
    behaviors = ("solve", "explain", "hint", "first_error")
    for i, ex in enumerate(gsm):
        if len(rows) >= limit:
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
            emit(f"en_en_{behavior}_gsm8k_v4_{i:05d}", "en_en", behavior, user, asst, "gsm8k_train_v4")
        )
    return rows


def authored_am_tutoring() -> list[dict]:
    specs: list[tuple[str, str, str, str]] = [
        ("solve", "ፍታ፡ 3x + 4 = 19 ከሆነ x ስንት ነው?",
         "3x + 4 = 19\n3x = 19 - 4\n3x = 15\nx = 15 / 3\nx = 5\n#### 5", "alg01"),
        ("solve", "ፍታ፡ 7x - 3 = 18 ከሆነ x ስንት ነው?",
         "7x - 3 = 18\n7x = 18 + 3\n7x = 21\nx = 21 / 7\nx = 3\n#### 3", "alg02"),
        ("solve", "አንድ ታንከር 80 ሊትር ይይዛል። 2/5 ብቻ ተሞልቷል። ስንት ሊትር አለ?",
         "80 * 2/5 = 160/5 = 32 ሊትር አለ።\n#### 32", "frac01"),
        ("hint", "ተማሪው በ 4x = 28 ላይ ተጣብቋል። መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።",
         "ፍንጭ፡ ሁለቱንም ጎኖች በ 4 ከፋፍል። የመጨረሻውን ቁጥር አትናገር።", "hint01"),
        ("first_error",
         "አንድ ተማሪ 3x + 4 = 19 ብሎ ጽፎ x = 7 አለ። የመጀመሪያውን ስህተት ጠቁም፣ ከዚያም መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።",
         "የመጀመሪያ ስህተት፡ 4ን በትክክል ሳያስወግድ ወደ x መሄዱ። ፍንጭ፡ መጀመሪያ 4ን ከሁለቱም ጎኖች ቀንስ፣ ከዚያ በ 3 ከፋፍል። ጥያቄውን አትድገም።",
         "fe01"),
        ("first_error",
         "አንድ ተማሪ 8x - 2 = 30 ብሎ ጽፎ x = 5 አለ። የመጀመሪያውን ስህተት ጠቁም፣ ከዚያም መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።",
         "የመጀመሪያ ስህተት፡ -2ን ሳያስተካክል ወይም በ 8 ሲከፋፈል ማጣት። ፍንጭ፡ 2ን ወደ ሁለቱም ጎኖች ጨምር፣ ከዚያ በ 8 ከፋፍል።",
         "fe_try8"),
        ("explain", "ተለዋዋጭ (variable) በአልጀብራ ምንድን ነው? በአንድ አጭር ዓረፍተ ነገር በአማርኛ ንገር።",
         "ተለዋዋጭ ማለት ያልታወቀን ወይም የሚቀያየርን ቁጥር የሚወክል ምልክት (ብዙ ጊዜ x) ነው።", "exp_var"),
        ("code_switch",
         "Explain what a variable is in algebra, then give the same idea in Amharic in one sentence.",
         "A variable is a symbol (often a letter like x) that stands for an unknown or changing number.\n"
         "ተለዋዋጭ ማለት ያልታወቀን ቁጥር የሚወክል ምልክት ነው።", "cs_var"),
        ("code_switch",
         "Explain what an equation is in algebra, then give the same idea in Amharic in one sentence.",
         "In algebra, an equation says two expressions are equal, often used to find an unknown.\n"
         "በአልጀብራ እኩልታ ማለት ሁለት አገላለጾች እኩል መሆናቸውን የሚያሳይ ዓረፍተ ነገር ነው።", "cs01"),
        ("code_switch",
         "Explain what a fraction is, then give the same idea in Amharic in one sentence.",
         "A fraction shows a part of a whole using a numerator over a denominator.\n"
         "ክፍልፋይ ማለት አንድ ሙሉ ነገር ክፍልን በአሃዝና ተከፋይ የሚያሳይ ቁጥር ነው።", "cs02"),
        ("code_switch",
         "Explain what a common denominator is, then give the same idea in Amharic in one sentence.",
         "A common denominator is a shared bottom number that lets you add or compare fractions.\n"
         "አንድ አይነት ተከፋይ ማለት ክፍልፋዮችን ለመደመር የሚያገለግል አንድ አይነት ታችኛ ቁጥር ነው።", "cs03"),
        ("code_switch",
         "Explain what a ratio is, then give the same idea in Amharic in one sentence.",
         "A ratio compares two quantities by division.\nሬሾ ማለት ሁለት መጠኖችን በማካፈል የሚያነጻጽር ግንኙነት ነው።", "cs04"),
        ("code_switch",
         "Explain what the distributive property is, then give the same idea in Amharic in one sentence.",
         "The distributive property says a(b + c) = ab + ac.\n"
         "የተሰራጨ ባህሪ ማለት a(b + c) = ab + ac ማለት ነው።", "cs05"),
        ("code_switch",
         "Explain what an average (mean) is, then give the same idea in Amharic in one sentence.",
         "The mean is the sum of values divided by how many values there are.\n"
         "አማካይ ማለት የቁጥሮች ድምር በቁጥራቸው መከፋፈል ነው።", "cs06"),
        ("code_switch",
         "Explain what a percentage is, then give the same idea in Amharic in one sentence.",
         "A percentage is a number out of one hundred, written with %.\n"
         "መቶኛ ማለት ከመቶ ውስጥ ያለ ክፍል፣ በ % የሚጻፍ ነው።", "cs07"),
        ("code_switch",
         "Explain what a linear equation is, then give the same idea in Amharic in one sentence.",
         "A linear equation has variables only to the first power, like ax + b = c.\n"
         "መስመራዊ እኩልታ ማለት ተለዋዋጮች በአንደኛ ኃይል ብቻ ያሉበት እኩልታ ነው።", "cs08"),
    ]

    more_alg = [
        (6, 1, 19), (8, 2, 26), (9, 3, 30), (4, 6, 22), (5, 7, 32),
        (10, 5, 35), (3, 9, 21), (7, 4, 32), (11, 2, 35), (12, 3, 39),
        (6, 5, 29), (8, 4, 28), (9, 6, 33), (4, 8, 24), (5, 9, 34),
        (7, 7, 35), (3, 6, 18), (10, 8, 48), (11, 5, 38), (12, 6, 42),
        (6, 8, 26), (8, 6, 30), (9, 1, 28), (4, 3, 19), (5, 4, 29),
        (7, 8, 36), (3, 5, 17), (10, 2, 42), (11, 8, 41), (12, 0, 36),
        (6, 2, 20), (8, 1, 25), (9, 4, 31), (4, 2, 18), (5, 1, 21),
        (7, 2, 23), (3, 2, 14), (10, 4, 34), (11, 1, 34), (12, 4, 40),
    ]
    for i, (a, b, c) in enumerate(more_alg):
        x = (c - b) // a
        if a * x + b != c:
            continue
        tag = f"alg{i:03d}"
        specs.append(
            ("solve", f"ፍታ፡ {a}x + {b} = {c} ከሆነ x ስንት ነው?",
             f"{a}x + {b} = {c}\n{a}x = {c} - {b}\n{a}x = {c - b}\nx = {c - b} / {a}\nx = {x}\n#### {x}", tag)
        )
        specs.append(
            ("hint", f"ተማሪው በ {a}x + {b} = {c} ላይ ተጣብቋል። መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።",
             f"ፍንጭ፡ መጀመሪያ {b}ን ከሁለቱም ጎኖች ቀንስ፣ ከዚያ በ {a} ከፋፍል። የ x እሴት አትናገር።", f"hint_{tag}")
        )
        wrong = x + 2
        specs.append(
            ("first_error",
             f"አንድ ተማሪ {a}x + {b} = {c} ብሎ ጽፎ x = {wrong} አለ። የመጀመሪያውን ስህተት ጠቁም፣ ከዚያም መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።",
             f"የመጀመሪያ ስህተት፡ {b}ን ካላስተካከለ በኋላ ወደ x መሄድ ወይም በ {a} ሲከፋፈል ማጣት። "
             f"ፍንጭ፡ መጀመሪያ {b}ን አስተካክል፣ ከዚያ በ {a} ከፋፍል። ትክክለኛውን x አትናገር። ጥያቄውን አትድገም።",
             f"fe_{tag}")
        )

    specs.extend([
        ("solve", "አንድ ሸቀጥ 200 ብር ነው። 10% ቅናሽ ካለ አዲስ ዋጋ ስንት ነው?",
         "ቅናሹ 200 * 0.10 = 20 ብር። አዲስ ዋጋ 200 - 20 = 180 ብር።\n#### 180", "pct02"),
        ("solve", "በክፍል 36 ተማሪዎች አሉ። 1/4 ወንዶች ናቸው። ስንት ወንዶች አሉ?",
         "36 * 1/4 = 9 ወንዶች።\n#### 9", "frac04"),
        ("first_error", "አንድ ተማሪ 1/2 + 1/4 = 2/6 ብሎ ጽፏል። የመጀመሪያውን ስህተት ጠቁም፣ መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።",
         "የመጀመሪያ ስህተት፡ አሃዞችንና ተከፋዮችን ለየብቻ መደመር። ፍንጭ፡ መጀመሪያ አንድ አይነት ተከፋይ (4) አግኝ።", "fe_frac"),
        ("explain", "ክፍልፋይ ለምን በአንድ አይነት ተከፋይ ከመደመሩ በፊት ይፈለጋል? በአጭሩ በአማርኛ አብራራ።",
         "ተከፋዮቹ የተለያዩ ከሆኑ ክፍሎቹ የተለያየ መጠን አላቸው። አንድ አይነት ተከፋይ ሲኖር አሃዞቹን በቀጥታ መደመር ይቻላል።", "exp_frac"),
    ])

    rows: list[dict] = []
    for i, (behavior, user, asst, tag) in enumerate(specs):
        direction = "en_am" if behavior == "code_switch" else "am_am"
        rows.append(
            emit(f"am_{behavior}_tutor_v4_{tag}_{i:03d}", direction, behavior, user, asst, "am_tutoring_v4")
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--en-limit", type=int, default=2000)
    ap.add_argument("--en-out", type=Path, default=EN_OUT)
    ap.add_argument("--am-out", type=Path, default=AM_OUT)
    args = ap.parse_args()
    en_rows = build_en_stem(args.en_limit)
    am_rows = authored_am_tutoring()
    write_jsonl(args.en_out, en_rows)
    write_jsonl(args.am_out, am_rows)
    print(f"wrote {args.en_out} ({len(en_rows)})")
    print(f"wrote {args.am_out} ({len(am_rows)})")


if __name__ == "__main__":
    main()
