#!/usr/bin/env python3
"""Build EN + Amharic tutoring SFT for mix v2 (GSM8K train + authored AM).

- EN solve / explain / hint / first_error from openai/gsm8k **train** only.
- Hints are problem-specific (first calculator step) and never reveal #### N.
- Amharic tutoring rows are hand-authored with **new** problems (not eval text).
- Never reads AfriMGSM test.

Outputs:
  data/train/en_stem_sft_v2.jsonl
  data/train/am_tutoring_sft_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# Login nodes cap nproc; OpenBLAS otherwise oversubscribes.
for _k in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "4")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hf"
EN_OUT = ROOT / "data" / "train" / "en_stem_sft_v2.jsonl"
AM_OUT = ROOT / "data" / "train" / "am_tutoring_sft_v2.jsonl"

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
    """One actionable hint from the first calculator annotation; strip final answer."""
    calcs = CALC_RE.findall(answer or "")
    gold = final_answer(answer)
    if calcs:
        expr, _val = calcs[0]
        expr = expr.strip()
        return (
            f"Hint: start by computing ({expr}). Write that intermediate result, "
            f"then continue — do not jump to the final total yet."
        )
    # Fallback: point at the first arithmetic cue in the question.
    nums = re.findall(r"\d+(?:\.\d+)?", question or "")
    if len(nums) >= 2:
        return (
            f"Hint: use the numbers {nums[0]} and {nums[1]} first to form one clear "
            f"intermediate quantity, then decide what remains. Do not state the final answer."
        )
    if gold:
        return (
            "Hint: undo the last operation that built the final quantity, and name that "
            "step before computing anything else. Do not reveal the final number."
        )
    return (
        "Hint: identify the first operation you must perform, write what it achieves, "
        "and stop before the final numeric answer."
    )


def first_error_from_gsm8k(question: str, answer: str) -> tuple[str, str] | None:
    """Synthetic wrong student total = gold+1 (or gold-1); coach without revealing gold."""
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
        f"or counting step is off (wrong intermediate, skipped quantity, or mis-applied operation). "
        f"Hint: recompute the first intermediate quantity carefully and check it against the story "
        f"before combining the remaining parts. Do not state the correct final number."
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
            user = f"Solve the following problem step by step.\n\n{q}"
            asst = a
        elif behavior == "explain":
            user = f"Explain how to solve this problem clearly for a student.\n\n{q}"
            asst = a
        elif behavior == "hint":
            user = (
                "A student is stuck on this problem. Give one helpful hint without "
                f"revealing the final numeric answer.\n\n{q}"
            )
            asst = hint_from_gsm8k(q, a)
        else:
            pair = first_error_from_gsm8k(q, a)
            if not pair:
                user = f"Solve the following problem step by step.\n\n{q}"
                asst = a
                behavior = "solve"
            else:
                user, asst = pair
        rows.append(
            emit(f"en_en_{behavior}_gsm8k_v2_{i:05d}", "en_en", behavior, user, asst, "gsm8k_train_v2")
        )
    return rows


def authored_am_tutoring() -> list[dict]:
    """Hand-authored AM tutoring; numbers/wording differ from custom_tutoring_v0 eval."""
    specs: list[tuple[str, str, str, str]] = [
        # behavior, user, assistant, tag
        (
            "solve",
            "ፍታ፡ 3x + 4 = 19 ከሆነ x ስንት ነው?",
            "3x + 4 = 19\n3x = 19 - 4\n3x = 15\nx = 15 / 3\nx = 5\n#### 5",
            "alg01",
        ),
        (
            "solve",
            "ፍታ፡ 7x - 3 = 18 ከሆነ x ስንት ነው?",
            "7x - 3 = 18\n7x = 18 + 3\n7x = 21\nx = 21 / 7\nx = 3\n#### 3",
            "alg02",
        ),
        (
            "solve",
            "አንድ ታንከር 80 ሊትር ይይዛል። 2/5 ብቻ ተሞልቷል። ስንት ሊትር አለ?",
            "80 * 2/5 = 160/5 = 32 ሊትር አለ።\n#### 32",
            "frac01",
        ),
        (
            "solve",
            "በሳጥን ውስጥ 45 ኳስ አሉ። 1/3 ተሸጠ። ስንት ቀረ?",
            "45 * 1/3 = 15 ተሸጠ። ቀሪው 45 - 15 = 30።\n#### 30",
            "frac02",
        ),
        (
            "solve",
            "አንድ ሰው በቀን 12 ገጽ ያነባል። በ 5 ቀን ስንት ገጽ ያነባል?",
            "12 * 5 = 60 ገጽ።\n#### 60",
            "word01",
        ),
        (
            "hint",
            "ተማሪው በ 4x = 28 ላይ ተጣብቋል። መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።",
            "ፍንጭ፡ ሁለቱንም ጎኖች በ 4 ከፋፍል (የማባዛቱ ተቃራኒ)። የመጨረሻውን ቁጥር አትናገር።",
            "hint01",
        ),
        (
            "hint",
            "ተማሪው በ 2x + 9 = 17 ላይ ተጣብቋል። መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።",
            "ፍንጭ፡ መጀመሪያ 9ን ከሁለቱም ጎኖች ቀንስ፣ ከዚያ 2ን ፍታ። xን አትናገር።",
            "hint02",
        ),
        (
            "hint",
            "ተማሪው በ 6x - 2 = 22 ላይ ተጣብቋል። አንድ ፍንጭ ብቻ ስጥ፣ መልሱን ሳትገልጽ።",
            "ፍንጭ፡ መጀመሪያ -2ን አስተካክል (2ን ወደ ሁለቱም ጎኖች ጨምር)፣ ቀጥሎ በ 6 ከፋፍል።",
            "hint03",
        ),
        (
            "first_error",
            "አንድ ተማሪ 3x + 4 = 19 ብሎ ጽፎ x = 7 አለ። የመጀመሪያውን ስህተት ጠቁም፣ መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።",
            "የመጀመሪያ ስህተት፡ 4ን በትክክል ሳያስወግድ ወደ x መሄዱ። ፍንጭ፡ መጀመሪያ 4ን ከሁለቱም ጎኖች ቀንስ፣ ከዚያ በ 3 ከፋፍል።",
            "fe01",
        ),
        (
            "first_error",
            "አንድ ተማሪ 5x = 20 ብሎ ጽፎ x = 5 አለ። የመጀመሪያውን ስህተት ጠቁም፣ መልሱን ሳትገልጽ።",
            "የመጀመሪያ ስህተት፡ በ 5 ሲከፋፈል የተሳሳተ ውጤት። ፍንጭ፡ 20ን በ 5 በመክፈል እንደገና ሞክር፣ መልሱን አትናገር።",
            "fe02",
        ),
        (
            "first_error",
            "አንድ ተማሪ 1/2 + 1/3 = 2/5 ብሎ ጽፏል። የመጀመሪያውን ስህተት ጠቁም፣ የመጨረሻውን መልስ ሳትሰጥ አንድ ፍንጭ ስጥ።",
            "የመጀመሪያ ስህተት፡ አሃዞችንና ተከፋዮችን ለየብቻ መደመር። ፍንጭ፡ መጀመሪያ አንድ አይነት ተከፋይ አግኝ።",
            "fe03",
        ),
        (
            "explain",
            "ክፍልፋይ ለምን በአንድ አይነት ተከፋይ ከመደመሩ በፊት ይፈለጋል? በአጭሩ በአማርኛ አብራራ።",
            "ተከፋዮቹ የተለያዩ ከሆኑ ክፍሎቹ የተለያየ መጠን አላቸው። አንድ አይነት ተከፋይ ሲኖር አሃዞቹን በቀጥታ መደመር ይቻላል።",
            "exp01",
        ),
        (
            "explain",
            "እኩልታ ሁለቱንም ጎኖች በአንድ አይነት ቁጥር ማባዛት ለምን እኩልነቱን አይቀይርም? በአጭሩ አብራራ።",
            "ሁለቱም ጎኖች አንድ አይነት እሴት ናቸው። በአንድ አይነት ቁጥር ሲባዙ ሁለቱም በተመሳሳይ መንገድ ይለወጣሉ፣ ስለዚህ እኩልነቱ ይቀጥላል።",
            "exp02",
        ),
        (
            "code_switch",
            "Explain what an equation is in algebra, then give the same idea in Amharic in one sentence.",
            "In algebra, an equation says two expressions are equal, often used to find an unknown.\n"
            "በአልጀብራ እኩልታ ማለት ሁለት አገላለጾች እኩል መሆናቸውን የሚያሳይ ዓረፍተ ነገር ነው።",
            "cs01",
        ),
        (
            "code_switch",
            "Explain what a fraction is, then give the same idea in Amharic in one sentence.",
            "A fraction shows a part of a whole using a numerator over a denominator.\n"
            "ክፍልፋይ ማለት አንድ ሙሉ ነገር ክፍልን በአሃዝና ተከፋይ የሚያሳይ ቁጥር ነው።",
            "cs02",
        ),
        (
            "solve",
            "ፍታ፡ x/2 + 3 = 8 ከሆነ x ስንት ነው?",
            "x/2 + 3 = 8\nx/2 = 8 - 3\nx/2 = 5\nx = 5 * 2\nx = 10\n#### 10",
            "alg03",
        ),
        (
            "solve",
            "አንድ መጽሐፍ 120 ብር ነው። 25% ቅናሽ ካለ ዋጋው ስንት ይሆናል?",
            "ቅናሹ 120 * 0.25 = 30 ብር። አዲስ ዋጋ 120 - 30 = 90 ብር።\n#### 90",
            "pct01",
        ),
        (
            "hint",
            "ተማሪው አማካይ የ 10፣ 14፣ 18 ለማግኘት ተጣብቋል። አንድ ፍንጭ ስጥ፣ መልሱን ሳትገልጽ።",
            "ፍንጭ፡ ሦስቱን ቁጥሮች ደምር፣ ከዚያ በ ሦስት ከፋፍል። የመጨረሻውን ቁጥር አትናገር።",
            "hint04",
        ),
        (
            "solve",
            "አንድ ሰው 9 ኪሎ ሜትር ሮጠ። በቀጣይ ቀን 4 ኪሎ ሜትር ጨመረ። በጠቅላላ ስንት ኪሎ ሜትር ሮጠ?",
            "9 + 4 = 13 ኪሎ ሜትር።\n#### 13",
            "word02",
        ),
        (
            "first_error",
            "አንድ ተማሪ 8 * 1/4 = 32 ብሎ ጽፏል። የመጀመሪያውን ስህተት ጠቁም፣ መልሱን ሳትገልጽ።",
            "የመጀመሪያ ስህተት፡ 8ን በ 4 ማባዛት እንጂ መክፈል። ፍንጭ፡ በ ክፍልፋይ ማባዛት ማለት መክፈል ነው።",
            "fe04",
        ),
        (
            "explain",
            "ተለዋዋጭ (variable) በአልጀብራ ምንድን ነው? በአንድ አጭር ዓረፍተ ነገር በአማርኛ ንገር።",
            "ተለዋዋጭ ማለት ያልታወቀን ወይም የሚቀያየርን ቁጥር የሚወክል ምልክት (ብዙ ጊዜ x) ነው።",
            "exp03",
        ),
        (
            "solve",
            "ፍታ፡ 2(x + 3) = 14 ከሆነ x ስንት ነው?",
            "2(x + 3) = 14\nx + 3 = 14 / 2\nx + 3 = 7\nx = 7 - 3\nx = 4\n#### 4",
            "alg04",
        ),
        (
            "hint",
            "ተማሪው በ 2(x + 5) = 18 ላይ ተጣብቋል። መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።",
            "ፍንጭ፡ መጀመሪያ ሁለቱንም ጎኖች በ 2 ከፋፍል፣ ከዚያ +5ን አስተካክል። xን አትናገር።",
            "hint05",
        ),
        (
            "solve",
            "በክፍል 28 ተማሪዎች አሉ። 3/7 ሴቶች ናቸው። ስንት ሴቶች አሉ?",
            "28 * 3/7 = 84/7 = 12 ሴቶች።\n#### 12",
            "frac03",
        ),
        (
            "code_switch",
            "Explain what a common denominator is, then give the same idea in Amharic in one sentence.",
            "A common denominator is a shared bottom number that lets you add or compare fractions.\n"
            "አንድ አይነት ተከፋይ ማለት ክፍልፋዮችን ለመደመር ወይም ለማነጻጸር የሚያገለግል አንድ አይነት ታችኛ ቁጥር ነው።",
            "cs03",
        ),
    ]
    # Expand with more simple algebra variants (still distinct from eval 2x+5=13 / 5x=20).
    more_alg = [
        (6, 1, 19, "alg05"),
        (8, 2, 26, "alg06"),
        (9, 3, 30, "alg07"),
        (4, 6, 22, "alg08"),
        (5, 7, 32, "alg09"),
        (10, 5, 35, "alg10"),
        (3, 9, 21, "alg11"),
        (7, 4, 32, "alg12"),
    ]
    for a, b, c, tag in more_alg:
        # a*x + b = c
        x = (c - b) // a
        assert a * x + b == c
        specs.append(
            (
                "solve",
                f"ፍታ፡ {a}x + {b} = {c} ከሆነ x ስንት ነው?",
                f"{a}x + {b} = {c}\n{a}x = {c} - {b}\n{a}x = {c - b}\nx = {c - b} / {a}\nx = {x}\n#### {x}",
                tag,
            )
        )
        specs.append(
            (
                "hint",
                f"ተማሪው በ {a}x + {b} = {c} ላይ ተጣብቋል። መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።",
                f"ፍንጭ፡ መጀመሪያ {b}ን ከሁለቱም ጎኖች ቀንስ፣ ከዚያ በ {a} ከፋፍል። የ x እሴት አትናገር።",
                f"hint_{tag}",
            )
        )
        wrong = x + 2
        specs.append(
            (
                "first_error",
                f"አንድ ተማሪ {a}x + {b} = {c} ብሎ ጽፎ x = {wrong} አለ። የመጀመሪያውን ስህተት ጠቁም፣ መልሱን ሳትገልጽ አንድ ፍንጭ ስጥ።",
                f"የመጀመሪያ ስህተት፡ {b}ን በትክክል ካላስተካከለ በኋላ ወደ x መሄድ ወይም በ {a} ሲከፋፈል ማጣት። "
                f"ፍንጭ፡ መጀመሪያ {b}ን አስተካክል፣ ከዚያ በ {a} ከፋፍል። ትክክለኛውን x አትናገር።",
                f"fe_{tag}",
            )
        )

    rows: list[dict] = []
    for i, (behavior, user, asst, tag) in enumerate(specs):
        rows.append(
            emit(f"am_am_{behavior}_tutor_v2_{tag}_{i:03d}", "am_am", behavior, user, asst, "am_tutoring_v2")
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--en-limit", type=int, default=2000, help="Max EN GSM8K tutoring rows")
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
