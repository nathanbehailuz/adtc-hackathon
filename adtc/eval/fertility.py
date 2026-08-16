#!/usr/bin/env python3
"""Tokenizer fertility diagnostics for Amharic vs English.

Computes approximate:
  F_am = tokens_am / words_am
  R_am/en = tokens_am / tokens_en on parallel sentences

Requires a HF tokenizer id (run after models are available on cloud):
  python fertility.py --tokenizer Qwen/Qwen3-1.7B
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARALLEL = ROOT / "data" / "eval" / "fertility_parallel_v0.jsonl"


def word_count(text: str) -> int:
    return max(1, len(text.split()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer / model id")
    parser.add_argument("--parallel", type=Path, default=PARALLEL)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "eval" / "fertility_report_v0.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rows = [json.loads(line) for line in args.parallel.read_text(encoding="utf-8").splitlines() if line.strip()]

    per = []
    sum_f_am = 0.0
    sum_r = 0.0
    for row in rows:
        en = row["en"]
        am = row["am"]
        t_en = len(tok.encode(en, add_special_tokens=False))
        t_am = len(tok.encode(am, add_special_tokens=False))
        f_am = t_am / word_count(am)
        r = t_am / max(1, t_en)
        per.append({"id": row.get("id"), "tokens_en": t_en, "tokens_am": t_am, "F_am": f_am, "R_am_en": r})
        sum_f_am += f_am
        sum_r += r

    n = max(1, len(per))
    report = {
        "tokenizer": args.tokenizer,
        "n_pairs": len(per),
        "mean_F_am": sum_f_am / n,
        "mean_R_am_en": sum_r / n,
        "pairs": per,
        "note": "High F_am or R_am/en suggests fragmentation; consider vocab work only if severe.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("tokenizer", "n_pairs", "mean_F_am", "mean_R_am_en")}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
