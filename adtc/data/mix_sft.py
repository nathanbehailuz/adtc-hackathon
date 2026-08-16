#!/usr/bin/env python3
"""Mix bilingual SFT pools and dedup against frozen eval.

Example:
  python mix_sft.py \\
    --en-stem data/train/en_stem_sft_v0.jsonl \\
    --am-stem data/train/am_stem_sft_v0.jsonl \\
    --eval data/eval/custom_tutoring_v0.jsonl data/eval/en_stem_holdout_v0.jsonl \\
    --out data/train/sft_mix_v0.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEDUP = ROOT / "eval" / "dedup_against_eval.py"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def sample_pool(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    if k <= 0 or not rows:
        return []
    if k >= len(rows):
        return list(rows)
    return rng.sample(rows, k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--en-stem", type=Path, required=True)
    parser.add_argument("--am-stem", type=Path, default=None)
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "sft_mix_v0.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--en-ratio", type=float, default=0.35, help="Fraction of final mix from EN STEM")
    parser.add_argument("--am-ratio", type=float, default=0.45, help="Fraction from translated/am pools")
    parser.add_argument("--replay-ratio", type=float, default=0.20, help="Extra EN replay from en-stem")
    parser.add_argument("--total", type=int, default=2000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    en_rows = read_jsonl(args.en_stem)
    am_rows = read_jsonl(args.am_stem) if args.am_stem else []

    n_en = int(args.total * args.en_ratio)
    n_am = int(args.total * args.am_ratio)
    n_replay = max(0, args.total - n_en - n_am)

    mixed = (
        sample_pool(en_rows, n_en, rng)
        + sample_pool(am_rows, n_am, rng)
        + sample_pool(en_rows, n_replay, rng)
    )
    rng.shuffle(mixed)

    tmp = args.out.with_suffix(".pre_dedup.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        for row in mixed:
            # mark replay copies
            if row in en_rows and mixed.count(row) > 1:
                row = dict(row)
                row["source"] = row.get("source", "") + "+replay"
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cmd = [
        sys.executable,
        str(DEDUP),
        "--train",
        str(tmp),
        "--eval",
        *[str(p) for p in args.eval],
        "--out",
        str(args.out),
    ]
    subprocess.check_call(cmd)
    tmp.unlink(missing_ok=True)
    print(f"mixed -> {args.out}")


if __name__ == "__main__":
    main()
