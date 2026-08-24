#!/usr/bin/env python3
"""Build AM STEM solve pool for mix v4 when AfriqueLLM Hub is unavailable.

Priority:
  1. afriquellm_gsm8k_am_sft_v0.jsonl (if present)
  2. else NLLB filtered am_am solve (good Ethiopic prompts) + capped simonbutt fill

Writes data/train/am_stem_pool_v4.jsonl and prints JSON stats to stdout.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ETH = re.compile(r"[\u1200-\u137F]")
FINAL = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
BAD_USER_MARKERS = ("በዚህ ርዕስ",)


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


def good_am_solve(row: dict) -> bool:
    if row.get("direction") != "am_am" or row.get("behavior") != "solve":
        return False
    msgs = row.get("messages") or []
    if len(msgs) < 2:
        return False
    user = msgs[0].get("content") or ""
    asst = msgs[1].get("content") or ""
    if not ETH.search(user) or not FINAL.search(asst):
        return False
    return not any(m in user for m in BAD_USER_MARKERS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--afri",
        type=Path,
        default=ROOT / "data/train/sources/afriquellm_gsm8k_am_sft_v0.jsonl",
    )
    ap.add_argument(
        "--nllb",
        type=Path,
        default=ROOT / "data/train/am_stem_sft_nllb_v2_filtered.jsonl",
    )
    ap.add_argument(
        "--simonbutt",
        type=Path,
        default=ROOT / "data/train/sources/simonbutt_amharic_gsm8k_sft_v0.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/train/am_stem_pool_v4.jsonl",
    )
    ap.add_argument("--target", type=int, default=3200, help="Rows to aim for in pool")
    ap.add_argument("--simonbutt-cap", type=int, default=2500, help="Max simonbutt rows in fallback")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    stats: dict = {"mode": None, "sources": {}}

    if args.afri.exists():
        afri = read_jsonl(args.afri)
        stats["sources"]["afriquellm"] = len(afri)
        if len(afri) >= 1500:
            write_jsonl(args.out, afri)
            stats["mode"] = "afriquellm"
            stats["n_out"] = len(afri)
            print(json.dumps(stats, indent=2))
            return

    nllb_rows = [r for r in read_jsonl(args.nllb) if good_am_solve(r)] if args.nllb.exists() else []
    simon_rows = read_jsonl(args.simonbutt) if args.simonbutt.exists() else []
    stats["sources"]["nllb_good"] = len(nllb_rows)
    stats["sources"]["simonbutt_total"] = len(simon_rows)

    if len(nllb_rows) + len(simon_rows) < 1500:
        print(
            f"error: fallback AM stem pool too small ({len(nllb_rows)} nllb + {len(simon_rows)} simonbutt)",
            file=sys.stderr,
        )
        raise SystemExit(1)

    rng = random.Random(args.seed)
    need = max(0, args.target - len(nllb_rows))
    cap = min(args.simonbutt_cap, need, len(simon_rows))
    rng.shuffle(simon_rows)
    pool = nllb_rows + simon_rows[:cap]
    rng.shuffle(pool)
    write_jsonl(args.out, pool)
    stats["mode"] = "nllb+simonbutt_capped"
    stats["n_out"] = len(pool)
    stats["simonbutt_used"] = cap
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
