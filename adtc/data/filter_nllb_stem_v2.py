#!/usr/bin/env python3
"""Filter existing NLLB STEM JSONL for Ethiopic + math markers (no GPU).

Reads am_stem_sft_nllb_v1.jsonl (or any MT JSONL), keeps am_am / en_am rows that
look usable for mix v2. Also restores #### from paired English source when possible.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ETHIOPIC_RE = re.compile(r"[\u1200-\u137F]")
HASH_RE = re.compile(r"####\s*-?\d+(?:\.\d+)?")


def looks_am(text: str) -> bool:
    return bool(ETHIOPIC_RE.search(text or ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in",
        dest="inp",
        type=Path,
        default=ROOT / "data" / "train" / "am_stem_sft_nllb_v1.jsonl",
    )
    ap.add_argument(
        "--en",
        type=Path,
        default=ROOT / "data" / "train" / "en_stem_sft_v0.jsonl",
        help="Optional EN twin file to restore #### markers by shared id prefix",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "train" / "am_stem_sft_nllb_v2_filtered.jsonl",
    )
    ap.add_argument("--directions", default="am_am,en_am")
    args = ap.parse_args()

    en_hash: dict[str, str] = {}
    if args.en.exists():
        with args.en.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                asst = next(m["content"] for m in obj["messages"] if m["role"] == "assistant")
                m = HASH_RE.search(asst)
                if m:
                    en_hash[obj["id"]] = m.group(0)

    wanted = {d.strip() for d in args.directions.split(",") if d.strip()}
    n_in = n_out = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.inp.open(encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            obj = json.loads(line)
            n_in += 1
            direction = obj.get("direction", "")
            if direction not in wanted:
                continue
            user = next(m["content"] for m in obj["messages"] if m["role"] == "user")
            asst = next(m["content"] for m in obj["messages"] if m["role"] == "assistant")
            if direction == "am_am" and not looks_am(user):
                continue
            if direction.startswith("am") and not (looks_am(user) or looks_am(asst)):
                continue
            behavior = obj.get("behavior", "solve")
            if behavior in ("solve", "explain") and "####" not in asst:
                # id like en_en_solve_gsm8k_00000_am_am → base en_en_solve_gsm8k_00000
                base = obj["id"]
                for suf in ("_am_am", "_en_am", "_am_en"):
                    if base.endswith(suf):
                        base = base[: -len(suf)]
                        break
                marker = en_hash.get(base)
                if marker:
                    asst = asst.rstrip() + "\n" + marker
                    obj["messages"] = [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": asst},
                    ]
                elif behavior == "solve" and direction == "am_am":
                    continue
            obj["source"] = obj.get("source", "mt_nllb") + ":filtered_v2"
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"filtered {args.inp} -> {args.out} kept={n_out}/{n_in}")


if __name__ == "__main__":
    main()
