#!/usr/bin/env python3
"""Remove train JSONL rows that collide with frozen eval prompts (exact normalized match)."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def norm(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def text_hash(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8")).hexdigest()


def extract_prompt(obj: dict) -> str:
    if "question" in obj and obj["question"]:
        return str(obj["question"])
    if "messages" in obj and obj["messages"]:
        for m in obj["messages"]:
            if m.get("role") == "user":
                return str(m.get("content", ""))
    if "example" in obj and isinstance(obj["example"], dict):
        for k in ("question", "premise", "prompt", "text"):
            if obj["example"].get(k):
                return str(obj["example"][k])
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def load_eval_hashes(eval_paths: list[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in eval_paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                hashes.add(text_hash(extract_prompt(obj)))
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    banned = load_eval_hashes(args.eval)
    kept = 0
    dropped = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.train.open(encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            obj = json.loads(line)
            h = text_hash(extract_prompt(obj))
            if h in banned:
                dropped += 1
                continue
            dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
            kept += 1
    print(f"kept={kept} dropped={dropped} -> {args.out}")


if __name__ == "__main__":
    main()
