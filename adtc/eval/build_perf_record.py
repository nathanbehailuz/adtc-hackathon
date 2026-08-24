#!/usr/bin/env python3
"""Merge profiler / frozen / translate artifacts into one perf/<key>_v0.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_perf import (  # noqa: E402
    empty_record,
    fill_adtc_shaped,
    merge_frozen,
    merge_profiler,
    merge_translate,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--profiler", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    ap.add_argument("--translate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rec = empty_record(args.model_key, str(args.gguf.resolve()))
    merge_profiler(rec, args.profiler.resolve())
    merge_frozen(rec, args.frozen.resolve())
    merge_translate(rec, args.translate.resolve())
    fill_adtc_shaped(rec)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(rec["adtc_shaped"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
