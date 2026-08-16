#!/usr/bin/env python3
"""Mix bilingual SFT pools and dedup against frozen eval.

Example:
  python mix_sft.py \\
    --sft data/train/sources/walia_sft_v0.jsonl data/train/sources/finetome_am_sft_v0.jsonl \\
    --en-stem data/train/en_stem_sft_v0.jsonl \\
    --eval data/eval/custom_tutoring_v0.jsonl data/eval/en_stem_holdout_v0.jsonl \\
    --out data/train/sft_mix_v0.jsonl

Run log: ``logs/mix_sft/<run>.*``
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.run_log import RunLogger  # noqa: E402

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
    parser.add_argument("--en-stem", type=Path, default=None)
    parser.add_argument("--am-stem", type=Path, default=None, help="Legacy single Amharic/MT file")
    parser.add_argument(
        "--sft",
        type=Path,
        nargs="*",
        default=[],
        help="Additional normalized SFT JSONL files (Walia, FineTome, …)",
    )
    parser.add_argument("--eval", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "sft_mix_v0.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--en-ratio", type=float, default=0.30)
    parser.add_argument("--am-ratio", type=float, default=0.55, help="Share for all Amharic/SFT pools combined")
    parser.add_argument("--replay-ratio", type=float, default=0.15)
    parser.add_argument("--total", type=int, default=2000)
    args = parser.parse_args()

    log = RunLogger(
        "mix_sft",
        meta={
            "out": str(args.out),
            "total": args.total,
            "seed": args.seed,
            "en_ratio": args.en_ratio,
            "am_ratio": args.am_ratio,
        },
    )
    try:
        rng = random.Random(args.seed)
        en_rows = read_jsonl(args.en_stem) if args.en_stem else []
        if args.en_stem:
            if en_rows:
                log.item_ok("en_stem", path=str(args.en_stem), n_rows=len(en_rows))
            else:
                log.item_error("en_stem", f"empty or missing: {args.en_stem}")

        am_rows: list[dict] = []
        if args.am_stem:
            rows = read_jsonl(args.am_stem)
            am_rows.extend(rows)
            if rows:
                log.item_ok("am_stem", path=str(args.am_stem), n_rows=len(rows))
            else:
                log.item_error("am_stem", f"empty or missing: {args.am_stem}")

        for p in args.sft or []:
            rows = read_jsonl(p)
            am_rows.extend(rows)
            key = p.stem
            if rows:
                log.item_ok(key, path=str(p), n_rows=len(rows))
            else:
                log.item_error(key, f"empty or missing: {p}")

        if not en_rows and not am_rows:
            log.item_error("mix", "No input rows: provide --en-stem and/or --sft / --am-stem")
            log.finish(status="error")
            raise SystemExit("No input rows: provide --en-stem and/or --sft / --am-stem")

        n_en = int(args.total * args.en_ratio) if en_rows else 0
        n_am = int(args.total * args.am_ratio) if am_rows else 0
        n_replay = max(0, args.total - n_en - n_am) if en_rows else 0
        if not en_rows:
            n_am = args.total

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
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        log.item_start("dedup", n_before=len(mixed), out=str(args.out))
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
        n_after = sum(1 for _ in args.out.open(encoding="utf-8")) if args.out.exists() else 0
        log.item_ok("dedup", n_before=len(mixed), n_after=n_after, path=str(args.out))
        log.item_ok("mix", n_before_dedup=len(mixed), n_after_dedup=n_after, path=str(args.out))
    except KeyboardInterrupt:
        log.warn("KeyboardInterrupt")
        log.finish(status="interrupted", message="interrupted by user")
        raise SystemExit(130) from None
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        log.item_error("mix", e)
        log.finish(status="error")
        raise

    summary = log.finish()
    print(f"mixed -> {args.out}")
    print(f"run log -> {log.log_path}")
    print(f"summary -> {summary}")


if __name__ == "__main__":
    main()
