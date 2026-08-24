#!/usr/bin/env python3
"""Assemble sft_mix_v2 with explicit STEM / tutoring / instruct buckets.

Target shares (default total 5000):
  EN STEM        25%
  AM STEM solve  35%
  AM tutoring    15%
  AM instruct    15%
  EN replay      10%

Example:
  python data/mix_sft_v2.py \\
    --en-stem data/train/en_stem_sft_v2.jsonl \\
    --am-stem data/train/sources/afriquellm_gsm8k_am_sft_v0.jsonl \\
              data/train/am_stem_sft_nllb_v2_filtered.jsonl \\
    --am-tutoring data/train/am_tutoring_sft_v2.jsonl \\
    --am-instruct data/train/sources/walia_sft_v0.jsonl \\
                  data/train/sources/dolly_am_sft_v0.jsonl \\
    --out data/train/sft_mix_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.run_log import RunLogger  # noqa: E402

DEDUP = ROOT / "eval" / "dedup_against_eval.py"
ETHIOPIC_RE = re.compile(r"[\u1200-\u137F]")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
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


def sample_pool(rows: list[dict], k: int, rng: random.Random) -> list[dict]:
    if k <= 0 or not rows:
        return []
    if k >= len(rows):
        return list(rows)
    return rng.sample(rows, k)


def ethiopic_both(row: dict) -> bool:
    msgs = row.get("messages") or []
    user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
    asst = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
    return bool(ETHIOPIC_RE.search(user or "")) and bool(ETHIOPIC_RE.search(asst or ""))


def is_am_solve(row: dict) -> bool:
    if row.get("behavior") not in (None, "solve", "explain"):
        # allow solve/explain; tutoring behaviors handled elsewhere
        if row.get("behavior") in ("hint", "first_error", "code_switch", "instruct"):
            return False
    direction = row.get("direction", "")
    if direction not in ("am_am", "en_am"):
        # still accept if Ethiopic present
        if not ethiopic_both(row) and direction != "am_am":
            return False
    src = str(row.get("source", ""))
    if "tutoring" in src:
        return False
    return ethiopic_both(row) or direction == "am_am"


def load_many(paths: list[Path], *, ethiopic_only: bool) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        for r in read_jsonl(p):
            if ethiopic_only and not ethiopic_both(r):
                continue
            rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--en-stem", type=Path, nargs="+", required=True)
    ap.add_argument("--am-stem", type=Path, nargs="*", default=[])
    ap.add_argument("--am-tutoring", type=Path, nargs="*", default=[])
    ap.add_argument("--am-instruct", type=Path, nargs="*", default=[])
    ap.add_argument(
        "--eval",
        type=Path,
        nargs="+",
        default=[
            ROOT / "data" / "eval" / "custom_tutoring_v0.jsonl",
            ROOT / "data" / "eval" / "en_stem_holdout_v0.jsonl",
            ROOT / "data" / "eval" / "afrimgsm_amh_test_v0.jsonl",
            ROOT / "data" / "eval" / "afrimgsm_eng_test_v0.jsonl",
        ],
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "sft_mix_v2.jsonl")
    ap.add_argument("--total", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--en-ratio", type=float, default=0.25)
    ap.add_argument("--am-stem-ratio", type=float, default=0.35)
    ap.add_argument("--am-tutoring-ratio", type=float, default=0.15)
    ap.add_argument("--am-instruct-ratio", type=float, default=0.15)
    ap.add_argument("--replay-ratio", type=float, default=0.10)
    ap.add_argument("--counts-out", type=Path, default=ROOT / "docs" / "artifacts" / "sft_mix_v2_counts.json")
    args = ap.parse_args()

    ratios = (
        args.en_ratio
        + args.am_stem_ratio
        + args.am_tutoring_ratio
        + args.am_instruct_ratio
        + args.replay_ratio
    )
    if abs(ratios - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {ratios}")

    log = RunLogger(
        "mix_sft_v2",
        meta={"out": str(args.out), "total": args.total, "seed": args.seed},
    )
    rng = random.Random(args.seed)

    try:
        en_rows: list[dict] = []
        for p in args.en_stem:
            chunk = read_jsonl(p)
            en_rows.extend(chunk)
            log.item_ok("en_stem", path=str(p), n_rows=len(chunk))

        am_stem_raw = load_many(args.am_stem, ethiopic_only=False)
        am_stem = [r for r in am_stem_raw if is_am_solve(r) or r.get("behavior") == "solve"]
        # Prefer am_am
        am_stem_am = [r for r in am_stem if r.get("direction") == "am_am" or ethiopic_both(r)]
        if am_stem_am:
            am_stem = am_stem_am
        log.item_ok("am_stem", n_rows=len(am_stem), n_raw=len(am_stem_raw))

        am_tutor = load_many(args.am_tutoring, ethiopic_only=False)
        log.item_ok("am_tutoring", n_rows=len(am_tutor))

        am_instr = load_many(args.am_instruct, ethiopic_only=True)
        log.item_ok("am_instruct", n_rows=len(am_instr))

        n_en = int(args.total * args.en_ratio)
        n_am_stem = int(args.total * args.am_stem_ratio)
        n_am_tutor = int(args.total * args.am_tutoring_ratio)
        n_am_instr = int(args.total * args.am_instruct_ratio)
        n_replay = args.total - n_en - n_am_stem - n_am_tutor - n_am_instr

        # If a bucket is short, redistribute remainder into AM stem then EN.
        picked_am_stem = sample_pool(am_stem, n_am_stem, rng)
        short_stem = n_am_stem - len(picked_am_stem)
        picked_tutor = sample_pool(am_tutor, n_am_tutor, rng)
        short_tutor = n_am_tutor - len(picked_tutor)
        picked_instr = sample_pool(am_instr, n_am_instr, rng)
        short_instr = n_am_instr - len(picked_instr)

        extra = short_stem + short_tutor + short_instr
        picked_en = sample_pool(en_rows, n_en + max(0, extra // 2), rng)
        picked_replay = sample_pool(en_rows, max(0, n_replay + extra - max(0, extra // 2)), rng)

        # Top up AM stem from leftover am_stem if tutor/instr short and stem has more
        if short_tutor or short_instr:
            already = {id(r) for r in picked_am_stem}
            rest = [r for r in am_stem if id(r) not in already]
            picked_am_stem.extend(sample_pool(rest, short_tutor + short_instr, rng))

        mixed = picked_en + picked_am_stem + picked_tutor + picked_instr + picked_replay
        rng.shuffle(mixed)

        tmp = args.out.with_suffix(".pre_dedup.jsonl")
        write_jsonl(tmp, mixed)
        log.item_start("dedup", n_before=len(mixed), out=str(args.out))
        eval_paths = [p for p in args.eval if p.exists()]
        cmd = [
            sys.executable,
            str(DEDUP),
            "--train",
            str(tmp),
            "--eval",
            *[str(p) for p in eval_paths],
            "--out",
            str(args.out),
        ]
        subprocess.check_call(cmd)
        tmp.unlink(missing_ok=True)
        final = read_jsonl(args.out)
        log.item_ok("dedup", n_before=len(mixed), n_after=len(final), path=str(args.out))

        src_c = Counter(r.get("source", "unk") for r in final)
        dir_c = Counter(r.get("direction", "unk") for r in final)
        beh_c = Counter(r.get("behavior", "unk") for r in final)
        counts = {
            "n": len(final),
            "source": dict(src_c),
            "direction": dict(dir_c),
            "behavior": dict(beh_c),
            "targets": {
                "en": n_en,
                "am_stem": n_am_stem,
                "am_tutoring": n_am_tutor,
                "am_instruct": n_am_instr,
                "replay": n_replay,
            },
        }
        args.counts_out.parent.mkdir(parents=True, exist_ok=True)
        args.counts_out.write_text(json.dumps(counts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log.item_ok("counts", path=str(args.counts_out), n=len(final))
    except KeyboardInterrupt:
        log.warn("KeyboardInterrupt")
        log.finish(status="interrupted", message="interrupted by user")
        raise SystemExit(130) from None
    except Exception as e:  # noqa: BLE001
        log.item_error("mix", e)
        log.finish(status="error")
        raise

    summary = log.finish()
    print(f"mixed -> {args.out}")
    print(f"counts -> {args.counts_out}")
    print(f"run log -> {log.log_path}")
    print(f"summary -> {summary}")


if __name__ == "__main__":
    main()
