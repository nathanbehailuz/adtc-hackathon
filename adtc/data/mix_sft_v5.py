#!/usr/bin/env python3
"""Assemble sft_mix_v5.jsonl — tutoring-heavy, simonbutt-capped, cross-lingual.

Target (~9000 rows):
  AM solve ~27%, AM tutoring ~22%, EN ~18%,
  am_en ~9%, en_am ~9%, instruct ~5%, fill remainder from AM pools.
simonbutt share capped at <=12%.
Schema: id, direction, behavior, messages, source.
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
EVAL_DIR = ROOT / "data" / "eval"
ETHIOPIC = re.compile(r"[\u1200-\u137F]")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
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
    user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    return bool(ETHIOPIC.search(str(user))) and bool(ETHIOPIC.search(str(asst)))


def load_many(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        rows.extend(read_jsonl(p))
    return rows


def normalize_row(row: dict, *, default_dir: str, default_beh: str) -> dict | None:
    msgs = row.get("messages")
    if not msgs or not isinstance(msgs, list):
        return None
    roles = [m.get("role") for m in msgs]
    if "user" not in roles or "assistant" not in roles:
        return None
    asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    if not str(asst).strip():
        return None
    out = dict(row)
    out["direction"] = out.get("direction") or default_dir
    out["behavior"] = out.get("behavior") or default_beh
    out["source"] = str(out.get("source") or "unknown")
    if "id" not in out:
        out["id"] = f"{out['source']}_{hash(json.dumps(msgs, ensure_ascii=False)) & 0xFFFFFFFF:08x}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--en-stem", type=Path, nargs="+", required=True)
    ap.add_argument("--am-stem", type=Path, nargs="+", required=True)
    ap.add_argument("--am-tutoring", type=Path, nargs="+", required=True)
    ap.add_argument("--am-instruct", type=Path, nargs="+", required=True)
    ap.add_argument("--xling", type=Path, nargs="*", default=[])
    ap.add_argument("--simonbutt", type=Path, nargs="*", default=[])
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "sft_mix_v5.jsonl")
    ap.add_argument("--total", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--simonbutt-max-frac", type=float, default=0.12)
    ap.add_argument("--am-solve-ratio", type=float, default=0.27)
    ap.add_argument("--am-tutor-ratio", type=float, default=0.22)
    ap.add_argument("--en-ratio", type=float, default=0.18)
    ap.add_argument("--am-en-ratio", type=float, default=0.09)
    ap.add_argument("--en-am-ratio", type=float, default=0.09)
    ap.add_argument("--instruct-ratio", type=float, default=0.05)
    ap.add_argument(
        "--counts-out",
        type=Path,
        default=ROOT / "docs" / "artifacts" / "v5" / "sft_mix_v5_counts.json",
    )
    ap.add_argument(
        "--report-out",
        type=Path,
        default=ROOT / "docs" / "artifacts" / "v5" / "sft_mix_v5_report.md",
    )
    args = ap.parse_args()

    log = RunLogger("mix_sft_v5", meta={"out": str(args.out), "total": args.total})
    rng = random.Random(args.seed)

    try:
        en = [
            normalize_row(r, default_dir="en_en", default_beh="solve")
            for r in load_many(args.en_stem)
        ]
        en = [r for r in en if r]

        am_solve: list[dict] = []
        for r in load_many(args.am_stem):
            nr = normalize_row(r, default_dir="am_am", default_beh="solve")
            if not nr:
                continue
            if nr.get("behavior") in ("hint", "first_error", "code_switch"):
                continue
            if nr.get("direction") not in ("am_am", "en_am") and not ethiopic_both(nr):
                continue
            am_solve.append(nr)

        am_tutor = [
            normalize_row(r, default_dir="am_am", default_beh=str(r.get("behavior") or "explain"))
            for r in load_many(args.am_tutoring)
        ]
        am_tutor = [r for r in am_tutor if r]

        am_instr = [
            normalize_row(r, default_dir="am_am", default_beh="instruct")
            for r in load_many(args.am_instruct)
        ]
        am_instr = [r for r in am_instr if r and ethiopic_both(r)]

        xling = [
            normalize_row(r, default_dir=str(r.get("direction") or "am_en"), default_beh="explain")
            for r in load_many(args.xling)
        ]
        xling = [r for r in xling if r]
        am_en = [r for r in xling if r.get("direction") == "am_en"]
        en_am = [r for r in xling if r.get("direction") == "en_am"]

        simon = [
            normalize_row(r, default_dir="am_am", default_beh="solve")
            for r in load_many(args.simonbutt)
        ]
        simon = [r for r in simon if r]

        log.item_ok(
            "pools",
            en=len(en),
            am_solve=len(am_solve),
            am_tutor=len(am_tutor),
            am_instr=len(am_instr),
            am_en=len(am_en),
            en_am=len(en_am),
            simon=len(simon),
        )

        n_am_solve = int(args.total * args.am_solve_ratio)
        n_am_tutor = int(args.total * args.am_tutor_ratio)
        n_en = int(args.total * args.en_ratio)
        n_am_en = int(args.total * args.am_en_ratio)
        n_en_am = int(args.total * args.en_am_ratio)
        n_instr = int(args.total * args.instruct_ratio)
        n_simon_cap = int(args.total * args.simonbutt_max_frac)
        allocated = n_am_solve + n_am_tutor + n_en + n_am_en + n_en_am + n_instr
        n_fill = max(0, args.total - allocated)

        am_solve_nonsimon = [
            r for r in am_solve if "simon" not in str(r.get("source", "")).lower()
        ]
        picked_am_solve = sample_pool(am_solve_nonsimon, n_am_solve, rng)
        short = n_am_solve - len(picked_am_solve)
        if short > 0:
            picked_am_solve.extend(sample_pool(simon, min(short, n_simon_cap), rng))

        picked: list[dict] = []
        picked.extend(picked_am_solve)
        picked.extend(sample_pool(am_tutor, n_am_tutor, rng))
        picked.extend(sample_pool(en, n_en, rng))
        picked.extend(sample_pool(am_en, n_am_en, rng))
        picked.extend(sample_pool(en_am, n_en_am, rng))
        picked.extend(sample_pool(am_instr, n_instr, rng))
        if n_fill:
            have = {r["id"] for r in picked}
            fill_pool = [r for r in am_solve_nonsimon + am_tutor if r["id"] not in have]
            picked.extend(sample_pool(fill_pool, n_fill, rng))

        simon_idx = [
            i for i, r in enumerate(picked) if "simon" in str(r.get("source", "")).lower()
        ]
        if len(simon_idx) > n_simon_cap:
            drop = set(simon_idx[n_simon_cap:])
            picked = [r for i, r in enumerate(picked) if i not in drop]
            have = {r["id"] for r in picked}
            refill = [r for r in am_solve_nonsimon + am_tutor if r["id"] not in have]
            picked.extend(sample_pool(refill, args.total - len(picked), rng))

        rng.shuffle(picked)
        picked = picked[: args.total]

        tmp = args.out.with_suffix(".prededup.jsonl")
        write_jsonl(tmp, picked)
        eval_files = sorted(EVAL_DIR.glob("*_v0.jsonl"))
        subprocess.check_call(
            [
                sys.executable,
                str(DEDUP),
                "--train",
                str(tmp),
                "--eval",
                *[str(p) for p in eval_files],
                "--out",
                str(args.out),
            ]
        )
        tmp.unlink(missing_ok=True)

        final = read_jsonl(args.out)
        src = Counter(str(r.get("source", "?")) for r in final)
        direction = Counter(str(r.get("direction", "?")) for r in final)
        behavior = Counter(str(r.get("behavior", "?")) for r in final)
        simon_n = sum(v for k, v in src.items() if "simon" in k.lower())
        counts = {
            "n": len(final),
            "source": dict(src),
            "direction": dict(direction),
            "behavior": dict(behavior),
            "simonbutt_n": simon_n,
            "simonbutt_frac": round(simon_n / max(1, len(final)), 4),
            "seed": args.seed,
        }
        args.counts_out.parent.mkdir(parents=True, exist_ok=True)
        args.counts_out.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# SFT mix v5 report",
            "",
            f"- rows: **{len(final)}**",
            f"- simonbutt: {simon_n} ({counts['simonbutt_frac']:.1%}) cap={args.simonbutt_max_frac:.0%}",
            "",
            "## Direction",
            "",
            "| direction | n |",
            "|-----------|--:|",
        ]
        for k, v in direction.most_common():
            lines.append(f"| `{k}` | {v} |")
        lines += ["", "## Behavior", "", "| behavior | n |", "|----------|--:|"]
        for k, v in behavior.most_common():
            lines.append(f"| `{k}` | {v} |")
        lines += ["", "## Source (top)", "", "| source | n |", "|--------|--:|"]
        for k, v in src.most_common(20):
            lines.append(f"| `{k}` | {v} |")
        lines.append("")
        args.report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        log.item_ok(
            "write",
            path=str(args.out),
            n=len(final),
            simon_frac=counts["simonbutt_frac"],
        )
    except Exception as e:  # noqa: BLE001
        log.item_error("mix_sft_v5", e)
        log.finish(status="error")
        raise

    log.finish()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
