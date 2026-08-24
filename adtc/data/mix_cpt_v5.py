#!/usr/bin/env python3
"""Assemble cpt_mix_v5.jsonl — small incremental STEM/Amharic CPT (not Afrique-scale).

Target (~25k docs):
  ~40% native Amharic (FineWeb2 + Wikipedia)
  ~25% Amharic STEM / parallel edu (AfriNLLB + filtered NLLB STEM)
  ~20% English STEM replay
  ~10% additional EN STEM
  ~5% replay fill

Rows: {id, text, source}. Deduped against frozen eval.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.run_log import RunLogger  # noqa: E402

CPT = ROOT / "data" / "train" / "cpt"
NATIVE = CPT / "native_am"
PARALLEL = CPT / "parallel"
DEDUP = ROOT / "eval" / "dedup_against_eval.py"
EVAL_DIR = ROOT / "data" / "eval"


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


def as_text_rows(path: Path, source: str) -> list[dict]:
    out: list[dict] = []
    for i, r in enumerate(read_jsonl(path)):
        text = (r.get("text") or "").strip()
        if len(text) < 40:
            continue
        out.append({"id": f"{source}_{i:06d}", "text": text, "source": source})
    return out


def sft_messages_as_text(path: Path, source: str) -> list[dict]:
    out: list[dict] = []
    for i, r in enumerate(read_jsonl(path)):
        msgs = r.get("messages") or []
        parts = [str(m.get("content", "")).strip() for m in msgs if m.get("content")]
        text = "\n".join(p for p in parts if p)
        if len(text) < 40:
            continue
        out.append({"id": f"{source}_{r.get('id', i)}", "text": text, "source": source})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fineweb", type=Path, default=NATIVE / "fineweb2_amh_100m_v0.jsonl")
    ap.add_argument("--wikipedia", type=Path, default=NATIVE / "wikipedia_amharic_v0.jsonl")
    ap.add_argument("--afrinllb", type=Path, default=PARALLEL / "afrinllb_v0.jsonl")
    ap.add_argument(
        "--am-stem-text",
        type=Path,
        default=ROOT / "data" / "train" / "am_stem_sft_nllb_v2_filtered.jsonl",
    )
    ap.add_argument(
        "--en-stem",
        type=Path,
        default=ROOT / "data" / "train" / "en_stem_sft_v4.jsonl",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "cpt_mix_v5.jsonl")
    ap.add_argument("--total", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--native-ratio", type=float, default=0.40)
    ap.add_argument("--am-stem-ratio", type=float, default=0.25)
    ap.add_argument("--en-stem-ratio", type=float, default=0.20)
    ap.add_argument("--struct-ratio", type=float, default=0.10)
    ap.add_argument("--replay-ratio", type=float, default=0.05)
    ap.add_argument(
        "--counts-out",
        type=Path,
        default=ROOT / "docs" / "artifacts" / "v5" / "cpt_mix_v5_counts.json",
    )
    ap.add_argument(
        "--report-out",
        type=Path,
        default=ROOT / "docs" / "artifacts" / "v5" / "cpt_mix_v5_report.md",
    )
    args = ap.parse_args()

    ratios = (
        args.native_ratio
        + args.am_stem_ratio
        + args.en_stem_ratio
        + args.struct_ratio
        + args.replay_ratio
    )
    if abs(ratios - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {ratios}")

    log = RunLogger("mix_cpt_v5", meta={"out": str(args.out), "total": args.total})
    rng = random.Random(args.seed)

    try:
        fineweb = as_text_rows(args.fineweb, "fineweb2_amh")
        wiki = as_text_rows(args.wikipedia, "wikipedia_amharic")
        native = fineweb + wiki
        afrinllb = as_text_rows(args.afrinllb, "afrinllb")
        am_stem = sft_messages_as_text(args.am_stem_text, "am_stem_nllb_filtered")
        am_stem_pool = afrinllb + am_stem
        en_stem = sft_messages_as_text(args.en_stem, "en_stem_sft_v4")
        log.item_ok(
            "pools",
            native=len(native),
            am_stem_pool=len(am_stem_pool),
            en_stem=len(en_stem),
        )

        n_native = int(args.total * args.native_ratio)
        n_am = int(args.total * args.am_stem_ratio)
        n_en = int(args.total * args.en_stem_ratio)
        n_struct = int(args.total * args.struct_ratio)
        n_replay = args.total - n_native - n_am - n_en - n_struct

        picked: list[dict] = []
        picked.extend(sample_pool(native, n_native, rng))
        picked.extend(sample_pool(am_stem_pool, n_am, rng))
        picked.extend(sample_pool(en_stem, n_en, rng))
        have = {r["id"] for r in picked}
        en_rest = [r for r in en_stem if r["id"] not in have]
        picked.extend(sample_pool(en_rest or en_stem, n_struct, rng))
        have = {r["id"] for r in picked}
        en_rest2 = [r for r in en_stem if r["id"] not in have]
        native_rest = [r for r in native if r["id"] not in have]
        replay_pool = en_rest2 + native_rest
        picked.extend(sample_pool(replay_pool or native, n_replay, rng))

        rng.shuffle(picked)
        picked = picked[: args.total]

        tmp = args.out.with_suffix(".prededup.jsonl")
        write_jsonl(tmp, picked)
        eval_files = sorted(EVAL_DIR.glob("*_v0.jsonl"))
        cmd = [
            sys.executable,
            str(DEDUP),
            "--train",
            str(tmp),
            "--eval",
            *[str(p) for p in eval_files],
            "--out",
            str(args.out),
        ]
        subprocess.check_call(cmd)
        tmp.unlink(missing_ok=True)

        final = read_jsonl(args.out)
        src_counts = Counter(r.get("source", "?") for r in final)
        counts = {
            "n": len(final),
            "target_total": args.total,
            "source": dict(src_counts),
            "targets": {
                "native": n_native,
                "am_stem": n_am,
                "en_stem": n_en,
                "struct": n_struct,
                "replay": n_replay,
            },
            "seed": args.seed,
        }
        args.counts_out.parent.mkdir(parents=True, exist_ok=True)
        args.counts_out.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
        report = [
            "# CPT mix v5 report",
            "",
            f"- rows: **{len(final)}** (target {args.total})",
            f"- seed: {args.seed}",
            "",
            "## Sources",
            "",
            "| source | n |",
            "|--------|--:|",
        ]
        for k, v in src_counts.most_common():
            report.append(f"| `{k}` | {v} |")
        report.append("")
        args.report_out.write_text("\n".join(report) + "\n", encoding="utf-8")
        log.item_ok("write", path=str(args.out), n=len(final))
    except Exception as e:  # noqa: BLE001
        log.item_error("mix_cpt_v5", e)
        log.finish(status="error")
        raise

    log.finish()
    print(f"wrote {args.out} n={len(read_jsonl(args.out))}")


if __name__ == "__main__":
    main()
