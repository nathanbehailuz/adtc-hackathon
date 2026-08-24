#!/usr/bin/env python3
"""Assemble cpt_mix_v3.jsonl from on-disk CPT pools (+ EN stem text slices).

Suggested shares (DATASETS.md):
  native Amharic   35%  (FineWeb2-am + Wikipedia-am)
  parallel/edu     25%  (AfriNLLB)
  EN STEM text     20%  (from en_stem_sft_v2 messages)
  EN replay        10%  (more EN stem)
  native fill      10%  (extra FineWeb2/Wikipedia)

Each output row: {"id": "...", "text": "...", "source": "..."}.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.run_log import RunLogger  # noqa: E402

CPT = ROOT / "data" / "train" / "cpt"
NATIVE = CPT / "native_am"
PARALLEL = CPT / "parallel"


def read_jsonl(path: Path) -> list[dict]:
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
        if not text:
            continue
        out.append({"id": f"{source}_{i:06d}", "text": text, "source": source})
    return out


def en_stem_as_text(path: Path, source: str) -> list[dict]:
    out: list[dict] = []
    for i, r in enumerate(read_jsonl(path)):
        msgs = r.get("messages") or []
        parts = [str(m.get("content", "")).strip() for m in msgs if m.get("content")]
        text = "\n".join(p for p in parts if p)
        if len(text) < 20:
            continue
        out.append(
            {
                "id": f"{source}_{r.get('id', i)}",
                "text": text,
                "source": source,
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--fineweb",
        type=Path,
        default=NATIVE / "fineweb2_amh_100m_v0.jsonl",
    )
    ap.add_argument(
        "--wikipedia",
        type=Path,
        default=NATIVE / "wikipedia_amharic_v0.jsonl",
    )
    ap.add_argument(
        "--afrinllb",
        type=Path,
        default=PARALLEL / "afrinllb_v0.jsonl",
    )
    ap.add_argument(
        "--en-stem",
        type=Path,
        default=ROOT / "data" / "train" / "en_stem_sft_v2.jsonl",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "cpt_mix_v3.jsonl")
    ap.add_argument("--total", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--native-ratio", type=float, default=0.35)
    ap.add_argument("--parallel-ratio", type=float, default=0.25)
    ap.add_argument("--en-stem-ratio", type=float, default=0.20)
    ap.add_argument("--replay-ratio", type=float, default=0.10)
    ap.add_argument("--fill-ratio", type=float, default=0.10)
    ap.add_argument(
        "--counts-out",
        type=Path,
        default=ROOT / "docs" / "artifacts" / "cpt_mix_v3_counts.json",
    )
    args = ap.parse_args()

    ratios = (
        args.native_ratio
        + args.parallel_ratio
        + args.en_stem_ratio
        + args.replay_ratio
        + args.fill_ratio
    )
    if abs(ratios - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1.0, got {ratios}")

    for p in (args.fineweb, args.wikipedia, args.afrinllb, args.en_stem):
        if not p.exists():
            raise FileNotFoundError(f"missing CPT/SFT input: {p}")

    log = RunLogger(
        "mix_cpt_v3",
        meta={"out": str(args.out), "total": args.total, "seed": args.seed},
    )
    rng = random.Random(args.seed)

    try:
        fineweb = as_text_rows(args.fineweb, "fineweb2_amh")
        wiki = as_text_rows(args.wikipedia, "wikipedia_amharic")
        native = fineweb + wiki
        parallel = as_text_rows(args.afrinllb, "afrinllb")
        en_stem = en_stem_as_text(args.en_stem, "en_stem_sft_v2")
        log.item_ok("pools", native=len(native), parallel=len(parallel), en_stem=len(en_stem))

        n_native = int(args.total * args.native_ratio)
        n_par = int(args.total * args.parallel_ratio)
        n_en = int(args.total * args.en_stem_ratio)
        n_replay = int(args.total * args.replay_ratio)
        n_fill = args.total - n_native - n_par - n_en - n_replay

        picked = []
        picked.extend(sample_pool(native, n_native, rng))
        picked.extend(sample_pool(parallel, n_par, rng))
        picked.extend(sample_pool(en_stem, n_en, rng))
        # Replay: more EN stem, allowing overlap with n_en pool via independent sample
        picked.extend(sample_pool(en_stem, n_replay, rng))
        already = {r["id"] for r in picked if r.get("source", "").startswith("fineweb") or r.get("source") == "wikipedia_amharic"}
        native_rest = [r for r in native if r["id"] not in already]
        picked.extend(sample_pool(native_rest or native, n_fill, rng))

        # If EN stem pool is short, top up from remaining native text.
        if len(picked) < args.total:
            have = {r["id"] for r in picked}
            rest = [r for r in native if r["id"] not in have]
            picked.extend(sample_pool(rest or native, args.total - len(picked), rng))

        rng.shuffle(picked)
        picked = picked[: args.total]
        write_jsonl(args.out, picked)

        src_c = Counter(r.get("source", "unk") for r in picked)
        counts = {
            "n": len(picked),
            "source": dict(src_c),
            "targets": {
                "native": n_native,
                "parallel": n_par,
                "en_stem": n_en,
                "replay": n_replay,
                "fill": n_fill,
            },
        }
        args.counts_out.parent.mkdir(parents=True, exist_ok=True)
        args.counts_out.write_text(
            json.dumps(counts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        log.item_ok("write", path=str(args.out), n=len(picked), counts=str(args.counts_out))
    except Exception as e:  # noqa: BLE001
        log.item_error("mix_cpt", e)
        log.finish(status="error")
        raise

    summary = log.finish()
    print(f"cpt mix -> {args.out}")
    print(f"counts -> {args.counts_out}")
    print(f"run log -> {log.log_path}")
    print(f"summary -> {summary}")


if __name__ == "__main__":
    main()
