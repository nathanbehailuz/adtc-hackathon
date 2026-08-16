#!/usr/bin/env python3
"""Normalize CPT corpora to JSONL {\"text\": ...} under train/cpt/.

Prefers ``data/raw/snapshots/<key>.jsonl`` from download_train_sources.py
(avoids re-streaming HF and a known datasets teardown abort on Slurm).

Does not use frozen eval sets.

Run log: ``logs/normalize_cpt/<run>.*`` (OK/FAIL per source).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_DATASETS_NUM_PROC", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.run_log import RunLogger  # noqa: E402

RAW = ROOT / "data" / "raw" / "hf"
SNAPSHOTS = ROOT / "data" / "raw" / "snapshots"
OUT = ROOT / "data" / "train" / "cpt"


def text_from_example(ex: dict) -> str:
    for k in ("text", "content", "article", "body", "sentence", "document", "text_original"):
        if ex.get(k):
            return str(ex[k]).strip()
    src = ex.get("source") or ex.get("src") or ex.get("english") or ex.get("en")
    tgt = ex.get("target") or ex.get("tgt") or ex.get("amharic") or ex.get("am") or ex.get("amh")
    if src and tgt:
        return f"{src}\n{tgt}".strip()
    strings = [str(v).strip() for v in ex.values() if isinstance(v, str) and len(v.strip()) > 20]
    return max(strings, key=len) if strings else ""


def load_snapshot(name: str, limit: int | None) -> list[dict] | None:
    path = SNAPSHOTS / f"{name}.jsonl"
    if not path.exists():
        return None
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_hf(hf_id: str, limit: int | None, split: str = "train"):
    from datasets import load_dataset

    try:
        if limit:
            ds = load_dataset(hf_id, split=split, cache_dir=str(RAW), streaming=True)
            rows = []
            for i, ex in enumerate(ds):
                if i >= limit:
                    break
                rows.append(dict(ex))
            return rows
    except Exception:
        pass
    ds = load_dataset(hf_id, split=split, cache_dir=str(RAW))
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def load_rows(name: str, hf_id: str, limit: int | None):
    snap = load_snapshot(name, limit)
    if snap is not None:
        return snap, "snapshot"
    return load_hf(hf_id, limit), "hf"


def write_pool(
    name: str,
    hf_id: str,
    limit: int | None,
    subdir: str,
    log: RunLogger,
) -> dict:
    out_dir = OUT / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_v0.jsonl"
    log.item_start(name, hf_id=hf_id, limit=limit, subdir=subdir)
    try:
        ds, mode = load_rows(name, hf_id, limit)
        n = 0
        with out_path.open("w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                ex = dict(ex) if not isinstance(ex, dict) else ex
                text = text_from_example(ex)
                if not text:
                    continue
                f.write(
                    json.dumps(
                        {"id": f"{name}_{i:07d}", "text": text, "source": name},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
        rel = str(out_path.relative_to(ROOT))
        log.item_ok(name, n_rows=n, path=rel, hf_id=hf_id, mode=mode)
        return {"key": name, "hf_id": hf_id, "status": "ok", "n_rows": n, "path": rel, "mode": mode}
    except Exception as e:  # noqa: BLE001
        log.item_error(name, e, hf_id=hf_id)
        return {"key": name, "hf_id": hf_id, "status": "error", "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-per-source", type=int, default=20_000)
    parser.add_argument("--fineweb-limit", type=int, default=50_000)
    args = parser.parse_args()

    log = RunLogger(
        "normalize_cpt",
        meta={
            "limit_per_source": args.limit_per_source,
            "fineweb_limit": args.fineweb_limit,
        },
    )
    pools = [
        ("fineweb2_amh_100m", "MultilingualUnigramLM/FineWeb2-amh_Ethi-100M", args.fineweb_limit, "native_am"),
        ("amharic_news", "dagn/expanded-amharic-news-dataset", args.limit_per_source, "native_am"),
        ("wikipedia_amharic", "addisai/wikipedia-amharic", args.limit_per_source, "native_am"),
        ("afrinllb", "AfriNLP/AfriNLLB-train", args.limit_per_source, "parallel"),
    ]
    results: list[dict] = []
    try:
        for name, hf_id, limit, subdir in pools:
            if not (SNAPSHOTS / f"{name}.jsonl").exists():
                log.item_skip(name, "no local snapshot; re-download or set HF_TOKEN for gated sets")
                results.append({"key": name, "hf_id": hf_id, "status": "skipped", "reason": "no snapshot"})
                continue
            results.append(write_pool(name, hf_id, limit, subdir, log))
    except KeyboardInterrupt:
        log.warn("KeyboardInterrupt — writing partial CPT normalize summary")
        summary = OUT / "cpt_normalize_summary_v0.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        log.finish(status="interrupted", message="interrupted by user")
        raise SystemExit(130) from None

    summary = OUT / "cpt_normalize_summary_v0.json"
    summary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    run_summary = log.finish()
    print(f"wrote {summary}")
    print(f"run log -> {log.log_path}")
    print(f"summary -> {run_summary}")


if __name__ == "__main__":
    main()
