#!/usr/bin/env python3
"""Normalize CPT corpora to JSONL {\"text\": ...} under train/cpt/.

Does not use frozen eval sets.

Run log: ``logs/normalize_cpt/<run>.*`` (OK/FAIL per source).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.run_log import RunLogger  # noqa: E402

RAW = ROOT / "data" / "raw" / "hf"
OUT = ROOT / "data" / "train" / "cpt"


def text_from_example(ex: dict) -> str:
    for k in ("text", "content", "article", "body", "sentence", "document"):
        if ex.get(k):
            return str(ex[k]).strip()
    # parallel: join src/tgt
    src = ex.get("source") or ex.get("src") or ex.get("english") or ex.get("en")
    tgt = ex.get("target") or ex.get("tgt") or ex.get("amharic") or ex.get("am") or ex.get("amh")
    if src and tgt:
        return f"{src}\n{tgt}".strip()
    # fallback: longest string field
    strings = [str(v).strip() for v in ex.values() if isinstance(v, str) and len(v.strip()) > 20]
    return max(strings, key=len) if strings else ""


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
        ds = load_hf(hf_id, limit)
        n = 0
        with out_path.open("w", encoding="utf-8") as f:
            for i, ex in enumerate(ds):
                ex = dict(ex) if not isinstance(ex, dict) else ex
                text = text_from_example(ex)
                if not text:
                    continue
                f.write(json.dumps({"id": f"{name}_{i:07d}", "text": text, "source": name}, ensure_ascii=False) + "\n")
                n += 1
        rel = str(out_path.relative_to(ROOT))
        log.item_ok(name, n_rows=n, path=rel, hf_id=hf_id)
        return {"key": name, "hf_id": hf_id, "status": "ok", "n_rows": n, "path": rel}
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
