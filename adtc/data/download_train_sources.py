#!/usr/bin/env python3
"""Download Amharic/EN *training* sources into gitignored cache + JSONL snapshots.

Never touches frozen eval under data/eval/.

Usage:
  cd adtc
  source tools/adtc-profiler-venv/bin/activate   # or any env with `datasets`
  python data/download_train_sources.py --profile first_experiment

  # subset:
  python data/download_train_sources.py --only walia finetome_am

Outputs:
  data/raw/hf/                         HF datasets cache
  data/raw/snapshots/<key>.jsonl       row snapshots (capped by limit)
  data/raw/download_manifest_v0.json   latest status + row counts
  logs/download_train/<run>.*          per-run OK/FAIL log (survives Ctrl+C)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.run_log import RunLogger  # noqa: E402

RAW = ROOT / "data" / "raw" / "hf"
SNAPSHOTS = ROOT / "data" / "raw" / "snapshots"
MANIFEST = ROOT / "data" / "raw" / "download_manifest_v0.json"

FIRST_EXPERIMENT = [
    {
        "key": "fineweb2_amh_100m",
        "hf_id": "MultilingualUnigramLM/FineWeb2-amh_Ethi-100M",
        "config": None,
        "split": "train",
        "role": "cpt",
        "limit": 50_000,
    },
    {
        "key": "amharic_news",
        "hf_id": "dagn/expanded-amharic-news-dataset",
        "config": None,
        "split": "train",
        "role": "cpt",
        "limit": 20_000,
    },
    {
        "key": "wikipedia_amharic",
        "hf_id": "addisai/wikipedia-amharic",
        "config": None,
        "split": "train",
        "role": "cpt",
        "limit": 50_000,
    },
    {
        "key": "afrinllb",
        "hf_id": "AfriNLP/AfriNLLB-train",
        "config": None,
        "split": "train",
        "role": "cpt",
        "limit": 50_000,
    },
    {
        "key": "walia",
        "hf_id": "EthioNLP/Amharic_Instruction_dataset",
        "config": None,
        "split": "train",
        "role": "sft",
        "limit": None,
    },
    {
        "key": "finetome_am",
        "hf_id": "addisai/FineTome-single-turn-dedup-amharic",
        "config": None,
        "split": "train",
        "role": "sft",
        "limit": None,
    },
    {
        "key": "afriquellm_gsm8k",
        "hf_id": "peterlu02/afriquellm-coldstart-gsm8k-11lang",
        "config": None,
        "split": "train",
        "role": "sft",
        "limit": None,
    },
    {
        # Fallback Amharic GSM8K (Seamless MT) if AfriqueLLM Hub id flakes.
        "key": "simonbutt_amharic_gsm8k",
        "hf_id": "simonbutt/amharic_gsm8k",
        "config": None,
        "split": "train",
        "role": "sft",
        "limit": None,
    },
    {
        "key": "r1_multilingual",
        "hf_id": "lightblue/reasoning-multilingual-R1-Llama-70B-train",
        "config": None,
        "split": "train",
        "role": "sft",
        "limit": 10_000,
    },
    {
        "key": "dolly_am",
        "hf_id": "iocuydi/amharic-dolly-15k",
        "config": None,
        "split": "train",
        "role": "sft",
        "limit": None,
    },
    {
        "key": "taco_am",
        "hf_id": "CRLannister/Amharic",
        "config": None,
        "split": "train",
        "role": "sft",
        "limit": 20_000,
    },
]

SKIP = [
    {"key": "fineweb2_full", "hf_id": "HuggingFaceFW/fineweb-2", "reason": "scale-up only; use 100M slice"},
    {"key": "yoseali", "hf_id": "YoseAli/amharic-llm-training-data", "reason": "excluded wholesale (contamination risk)"},
]


def _jsonable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def _hf_token() -> str | None:
    import os

    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _try_load(hf_id: str, config: str | None, split: str, streaming: bool, *, data_dir: str | None = None):
    from datasets import load_dataset

    kwargs: dict = {"split": split, "cache_dir": str(RAW), "streaming": streaming}
    token = _hf_token()
    if token:
        kwargs["token"] = token
    if data_dir:
        kwargs["path"] = data_dir
    else:
        kwargs["path"] = hf_id
    if config:
        kwargs["name"] = config
    return load_dataset(**kwargs)


def _snapshot_local(hf_id: str) -> Path:
    """Download repo files via huggingface_hub (more reliable than datasets alone)."""
    from huggingface_hub import snapshot_download

    dest = ROOT / "data" / "raw" / "hub_snapshots" / hf_id.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    kwargs: dict = {
        "repo_id": hf_id,
        "repo_type": "dataset",
        "local_dir": str(dest),
    }
    token = _hf_token()
    if token:
        kwargs["token"] = token
    try:
        snapshot_download(**kwargs, local_dir_use_symlinks=False)
    except TypeError:
        snapshot_download(**kwargs)
    return dest


def load_rows(spec: dict) -> tuple[list[dict], str]:
    """Return (rows, mode). Tries train then common alternate splits.

    Falls back to ``snapshot_download`` → local ``load_dataset`` when Hub id
    fails via datasets alone (seen with AfriqueLLM coldstart).
    """
    splits = [spec.get("split") or "train", "train", "validation", "test", "dev"]
    # unique preserve order
    seen = set()
    splits = [s for s in splits if not (s in seen or seen.add(s))]
    last_err: Exception | None = None
    data_dirs: list[str | None] = [None]

    def _collect(split: str, streaming: bool, data_dir: str | None) -> tuple[list[dict], str] | None:
        nonlocal last_err
        try:
            ds = _try_load(
                spec["hf_id"], spec.get("config"), split, streaming, data_dir=data_dir
            )
            if streaming:
                rows = []
                lim = spec.get("limit")
                for i, ex in enumerate(ds):
                    if lim is not None and i >= lim:
                        break
                    rows.append(_jsonable(dict(ex)))
                if rows:
                    tag = f"streaming:{split}"
                    if data_dir:
                        tag = f"snap+{tag}"
                    return rows, tag
            else:
                if spec.get("limit"):
                    n = min(spec["limit"], len(ds))
                    ds = ds.select(range(n))
                rows = [_jsonable(dict(ds[i])) for i in range(len(ds))]
                if rows:
                    tag = f"map:{split}"
                    if data_dir:
                        tag = f"snap+{tag}"
                    return rows, tag
        except Exception as e:  # noqa: BLE001
            last_err = e
        return None

    for split in splits:
        if spec.get("limit"):
            got = _collect(split, streaming=True, data_dir=None)
            if got:
                return got
        got = _collect(split, streaming=False, data_dir=None)
        if got:
            return got

    # AfriqueLLM / flaky Hub: snapshot then load from local dir
    try:
        local = _snapshot_local(spec["hf_id"])
        data_dirs.append(str(local))
    except Exception as e:  # noqa: BLE001
        last_err = e

    for data_dir in data_dirs[1:]:
        for split in splits:
            if spec.get("limit"):
                got = _collect(split, streaming=True, data_dir=data_dir)
                if got:
                    return got
            got = _collect(split, streaming=False, data_dir=data_dir)
            if got:
                return got

    raise RuntimeError(f"failed to load {spec['hf_id']}: {last_err}")


def write_snapshot(key: str, rows: list[dict]) -> Path:
    SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOTS / f"{key}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_manifest(profile: str, results: list[dict], *, run_id: str | None, interrupted: bool) -> None:
    manifest = {
        "profile": profile,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cache_dir": str(RAW),
        "snapshots_dir": str(SNAPSHOTS),
        "skipped": SKIP,
        "run_id": run_id,
        "interrupted": interrupted,
        "downloads": results,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="first_experiment", choices=("first_experiment",))
    parser.add_argument("--only", nargs="*", help="Optional subset of keys")
    parser.add_argument("--no-snapshot", action="store_true", help="Only warm HF cache, skip JSONL export")
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    specs = FIRST_EXPERIMENT
    if args.only:
        wanted = set(args.only)
        specs = [s for s in specs if s["key"] in wanted]
        missing = wanted - {s["key"] for s in specs}
        if missing:
            print(f"warn: unknown keys ignored: {sorted(missing)}")

    log = RunLogger(
        "download_train",
        meta={
            "profile": args.profile,
            "only": args.only,
            "no_snapshot": args.no_snapshot,
            "n_planned": len(specs),
        },
    )
    for skip in SKIP:
        log.item_skip(skip["key"], skip["reason"], hf_id=skip["hf_id"], role="excluded")

    results: list[dict] = []
    interrupted = False
    try:
        for spec in specs:
            entry = {
                "key": spec["key"],
                "hf_id": spec["hf_id"],
                "role": spec["role"],
                "limit": spec.get("limit"),
                "status": "pending",
            }
            log.item_start(
                spec["key"],
                hf_id=spec["hf_id"],
                role=spec["role"],
                limit=spec.get("limit"),
            )
            try:
                rows, mode = load_rows(spec)
                entry["n_rows"] = len(rows)
                entry["mode"] = mode
                if not args.no_snapshot:
                    snap = write_snapshot(spec["key"], rows)
                    entry["snapshot"] = str(snap.relative_to(ROOT))
                    entry["bytes"] = snap.stat().st_size
                entry["status"] = "ok"
                log.item_ok(
                    spec["key"],
                    n_rows=len(rows),
                    mode=mode,
                    snapshot=entry.get("snapshot"),
                    bytes=entry.get("bytes"),
                    role=spec["role"],
                    hf_id=spec["hf_id"],
                )
            except Exception as e:  # noqa: BLE001
                entry["status"] = "error"
                entry["error"] = str(e)
                log.item_error(spec["key"], e, role=spec["role"], hf_id=spec["hf_id"])
            results.append(entry)
            # keep latest manifest even mid-run (HPC / Ctrl+C friendly)
            write_manifest(args.profile, results, run_id=log.run_id, interrupted=False)
    except KeyboardInterrupt:
        interrupted = True
        log.warn("KeyboardInterrupt — writing partial manifest/summary")
        write_manifest(args.profile, results, run_id=log.run_id, interrupted=True)
        summary = log.finish(status="interrupted", message="interrupted by user")
        print(f"\npartial summary -> {summary}")
        print(f"partial manifest -> {MANIFEST}")
        raise SystemExit(130) from None

    write_manifest(args.profile, results, run_id=log.run_id, interrupted=False)
    summary = log.finish()
    counts = log.counts()
    print(f"\nwrote {MANIFEST}")
    print(f"run log -> {log.log_path}")
    print(f"summary -> {summary}")
    print(f"summary ok={counts['ok']} error={counts['error']} skipped={counts['skipped']}")


if __name__ == "__main__":
    main()
