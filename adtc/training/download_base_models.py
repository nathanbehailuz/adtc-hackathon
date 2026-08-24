#!/usr/bin/env python3
"""Download the v6 English-only base model (Qwen3-1.7B) for QLoRA SFT.

Does not download GGUF submission weights (that stays in submission ``download_model.sh``).

Usage (from ``adtc/``)::

  python training/download_base_models.py
  python training/download_base_models.py --revision main

Run log: ``logs/download_models/<run>.*`` (OK/FAIL per model).
Cache: ``HF_HOME`` or default Hugging Face cache (override with ``--cache-dir``).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.run_log import RunLogger  # noqa: E402

# v6 English-only track — single base model.
DEFAULT_MODELS = [
    {
        "key": "qwen3_1_7b",
        "hf_id": "Qwen/Qwen3-1.7B",
        "role": "v6_primary",
    },
]


def download_one(hf_id: str, cache_dir: Path | None, revision: str | None) -> dict:
    from huggingface_hub import snapshot_download

    kwargs: dict = {"repo_id": hf_id, "repo_type": "model"}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision:
        kwargs["revision"] = revision
    path = snapshot_download(**kwargs)
    return {"local_path": path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="Optional subset of model keys")
    parser.add_argument("--cache-dir", type=Path, default=None, help="HF cache dir (optional)")
    parser.add_argument("--revision", default=None, help="Optional HF revision/tag")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=ROOT / "training" / "model_download_manifest_v0.json",
    )
    args = parser.parse_args()

    models = DEFAULT_MODELS
    if args.only:
        wanted = set(args.only)
        models = [m for m in models if m["key"] in wanted]
        missing = wanted - {m["key"] for m in models}
        if missing:
            print(f"warn: unknown keys ignored: {sorted(missing)}")

    log = RunLogger(
        "download_models",
        meta={
            "only": args.only,
            "cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "revision": args.revision,
            "n_planned": len(models),
        },
    )
    results: list[dict] = []
    try:
        for spec in models:
            entry = {**spec, "status": "pending"}
            log.item_start(spec["key"], hf_id=spec["hf_id"], role=spec.get("role"))
            try:
                info = download_one(spec["hf_id"], args.cache_dir, args.revision)
                entry.update(info)
                entry["status"] = "ok"
                log.item_ok(spec["key"], hf_id=spec["hf_id"], **info)
            except Exception as e:  # noqa: BLE001
                entry["status"] = "error"
                entry["error"] = str(e)
                log.item_error(spec["key"], e, hf_id=spec["hf_id"])
            results.append(entry)
            args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
            args.manifest_out.write_text(
                json.dumps(
                    {
                        "run_id": log.run_id,
                        "models": results,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    except KeyboardInterrupt:
        log.warn("KeyboardInterrupt — writing partial model download manifest")
        log.finish(status="interrupted", message="interrupted by user")
        raise SystemExit(130) from None

    summary = log.finish()
    counts = log.counts()
    print(f"wrote {args.manifest_out}")
    print(f"run log -> {log.log_path}")
    print(f"summary -> {summary}")
    print(f"summary ok={counts['ok']} error={counts['error']}")


if __name__ == "__main__":
    main()
