#!/usr/bin/env python3
"""Normalize Amharic/EN SFT HF datasets into project chat JSONL under train/sources/.

Dedups against frozen eval. Does not read eval into training outputs except for ban hashes.

Run log: ``logs/normalize_sft/<run>.*`` (OK/FAIL per source).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_DATASETS_NUM_PROC", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.run_log import RunLogger  # noqa: E402
RAW = ROOT / "data" / "raw" / "hf"
SNAPSHOTS = ROOT / "data" / "raw" / "snapshots"
OUT_DIR = ROOT / "data" / "train" / "sources"
DEDUP = ROOT / "eval" / "dedup_against_eval.py"
EVAL_FILES = [
    ROOT / "data" / "eval" / "custom_tutoring_v0.jsonl",
    ROOT / "data" / "eval" / "en_stem_holdout_v0.jsonl",
    ROOT / "data" / "eval" / "afrimgsm_amh_test_v0.jsonl",
    ROOT / "data" / "eval" / "afrimgsm_eng_test_v0.jsonl",
]

# Map normalize --sources name → download snapshot key
SNAPSHOT_KEY = {
    "walia": "walia",
    "finetome": "finetome_am",
    "afriquellm_gsm8k": "afriquellm_gsm8k",
    "r1": "r1_multilingual",
    "dolly": "dolly_am",
    "taco": "taco_am",
}


def emit(row_id: str, direction: str, behavior: str, user: str, assistant: str, source: str) -> dict:
    return {
        "id": row_id,
        "direction": direction,
        "behavior": behavior,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
    }


def looks_amharic(text: str) -> bool:
    # Ethiopic block
    return bool(re.search(r"[\u1200-\u137F]", text or ""))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_snapshot_rows(source_name: str, limit: int | None) -> list[dict] | None:
    key = SNAPSHOT_KEY.get(source_name, source_name)
    path = SNAPSHOTS / f"{key}.jsonl"
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


def load_hf(hf_id: str, split: str = "train", limit: int | None = None, config: str | None = None):
    from datasets import load_dataset

    kwargs = {"path": hf_id, "split": split, "cache_dir": str(RAW)}
    if config:
        kwargs["name"] = config
    ds = load_dataset(**kwargs)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def iter_examples(source_name: str, hf_id: str, limit: int | None = None, config: str | None = None):
    """Prefer local download snapshots; fall back to HF Hub/cache."""
    snap = load_snapshot_rows(source_name, limit)
    if snap is not None:
        return snap
    return load_hf(hf_id, limit=limit, config=config)


def from_instruction_fields(ex: dict) -> tuple[str, str] | None:
    # Common patterns: instruction/input/output, instruction/response, messages, prompt/completion
    if "messages" in ex and isinstance(ex["messages"], list) and len(ex["messages"]) >= 2:
        user = next((m.get("content", "") for m in ex["messages"] if m.get("role") in ("user", "human")), "")
        asst = next((m.get("content", "") for m in ex["messages"] if m.get("role") in ("assistant", "gpt")), "")
        if user and asst:
            return str(user), str(asst)
    instr = ex.get("instruction") or ex.get("prompt") or ex.get("question") or ""
    inp = ex.get("input") or ex.get("context") or ""
    out = ex.get("output") or ex.get("response") or ex.get("answer") or ex.get("completion") or ""
    if instr and out:
        user = f"{instr}\n\n{inp}".strip() if inp else str(instr)
        return user, str(out)
    return None


def normalize_walia(limit: int | None) -> Path:
    ds = iter_examples("walia", "EthioNLP/Amharic_Instruction_dataset", limit=limit)
    rows = []
    for i, ex in enumerate(ds):
        pair = from_instruction_fields(dict(ex))
        if not pair:
            continue
        u, a = pair
        rows.append(emit(f"walia_{i:06d}", "am_am", "instruct", u, a, "walia"))
    out = OUT_DIR / "walia_sft_v0.jsonl"
    write_jsonl(out, rows)
    print(f"walia -> {out} ({len(rows)})")
    return out


def normalize_finetome(limit: int | None) -> Path:
    ds = iter_examples("finetome", "addisai/FineTome-single-turn-dedup-amharic", limit=limit)
    rows = []
    for i, ex in enumerate(ds):
        pair = from_instruction_fields(dict(ex))
        if not pair:
            # FineTome sometimes: conversations
            conv = ex.get("conversations") or ex.get("conversation")
            if isinstance(conv, list) and len(conv) >= 2:
                u = conv[0].get("value") or conv[0].get("content") or ""
                a = conv[1].get("value") or conv[1].get("content") or ""
                if u and a:
                    pair = (str(u), str(a))
        if not pair:
            continue
        u, a = pair
        rows.append(emit(f"finetome_{i:06d}", "am_am", "instruct", u, a, "finetome_am"))
    out = OUT_DIR / "finetome_am_sft_v0.jsonl"
    write_jsonl(out, rows)
    print(f"finetome -> {out} ({len(rows)})")
    return out


def normalize_afriquellm_gsm8k(limit: int | None) -> Path:
    ds = iter_examples("afriquellm_gsm8k", "peterlu02/afriquellm-coldstart-gsm8k-11lang", limit=limit)
    rows = []
    for i, ex in enumerate(ds):
        ex = dict(ex)
        # Heuristic: keep Amharic-looking fields
        text_blob = json.dumps(ex, ensure_ascii=False)
        if not looks_amharic(text_blob):
            # also accept explicit lang tags
            lang = str(ex.get("language") or ex.get("lang") or "").lower()
            if lang not in ("am", "amh", "amharic"):
                continue
        pair = from_instruction_fields(ex)
        if not pair:
            q = ex.get("question") or ex.get("problem") or ""
            a = ex.get("answer") or ex.get("solution") or ex.get("response") or ""
            if q and a:
                pair = (str(q), str(a))
        if not pair:
            continue
        u, a = pair
        rows.append(emit(f"afriquellm_gsm8k_am_{i:06d}", "am_am", "solve", u, a, "afriquellm_gsm8k_am"))
    out = OUT_DIR / "afriquellm_gsm8k_am_sft_v0.jsonl"
    write_jsonl(out, rows)
    print(f"afriquellm_gsm8k_am -> {out} ({len(rows)})")
    return out


def normalize_r1(limit: int | None) -> Path:
    ds = iter_examples("r1", "lightblue/reasoning-multilingual-R1-Llama-70B-train", limit=limit)
    rows = []
    for i, ex in enumerate(ds):
        ex = dict(ex)
        blob = json.dumps(ex, ensure_ascii=False)
        if not looks_amharic(blob):
            lang = str(ex.get("language") or ex.get("lang") or "").lower()
            if lang not in ("am", "amh", "amharic"):
                continue
        pair = from_instruction_fields(ex)
        if not pair:
            continue
        u, a = pair
        rows.append(emit(f"r1_am_{i:06d}", "am_am", "explain", u, a, "r1_multilingual_am"))
    out = OUT_DIR / "r1_am_sft_v0.jsonl"
    write_jsonl(out, rows)
    print(f"r1_am -> {out} ({len(rows)})")
    return out


def normalize_dolly(limit: int | None) -> Path:
    ds = iter_examples("dolly", "iocuydi/amharic-dolly-15k", limit=limit)
    rows = []
    for i, ex in enumerate(ds):
        pair = from_instruction_fields(dict(ex))
        if not pair:
            continue
        u, a = pair
        rows.append(emit(f"dolly_am_{i:06d}", "am_am", "instruct", u, a, "dolly_am"))
    out = OUT_DIR / "dolly_am_sft_v0.jsonl"
    write_jsonl(out, rows)
    print(f"dolly_am -> {out} ({len(rows)})")
    return out


def normalize_taco(limit: int | None) -> Path:
    ds = iter_examples("taco", "CRLannister/Amharic", limit=limit)
    rows = []
    for i, ex in enumerate(ds):
        pair = from_instruction_fields(dict(ex))
        if not pair:
            continue
        u, a = pair
        rows.append(emit(f"taco_am_{i:06d}", "am_am", "instruct", u, a, "taco_am"))
    out = OUT_DIR / "taco_am_sft_v0.jsonl"
    write_jsonl(out, rows)
    print(f"taco_am -> {out} ({len(rows)})")
    return out


def dedup(path: Path) -> Path:
    eval_paths = [p for p in EVAL_FILES if p.exists()]
    if not eval_paths:
        print("warn: no eval files for dedup")
        return path
    out = path.with_name(path.stem + "_dedup.jsonl")
    cmd = [
        sys.executable,
        str(DEDUP),
        "--train",
        str(path),
        "--eval",
        *[str(p) for p in eval_paths],
        "--out",
        str(out),
    ]
    subprocess.check_call(cmd)
    path.unlink(missing_ok=True)
    out.rename(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Cap each source (smoke)")
    parser.add_argument(
        "--sources",
        nargs="*",
        default=["walia", "finetome", "afriquellm_gsm8k", "r1", "dolly", "taco"],
    )
    parser.add_argument("--skip-dedup", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    writers = {
        "walia": normalize_walia,
        "finetome": normalize_finetome,
        "afriquellm_gsm8k": normalize_afriquellm_gsm8k,
        "r1": normalize_r1,
        "dolly": normalize_dolly,
        "taco": normalize_taco,
    }
    log = RunLogger(
        "normalize_sft",
        meta={"sources": args.sources, "limit": args.limit, "skip_dedup": args.skip_dedup},
    )
    try:
        for name in args.sources:
            fn = writers.get(name)
            if not fn:
                log.item_skip(name, "unknown source name")
                continue
            log.item_start(name, limit=args.limit)
            try:
                key = SNAPSHOT_KEY.get(name, name)
                snap = SNAPSHOTS / f"{key}.jsonl"
                if not snap.exists() and name == "afriquellm_gsm8k":
                    log.item_skip(name, "no snapshot; download failed or missing on Hub")
                    continue
                path = fn(args.limit)
                deduped = False
                if not args.skip_dedup and path.exists() and path.stat().st_size > 0:
                    dedup(path)
                    deduped = True
                n_rows = sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0
                log.item_ok(
                    name,
                    path=str(path.relative_to(ROOT)),
                    n_rows=n_rows,
                    deduped=deduped,
                )
            except Exception as e:  # noqa: BLE001
                log.item_error(name, e)
    except KeyboardInterrupt:
        log.warn("KeyboardInterrupt — writing partial SFT normalize summary")
        log.finish(status="interrupted", message="interrupted by user")
        raise SystemExit(130) from None

    summary = log.finish()
    counts = log.counts()
    print(f"run log -> {log.log_path}")
    print(f"summary -> {summary}")
    print(f"summary ok={counts['ok']} error={counts['error']} skipped={counts['skipped']}")


if __name__ == "__main__":
    main()
