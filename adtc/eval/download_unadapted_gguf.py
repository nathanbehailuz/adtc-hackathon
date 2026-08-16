#!/usr/bin/env python3
"""Download official / community GGUF weights for Phase 2 unadapted screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# repo_id, filename patterns to try (first hit wins)
MANIFEST = [
    {
        "key": "qwen3_1_7b_q4_k_m",
        "repo": "unsloth/Qwen3-1.7B-GGUF",
        "files": ["Qwen3-1.7B-Q4_K_M.gguf", "qwen3-1.7b-q4_k_m.gguf"],
        "params": "1.7B",
        "quant": "Q4_K_M",
    },
    {
        "key": "qwen3_1_7b_q6_k",
        "repo": "unsloth/Qwen3-1.7B-GGUF",
        "files": ["Qwen3-1.7B-Q6_K.gguf", "qwen3-1.7b-q6_k.gguf"],
        "params": "1.7B",
        "quant": "Q6_K",
    },
    {
        "key": "qwen3_4b_q4_k_m",
        "repo": "Qwen/Qwen3-4B-GGUF",
        "files": ["Qwen3-4B-Q4_K_M.gguf", "qwen3-4b-q4_k_m.gguf"],
        "params": "4B",
        "quant": "Q4_K_M",
    },
    {
        "key": "qwen3_4b_q6_k",
        "repo": "Qwen/Qwen3-4B-GGUF",
        "files": ["Qwen3-4B-Q6_K.gguf", "qwen3-4b-q6_k.gguf"],
        "params": "4B",
        "quant": "Q6_K",
    },
    {
        "key": "qwen25_3b_q4_k_m",
        "repo": "Qwen/Qwen2.5-3B-Instruct-GGUF",
        "files": [
            "qwen2.5-3b-instruct-q4_k_m.gguf",
            "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        ],
        "params": "3B",
        "quant": "Q4_K_M",
    },
    {
        "key": "gemma3_4b_q4_k_m",
        "repo": "bartowski/google_gemma-3-4b-it-GGUF",
        "files": [
            "google_gemma-3-4b-it-Q4_K_M.gguf",
            "gemma-3-4b-it-Q4_K_M.gguf",
        ],
        "params": "4B",
        "quant": "Q4_K_M",
    },
    {
        "key": "gemma3_4b_q6_k",
        "repo": "bartowski/google_gemma-3-4b-it-GGUF",
        "files": [
            "google_gemma-3-4b-it-Q6_K.gguf",
            "gemma-3-4b-it-Q6_K.gguf",
        ],
        "params": "4B",
        "quant": "Q6_K",
    },
    {
        "key": "qwen35_2b_q4_k_m",
        "repo": "unsloth/Qwen3.5-2B-GGUF",
        "files": [
            "Qwen3.5-2B-Q4_K_M.gguf",
            "qwen3.5-2b-q4_k_m.gguf",
        ],
        "params": "2B",
        "quant": "Q4_K_M",
        "compat_only": True,
    },
]


def resolve_file(repo: str, candidates: list[str]) -> str | None:
    from huggingface_hub import list_repo_files

    try:
        files = set(list_repo_files(repo))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] list_repo_files failed for {repo}: {e}")
        return None
    for c in candidates:
        if c in files:
            return c
    # fuzzy: any file containing q4_k_m etc.
    lower = {f.lower(): f for f in files if f.endswith(".gguf")}
    for c in candidates:
        key = c.lower()
        if key in lower:
            return lower[key]
    # last resort: pick first matching quant token
    for c in candidates:
        token = c.lower().replace(".gguf", "")
        for lf, orig in lower.items():
            if token.split("-")[-1] in lf or token in lf:
                return orig
    print(f"[warn] no candidate matched in {repo}; sample={list(files)[:15]}")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "gguf" / "unadapted")
    ap.add_argument("--keys", nargs="*", default=None, help="Subset of manifest keys")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = MANIFEST
    if args.keys:
        want = set(args.keys)
        items = [m for m in MANIFEST if m["key"] in want]
    if args.limit is not None:
        items = items[: args.limit]

    summary = []
    for m in items:
        fname = resolve_file(m["repo"], m["files"])
        row = {**m, "resolved_file": fname, "path": None, "ok": False, "error": None}
        if not fname:
            row["error"] = "file_not_found"
            summary.append(row)
            continue
        try:
            path = hf_hub_download(
                repo_id=m["repo"],
                filename=fname,
                local_dir=str(args.out_dir / m["key"]),
            )
            row["path"] = path
            row["ok"] = True
            print(f"OK {m['key']} -> {path}")
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)
            print(f"FAIL {m['key']}: {e}")
        summary.append(row)

    out_json = args.out_dir / "download_manifest_v0.json"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    n_ok = sum(1 for s in summary if s["ok"])
    raise SystemExit(0 if n_ok > 0 else 1)


if __name__ == "__main__":
    main()
