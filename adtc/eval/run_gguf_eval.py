#!/usr/bin/env python3
"""Score a local GGUF on frozen eval suites via llama-cpp-python."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gold_answer(row: dict) -> str:
    ans = row.get("answer")
    if ans is not None and str(ans).strip() != "":
        return str(ans)
    raw = row.get("raw") or {}
    if isinstance(raw, dict):
        for k in ("answer_number", "answer", "target", "label"):
            if raw.get(k) is not None and str(raw.get(k)).strip() != "":
                return str(raw[k])
    return ""


def normalize_ans(s: str) -> str:
    s = (s or "").strip()
    if "</think>" in s:
        s = s.split("</think>")[-1]
    s = s.strip().lower().replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if nums:
        return nums[-1]
    return s.split()[-1] if s.split() else s


def make_llm(gguf: Path, n_ctx: int = 2048, n_threads: int | None = None):
    from llama_cpp import Llama

    kwargs = {
        "model_path": str(gguf),
        "n_ctx": n_ctx,
        "verbose": False,
    }
    if n_threads is not None:
        kwargs["n_threads"] = n_threads
    return Llama(**kwargs)


def gen(llm, prompt: str, max_new: int = 96) -> str:
    # Prefer chat if the model exposes a chat template via llama.cpp; else raw completion.
    # Qwen3: /no_think reduces CoT tokens (better TPS + cleaner answer extract).
    user = f"/no_think\n{prompt}"
    try:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": user}],
            max_tokens=max_new,
            temperature=0.0,
        )
        return out["choices"][0]["message"]["content"] or ""
    except Exception:  # noqa: BLE001
        out = llm(
            user,
            max_tokens=max_new,
            temperature=0.0,
            echo=False,
        )
        return out["choices"][0]["text"] or ""


def take(rows: list, limit: int | None) -> list:
    if limit is None:
        return rows
    return rows[:limit]


def score_qa(llm, rows: list[dict], limit: int | None) -> dict:
    rows = take(rows, limit)
    ok = 0
    for r in rows:
        pred = normalize_ans(gen(llm, r["question"]))
        gold = normalize_ans(gold_answer(r))
        ok += int(bool(gold) and pred == gold)
    n = max(1, len(rows))
    return {"n": len(rows), "correct": ok, "acc": ok / n}


def score_mmlu(llm, rows: list[dict], limit: int | None) -> dict:
    rows = take(rows, limit)
    ok = 0
    letters = ["A", "B", "C", "D"]
    for r in rows:
        ex = r.get("example") or r
        q = ex["question"]
        choices = ex.get("choices") or []
        prompt = q + "\n" + "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices[:4]))
        prompt += "\nAnswer with a single letter."
        pred = gen(llm, prompt, max_new=8).strip().upper()
        letter = pred[0] if pred else ""
        gold = str(ex.get("answer", "")).strip().upper()
        if gold.isdigit():
            gold = letters[int(gold)] if int(gold) < 4 else gold
        ok += int(bool(gold) and letter == gold[0])
    n = max(1, len(rows))
    return {"n": len(rows), "correct": ok, "acc": ok / n}


def score_tutoring(llm, rows: list[dict], limit: int | None) -> dict:
    rows = take(rows, limit)
    ok = 0
    samples = []
    for r in rows:
        msgs = r.get("messages") or []
        user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        if not user:
            continue
        pred = gen(llm, user, max_new=128)
        hit = len(pred.strip()) > 20
        ok += int(hit)
        if len(samples) < 3:
            samples.append({"prompt": user[:120], "pred": pred[:200]})
    n = max(1, len(rows))
    return {"n": len(rows), "correct": ok, "acc": ok / n, "samples": samples}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap items per suite (default: all frozen rows)",
    )
    ap.add_argument("--n-threads", type=int, default=None)
    ap.add_argument("--out", type=Path, required=True, help="JSON with frozen_suites (+ meta)")
    args = ap.parse_args()

    gguf = args.gguf.resolve()
    if not gguf.is_file():
        raise SystemExit(f"GGUF not found: {gguf}")

    llm = make_llm(gguf, n_threads=args.n_threads)
    eval_dir = ROOT / "data" / "eval"
    suites: dict = {}
    suites["afrimgsm_amh"] = score_qa(llm, load_jsonl(eval_dir / "afrimgsm_amh_test_v0.jsonl"), args.limit)
    suites["afrimgsm_eng"] = score_qa(llm, load_jsonl(eval_dir / "afrimgsm_eng_test_v0.jsonl"), args.limit)
    mmlu = eval_dir / "afrimmlu_amh_test_v0.jsonl"
    if mmlu.exists():
        suites["afrimmlu_amh"] = score_mmlu(llm, load_jsonl(mmlu), args.limit)
    suites["en_stem_holdout"] = score_qa(llm, load_jsonl(eval_dir / "en_stem_holdout_v0.jsonl"), args.limit)
    suites["custom_tutoring"] = score_tutoring(
        llm, load_jsonl(eval_dir / "custom_tutoring_v0.jsonl"), args.limit
    )

    report = {
        "gguf": str(gguf),
        "limit": args.limit,
        "frozen_suites": suites,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v.get("acc") for k, v in suites.items()}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
