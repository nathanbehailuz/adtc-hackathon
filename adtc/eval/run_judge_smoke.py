#!/usr/bin/env python3
"""Judge-aligned tutoring smoke checks (paraphrases; not Round-1 verbatim).

Scores heuristic checklist pass rates + GSM8K markup leak rate.
Can score from a predictions JSONL or by calling a local GGUF.

Examples:
  python eval/run_judge_smoke.py --preds /tmp/preds.jsonl --out docs/artifacts/v7/judge_smoke.json
  python eval/run_judge_smoke.py --gguf artifacts/gguf/adapted/foo.gguf --out docs/artifacts/v7/judge_smoke.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "data" / "eval" / "judge_smoke_v7.jsonl"

MARKUP_RE = re.compile(r"(####|<<[^>]*>>)")


def load_suite() -> list[dict]:
    rows = []
    with SUITE.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def user_text(row: dict) -> str:
    for m in row.get("messages") or []:
        if m.get("role") == "user":
            return m["content"]
    return ""


def check_one(name: str, text: str) -> bool:
    t = text or ""
    low = t.lower()
    if name == "no_markup":
        return MARKUP_RE.search(t) is None
    if name == "no_final_x":
        return not re.search(r"\bx\s*=\s*-?\d", low)
    if name == "no_final_a":
        return not re.search(r"\ba\s*=\s*-?\d", low)
    if name == "has_why":
        return any(k in low for k in ("why", "because", "valid", "so that"))
    if name == "has_question":
        return "?" in t
    if name == "has_two_questions":
        return t.count("?") >= 2
    if name == "names_mistake":
        return any(k in low for k in ("mistake", "wrong", "error", "line"))
    if name == "has_hint":
        return "hint" in low or "recompute" in low or "rewrite" in low or "divide" in low
    if name == "mentions_turbine":
        return "turbine" in low
    if name == "mentions_energy":
        return "energy" in low
    if name == "mentions_force_or_power":
        return "force" in low or "power" in low
    if name == "has_analogy":
        return any(k in low for k in ("analogy", "like ", "think of", "similar"))
    if name == "has_misconception":
        return "misconception" in low or "common mistake" in low or "people think" in low
    if name == "velocity_zero":
        return "velocity" in low and "zero" in low
    if name == "accel_nonzero":
        return "acceleration" in low and any(
            k in low for k in ("not zero", "nonzero", "non-zero", "still", "gravity")
        )
    if name == "completing_square":
        return "complet" in low and "square" in low
    if name == "has_final_formula":
        return "2a" in low and ("b^2" in low or "b²" in low or "discriminant" in low)
    if name == "limiting_oxygen":
        return "oxygen" in low or "o2" in low
    if name == "co2_33":
        return bool(re.search(r"\b33\b", t))
    if name == "line1_or_distribute":
        return "line 1" in low or "distribut" in low
    return False


def score_pred(row: dict, pred: str) -> dict:
    checks = row.get("checks") or ["no_markup"]
    results = {c: check_one(c, pred) for c in checks}
    return {
        "id": row["id"],
        "behavior": row.get("behavior"),
        "checks": results,
        "pass": all(results.values()),
        "markup_leak": MARKUP_RE.search(pred or "") is not None,
        "pred_preview": (pred or "")[:240],
    }


def gen_gguf(gguf: Path, prompt: str, n_threads: int | None, max_tokens: int) -> str:
    from llama_cpp import Llama

    llm = Llama(model_path=str(gguf), n_ctx=4096, n_threads=n_threads or 8, verbose=False)
    system = (
        "You are an English STEM tutor. Address every requested part. "
        "Never use #### or <<>> markup. Hints/diagnoses must not reveal final answers."
    )
    try:
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return out["choices"][0]["message"]["content"]
    except Exception:
        out = llm(f"{system}\n\nUser: {prompt}\nAssistant:", max_tokens=max_tokens, temperature=0.2)
        return out["choices"][0]["text"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", type=Path, default=None, help="JSONL with id + pred fields")
    ap.add_argument("--gguf", type=Path, default=None)
    ap.add_argument("--n-threads", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    suite = load_suite()
    preds: dict[str, str] = {}
    if args.preds:
        with args.preds.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                preds[obj["id"]] = obj.get("pred") or obj.get("assistant") or ""
    elif args.gguf:
        for row in suite:
            preds[row["id"]] = gen_gguf(args.gguf, user_text(row), args.n_threads, args.max_tokens)
    else:
        raise SystemExit("Provide --preds or --gguf")

    scored = []
    for row in suite:
        pred = preds.get(row["id"], "")
        scored.append(score_pred(row, pred))

    n = max(1, len(scored))
    report = {
        "n": len(scored),
        "checklist_pass_rate": sum(s["pass"] for s in scored) / n,
        "markup_leak_rate": sum(s["markup_leak"] for s in scored) / n,
        "items": scored,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "checklist_pass_rate": report["checklist_pass_rate"],
                "markup_leak_rate": report["markup_leak_rate"],
                "n": report["n"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
