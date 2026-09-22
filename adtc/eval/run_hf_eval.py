#!/usr/bin/env python3
"""Score merged HF checkpoints on frozen eval suites (exact/normalized match)."""
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
    # Drop Qwen3-style thinking blocks if present
    if "</think>" in s:
        s = s.split("</think>")[-1]
    s = s.strip().lower().replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if nums:
        return nums[-1]
    return s.split()[-1] if s.split() else s


def gen(model, tok, prompt: str, max_new: int = 96) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tok, "apply_chat_template"):
        try:
            # Qwen3: disable chain-of-thought for scoring speed + answer extraction.
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            try:
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:  # noqa: BLE001
                text = prompt
        except Exception:  # noqa: BLE001
            text = prompt
    else:
        text = prompt
    inputs = tok(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen_ids = out[0][inputs["input_ids"].shape[-1] :]
    return tok.decode(gen_ids, skip_special_tokens=True)


def take(rows: list, limit: int | None) -> list:
    if limit is None:
        return rows
    return rows[:limit]


def _tick(name: str, i: int, n: int) -> None:
    if i == 1 or i == n or i % 25 == 0:
        print(f"[{name}] {i}/{n}", flush=True)


def score_qa(model, tok, rows: list[dict], qkey: str, akey: str, limit: int | None, name: str = "qa") -> dict:
    rows = take(rows, limit)
    ok = 0
    n = len(rows)
    print(f"[{name}] start n={n}", flush=True)
    for i, r in enumerate(rows, start=1):
        pred = normalize_ans(gen(model, tok, r[qkey]))
        gold = normalize_ans(gold_answer(r) if akey == "answer" else str(r.get(akey, "")))
        ok += int(bool(gold) and pred == gold)
        _tick(name, i, n)
    n = max(1, n)
    return {"n": len(rows), "correct": ok, "acc": ok / n}


def score_mmlu(model, tok, rows: list[dict], limit: int | None, name: str = "mmlu") -> dict:
    rows = take(rows, limit)
    ok = 0
    letters = ["A", "B", "C", "D"]
    n = len(rows)
    print(f"[{name}] start n={n}", flush=True)
    for i, r in enumerate(rows, start=1):
        ex = r.get("example") or r
        q = ex["question"]
        choices = ex.get("choices") or []
        prompt = q + "\n" + "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices[:4]))
        prompt += "\nAnswer with a single letter."
        pred = gen(model, tok, prompt, max_new=8).strip().upper()
        letter = pred[0] if pred else ""
        gold = str(ex.get("answer", "")).strip().upper()
        if gold.isdigit():
            gold = letters[int(gold)] if int(gold) < 4 else gold
        ok += int(letter == gold[0] if gold else False)
        _tick(name, i, n)
    n = max(1, n)
    return {"n": len(rows), "correct": ok, "acc": ok / n}


def score_tutoring(model, tok, rows: list[dict], limit: int | None, name: str = "tutoring") -> dict:
    """Soft pass if the model produces a non-empty reply (len>20)."""
    rows = take(rows, limit)
    ok = 0
    samples = []
    n = len(rows)
    print(f"[{name}] start n={n}", flush=True)
    for i, r in enumerate(rows, start=1):
        msgs = r.get("messages") or []
        user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        if not user:
            continue
        pred = gen(model, tok, user, max_new=128)
        hit = len(pred.strip()) > 20
        ok += int(hit)
        if len(samples) < 3:
            samples.append({"prompt": user[:120], "pred": pred[:200]})
        _tick(name, i, n)
    n = max(1, n)
    return {"n": len(rows), "correct": ok, "acc": ok / n, "samples": samples}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap items per suite (default: all frozen rows)",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    print(
        f"loaded {args.model} cuda={torch.cuda.is_available()} "
        f"(EN suites only; first CUDA generate can take 1–2 min)",
        flush=True,
    )

    eval_dir = ROOT / "data" / "eval"
    report = {"model": args.model, "limit": args.limit, "suites": {}, "lang_scope": "en"}

    en = load_jsonl(eval_dir / "afrimgsm_eng_test_v0.jsonl")
    report["suites"]["afrimgsm_eng"] = score_qa(
        model, tok, en, "question", "answer", args.limit, name="afrimgsm_eng"
    )

    hold = load_jsonl(eval_dir / "en_stem_holdout_v0.jsonl")
    report["suites"]["en_stem_holdout"] = score_qa(
        model, tok, hold, "question", "answer", args.limit, name="en_stem_holdout"
    )

    tut = load_jsonl(eval_dir / "custom_tutoring_v0.jsonl")
    report["suites"]["custom_tutoring"] = score_tutoring(
        model, tok, tut, args.limit, name="custom_tutoring"
    )

    report["forget_proxy"] = {
        "en_stem_acc": report["suites"]["en_stem_holdout"]["acc"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v.get("acc") if isinstance(v, dict) else v for k, v in report["suites"].items()}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
