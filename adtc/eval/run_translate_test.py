#!/usr/bin/env python3
"""Direct-Amharic vs English translate-test on frozen AfriMGSM pairs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    # take last number-like token
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    if nums:
        return nums[-1]
    return s.split()[-1] if s.split() else s


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def generate(model, tok, prompt: str, max_new: int = 64) -> str:
    import torch

    messages = [{"role": "user", "content": prompt}]
    if hasattr(tok, "apply_chat_template"):
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tok(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[-1] :]
    return tok.decode(gen, skip_special_tokens=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF model id or local merged path")
    ap.add_argument("--am", type=Path, default=ROOT / "data/eval/afrimgsm_amh_test_v0.jsonl")
    ap.add_argument("--en", type=Path, default=ROOT / "data/eval/afrimgsm_eng_test_v0.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/artifacts/phase2_translate_test_v0.md")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    am_rows = load_jsonl(args.am)[: args.limit]
    en_rows = load_jsonl(args.en)[: args.limit]
    # align by index (same suite order)
    n = min(len(am_rows), len(en_rows), args.limit)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")

    direct_ok = 0
    translate_ok = 0
    rows_out = []
    for i in range(n):
        am = am_rows[i]
        en = en_rows[i]
        gold = normalize_ans(gold_answer(am) or gold_answer(en))
        am_prompt = am["question"]
        # translate-test: English question (same item), model answers in EN
        en_prompt = en["question"]
        am_pred = normalize_ans(generate(model, tok, am_prompt))
        en_pred = normalize_ans(generate(model, tok, en_prompt))
        d_hit = bool(gold) and am_pred == gold
        t_hit = bool(gold) and en_pred == gold
        direct_ok += int(d_hit)
        translate_ok += int(t_hit)
        rows_out.append(
            {
                "i": i,
                "gold": gold,
                "direct_am_pred": am_pred,
                "translate_en_pred": en_pred,
                "direct_ok": d_hit,
                "translate_ok": t_hit,
            }
        )

    direct_acc = direct_ok / max(1, n)
    translate_acc = translate_ok / max(1, n)
    gap = translate_acc - direct_acc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# Phase 2 — Direct Amharic vs English translate-test",
        "",
        f"- Model: `{args.model}`",
        f"- N: {n} (AfriMGSM amh vs eng, index-aligned)",
        f"- Direct Amharic accuracy: **{direct_acc:.3f}** ({direct_ok}/{n})",
        f"- English translate-test accuracy: **{translate_acc:.3f}** ({translate_ok}/{n})",
        f"- Gap (translate − direct): **{gap:+.3f}**",
        "",
        "| i | gold | direct_am | translate_en | direct_ok | translate_ok |",
        "|---|------|-----------|--------------|-----------|--------------|",
    ]
    for r in rows_out[:20]:
        md.append(
            f"| {r['i']} | {r['gold']} | {r['direct_am_pred']} | {r['translate_en_pred']} | {r['direct_ok']} | {r['translate_ok']} |"
        )
    md.append("")
    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    json_path = args.out.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "n": n,
                "direct_acc": direct_acc,
                "translate_acc": translate_acc,
                "gap": gap,
                "rows": rows_out,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"direct_acc": direct_acc, "translate_acc": translate_acc, "gap": gap, "n": n}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
