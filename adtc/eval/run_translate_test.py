#!/usr/bin/env python3
"""Direct-Amharic vs English translate-test on frozen AfriMGSM pairs.

Supports HF checkpoints (--model) or local GGUF (--gguf) via llama-cpp-python.
"""
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


def generate_hf(model, tok, prompt: str, max_new: int = 64) -> str:
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


def generate_gguf(llm, prompt: str, max_new: int = 64) -> str:
    try:
        out = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new,
            temperature=0.0,
        )
        return out["choices"][0]["message"]["content"] or ""
    except Exception:  # noqa: BLE001
        out = llm(prompt, max_tokens=max_new, temperature=0.0, echo=False)
        return out["choices"][0]["text"] or ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="HF model id or local merged path")
    ap.add_argument("--gguf", type=Path, default=None, help="Local GGUF path (llama-cpp)")
    ap.add_argument("--am", type=Path, default=ROOT / "data/eval/afrimgsm_amh_test_v0.jsonl")
    ap.add_argument("--en", type=Path, default=ROOT / "data/eval/afrimgsm_eng_test_v0.jsonl")
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap aligned AfriMGSM pairs (default: all)",
    )
    ap.add_argument("--n-threads", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/artifacts/phase2_translate_test_v0.md")
    args = ap.parse_args()

    if bool(args.model) == bool(args.gguf):
        raise SystemExit("Provide exactly one of --model or --gguf")

    am_rows = load_jsonl(args.am)
    en_rows = load_jsonl(args.en)
    n = min(len(am_rows), len(en_rows))
    if args.limit is not None:
        n = min(n, args.limit)
    am_rows = am_rows[:n]
    en_rows = en_rows[:n]

    label: str
    if args.gguf:
        from llama_cpp import Llama

        gguf = args.gguf.resolve()
        if not gguf.is_file():
            raise SystemExit(f"GGUF not found: {gguf}")
        kwargs = {"model_path": str(gguf), "n_ctx": 2048, "verbose": False}
        if args.n_threads is not None:
            kwargs["n_threads"] = args.n_threads
        llm = Llama(**kwargs)
        label = str(gguf)
        generate_fn = lambda p: generate_gguf(llm, p)
    else:
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
        label = args.model
        generate_fn = lambda p: generate_hf(model, tok, p)

    direct_ok = 0
    translate_ok = 0
    rows_out = []
    for i in range(n):
        am = am_rows[i]
        en = en_rows[i]
        gold = normalize_ans(gold_answer(am) or gold_answer(en))
        am_pred = normalize_ans(generate_fn(am["question"]))
        en_pred = normalize_ans(generate_fn(en["question"]))
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
        "# Direct Amharic vs English translate-test",
        "",
        f"- Model: `{label}`",
        f"- N: {n} (AfriMGSM amh vs eng, index-aligned)",
        f"- Direct Amharic accuracy: **{direct_acc:.3f}** ({direct_ok}/{n})",
        f"- English translate-test accuracy: **{translate_acc:.3f}** ({translate_ok}/{n})",
        f"- Gap (translate − direct): **{gap:+.3f}**",
        "",
    ]
    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    payload = {
        "model": label,
        "n": n,
        "direct_acc": direct_acc,
        "translate_acc": translate_acc,
        "gap": gap,
        "rows": rows_out,
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"direct_acc": direct_acc, "translate_acc": translate_acc, "gap": gap, "n": n}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
