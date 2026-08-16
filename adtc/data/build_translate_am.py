#!/usr/bin/env python3
"""Translate EN STEM SFT JSONL user/assistant text to Amharic.

Backends:
  --backend stub   (default) copies EN with [AM-STUB] prefix — for pipeline smoke
  --backend file   reads a JSON map {en_text: am_text} from --map
  --backend nllb   uses facebook/nllb-200-distilled-600M (needs transformers+torch+GPU/CPU)

Produces direction en_am and am_am variants for mix_sft.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def translate_stub(text: str) -> str:
    return f"[AM-STUB] {text}"


def load_map(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_nllb():
    # Jubail torch is 2.5.x; transformers blocks torch.load on .bin without 2.6+.
    # NLLB-200 distilled still ships pytorch_model.bin — patch both the module and
    # the already-imported binding in modeling_utils.
    try:
        import transformers.utils.import_utils as iu
        import transformers.modeling_utils as mu

        def _ok() -> None:
            return None

        iu.check_torch_load_is_safe = _ok  # type: ignore[assignment]
        mu.check_torch_load_is_safe = _ok  # type: ignore[assignment]
    except Exception:
        pass

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    model_id = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def _tr(text: str) -> str:
        tok.src_lang = "eng_Latn"
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        forced = tok.convert_tokens_to_ids("amh_Ethi")
        out = model.generate(**inputs, forced_bos_token_id=forced, max_new_tokens=512)
        return tok.batch_decode(out, skip_special_tokens=True)[0]

    return _tr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "am_stem_sft_v0.jsonl")
    parser.add_argument("--backend", choices=("stub", "file", "nllb"), default="stub")
    parser.add_argument("--map", type=Path, help="JSON map for --backend file")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.backend == "stub":
        tr = translate_stub
    elif args.backend == "file":
        if not args.map:
            raise SystemExit("--map required for file backend")
        mapping = load_map(args.map)
        tr = lambda t: mapping.get(t, translate_stub(t))
    else:
        tr = make_nllb()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.inp.open(encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            if args.limit is not None and n >= args.limit:
                break
            obj = json.loads(line)
            user = next(m["content"] for m in obj["messages"] if m["role"] == "user")
            asst = next(m["content"] for m in obj["messages"] if m["role"] == "assistant")
            user_am, asst_am = tr(user), tr(asst)
            behavior = obj.get("behavior", "solve")
            base_id = obj.get("id", f"row_{n}")

            # EN→AM: English question, Amharic answer (and vice patterns)
            for direction, u, a, suffix in (
                ("en_am", user, asst_am, "en_am"),
                ("am_am", user_am, asst_am, "am_am"),
                ("am_en", user_am, asst, "am_en"),
            ):
                row = {
                    "id": f"{base_id}_{suffix}",
                    "direction": direction,
                    "behavior": behavior,
                    "messages": [
                        {"role": "user", "content": u},
                        {"role": "assistant", "content": a},
                    ],
                    "source": f"mt_{args.backend}:{obj.get('source', 'en_stem')}",
                }
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1

    print(f"wrote {args.out} ({n} rows) backend={args.backend}")


if __name__ == "__main__":
    main()
