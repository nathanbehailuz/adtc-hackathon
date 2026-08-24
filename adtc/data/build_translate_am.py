#!/usr/bin/env python3
"""Translate EN STEM SFT JSONL user/assistant text to Amharic.

Backends:
  --backend stub   (default) copies EN with [AM-STUB] prefix — for pipeline smoke
  --backend file   reads a JSON map {en_text: am_text} from --map
  --backend nllb   uses facebook/nllb-200-distilled-600M (needs transformers+torch+GPU/CPU)

Math-marker hardening (--backend nllb / all):
  Protects ``#### N`` and ``<<expr=val>>`` with placeholders before MT, restores after.
  Optional --quality-filter drops am_am rows lacking Ethiopic or (for solve) ####.

Produces direction en_am and am_am variants for mix_sft (am_en optional via --directions).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CALC_RE = re.compile(r"<<[^>]+>>")
HASH_RE = re.compile(r"####\s*-?\d+(?:\.\d+)?")
ETHIOPIC_RE = re.compile(r"[\u1200-\u137F]")


def translate_stub(text: str) -> str:
    return f"[AM-STUB] {text}"


def load_map(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def protect_math(text: str) -> tuple[str, list[str]]:
    """Replace #### and <<>> spans with placeholders; return (masked, spans)."""
    spans: list[str] = []

    def _sub(m: re.Match[str]) -> str:
        spans.append(m.group(0))
        return f"⟦M{len(spans) - 1}⟧"

    out = CALC_RE.sub(_sub, text)
    out = HASH_RE.sub(_sub, out)
    return out, spans


def restore_math(text: str, spans: list[str]) -> str:
    out = text
    for i, span in enumerate(spans):
        for tok in (f"⟦M{i}⟧", f"[M{i}]", f"<<M{i}>>", f"M{i}"):
            if tok in out:
                out = out.replace(tok, span)
                break
    # If #### vanished entirely but we had one, append English marker.
    for span in spans:
        if span.startswith("####") and "####" not in out:
            out = out.rstrip() + "\n" + span
    return out


def looks_amharic(text: str) -> bool:
    return bool(ETHIOPIC_RE.search(text or ""))


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
        masked, spans = protect_math(text)
        tok.src_lang = "eng_Latn"
        inputs = tok(masked, return_tensors="pt", truncation=True, max_length=512).to(device)
        forced = tok.convert_tokens_to_ids("amh_Ethi")
        out = model.generate(**inputs, forced_bos_token_id=forced, max_new_tokens=512)
        decoded = tok.batch_decode(out, skip_special_tokens=True)[0]
        return restore_math(decoded, spans)

    return _tr


def pass_quality(direction: str, behavior: str, user: str, asst: str) -> bool:
    if direction.startswith("am"):
        if not looks_amharic(user) and not looks_amharic(asst):
            return False
    if direction == "am_am" and behavior in ("solve", "explain"):
        if "####" not in asst and not looks_amharic(asst):
            return False
    if direction == "am_am" and not looks_amharic(user):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="inp", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "am_stem_sft_v0.jsonl")
    parser.add_argument("--backend", choices=("stub", "file", "nllb"), default="stub")
    parser.add_argument("--map", type=Path, help="JSON map for --backend file")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--directions",
        default="en_am,am_am",
        help="Comma list of directions to emit (default en_am,am_am; add am_en if needed)",
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Drop low-quality AM rows (no Ethiopic / missing #### on solve)",
    )
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

    wanted = {d.strip() for d in args.directions.split(",") if d.strip()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_out = 0
    n_drop = 0
    with args.inp.open(encoding="utf-8") as src, args.out.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            if args.limit is not None and n_in >= args.limit:
                break
            obj = json.loads(line)
            n_in += 1
            user = next(m["content"] for m in obj["messages"] if m["role"] == "user")
            asst = next(m["content"] for m in obj["messages"] if m["role"] == "assistant")
            user_am, asst_am = tr(user), tr(asst)
            # Prefer English math markers on assistant if MT dropped them.
            if "####" in asst and "####" not in asst_am:
                m = HASH_RE.search(asst)
                if m:
                    asst_am = asst_am.rstrip() + "\n" + m.group(0)
            for calc in CALC_RE.findall(asst):
                if calc not in asst_am:
                    asst_am = asst_am.replace(calc, calc)  # no-op; restored via protect ideally
            behavior = obj.get("behavior", "solve")
            base_id = obj.get("id", f"row_{n_in}")

            candidates = {
                "en_am": (user, asst_am, "en_am"),
                "am_am": (user_am, asst_am, "am_am"),
                "am_en": (user_am, asst, "am_en"),
            }
            for direction in wanted:
                if direction not in candidates:
                    continue
                u, a, suffix = candidates[direction]
                if args.quality_filter and not pass_quality(direction, behavior, u, a):
                    n_drop += 1
                    continue
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
                n_out += 1

    print(
        f"wrote {args.out} ({n_out} rows) backend={args.backend} "
        f"in={n_in} dropped={n_drop} directions={sorted(wanted)}"
    )


if __name__ == "__main__":
    main()
