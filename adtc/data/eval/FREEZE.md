# Eval freeze v0 — 16 Aug 2026

**Rule:** Do not edit these files in place. Any change requires a new version (`v1`) and an updated manifest / DEVLOG entry.

**Never train on these files.** Dedup train mixes against them via `adtc/eval/dedup_against_eval.py`.

## Frozen files

| File | Rows (approx.) | Role |
|------|----------------|------|
| `afrimgsm_amh_test_v0.jsonl` | 250 | Amharic math (Iroko) |
| `afrimgsm_eng_test_v0.jsonl` | 250 | English math control / translate-test |
| `afrimmlu_amh_test_v0.jsonl` | 500 | Amharic knowledge |
| `afrixnli_amh_test_v0.jsonl` | 600 | Amharic NLI |
| `en_stem_holdout_v0.jsonl` | 100 | EN STEM forgetting check (GSM8K test) |
| `custom_tutoring_v0.jsonl` | ~100 | Product tutoring behaviors EN↔am |
| `fertility_parallel_v0.jsonl` | 10 | Tokenizer fertility pairs |
| `eval_manifest_v0.json` | — | SHA256 for Iroko exports |

## Amharic review

Nathan Behailu (fluent validator) should spot-check a stratified sample of Amharic custom tutoring rows before Gate 1 submit. Issues found → fix in `custom_tutoring_v1`, do not silently rewrite v0.

## Fertility metrics

Set + script frozen; run `python eval/fertility.py --tokenizer …` in Phase 2 (tokenizer on cloud). Metrics are not part of Gate 1a.
