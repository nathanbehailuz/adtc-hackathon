# CPT source audit (v5 planning)

Based on existing `cpt_mix_v3.jsonl` and on-disk CPT pools. v5 remixes these pools with a STEM-heavier recipe; it does **not** reproduce AfriqueLLM-scale CPT.

## cpt_mix_v3

- total rows: **20000**
- approx chars: 45,589,961
- approx tokens (~4 chars/tok): ~11,397,490

| source | n | share |
|--------|--:|------:|
| `fineweb2_amh` | 7816 | 39.1% |
| `afrinllb` | 5000 | 25.0% |
| `en_stem_sft_v2` | 4000 | 20.0% |
| `wikipedia_amharic` | 3184 | 15.9% |

## On-disk pools

| path | n |
|------|--:|
| `data/train/cpt/native_am/fineweb2_amh_100m_v0.jsonl` | 50000 |
| `data/train/cpt/native_am/wikipedia_amharic_v0.jsonl` | 20000 |
| `data/train/cpt/parallel/afrinllb_v0.jsonl` | 20000 |
| `data/train/am_stem_sft_nllb_v2_filtered.jsonl` | 4000 |
| `data/train/en_stem_sft_v4.jsonl` | 2000 |

## v5 CPT targets (quality > quota)

- ~40% native Amharic
- ~25% Amharic STEM / parallel edu
- ~20% English STEM replay
- ~10% structured EN STEM
- ~5% replay fill
- total target ~20–30k documents
- dedup against frozen `data/eval/*_v0.jsonl`

## Overlap / notes

- Frozen eval never used for training (enforced by `eval/dedup_against_eval.py`).
- AfriqueLLM GSM8K Hub id remains unavailable; AM STEM continues to use NLLB-filtered + simonbutt (capped in SFT).

