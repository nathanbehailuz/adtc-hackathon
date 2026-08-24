# Amharic review checklist — mix v3 (Nathan)

## Files

- Stratified sample: [`amharic_review_sample_v3.jsonl`](./amharic_review_sample_v3.jsonl) (32 Ethiopic rows from `sft_mix_v3`)
- Live outputs: after GGUF exists, `MODEL=7 PROMPTS=6,7,8,9,10 sbatch 11_try_prompt.sbatch` → `adtc/logs/try_prompt/`

## Checks per row / reply

1. Readable Amharic (not gibberish)
2. STEM terminology acceptable for school use
3. Numbers / math notation intact
4. Tutoring register OK
5. Direction matches text (`am_am` / `en_am` / …)

## Claim

Keep `african_alpha_claim: true` only if sample + live Gemma v3 outputs pass. Log pass/fail counts in DEVLOG; do not flip metadata until review is done.
