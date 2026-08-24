# Adapted HF eval — SFT mix v3 (Gemma CPT→SFT)

GPU / transformers on `gemma3_4b_merged_v3`. Compare to v2: am MGSM 0.204, translate direct 0.12.
Not a substitute for CPU llama.cpp GGUF packaging scores.

| model | am MGSM | en MGSM | am MMLU | en holdout | tutoring |
|-------|--------:|--------:|--------:|-----------:|---------:|
| `gemma3_4b_merged_v3` | 0.232 | 0.336 | 0.208 | 0.34 | 0.9108910891089109 |

