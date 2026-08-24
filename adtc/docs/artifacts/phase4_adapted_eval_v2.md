# Adapted HF eval — SFT mix v2

GPU / transformers on `*_merged_v2`. Compare to Phase 4 v0 table in RESULTS_REPORT.
Not a substitute for CPU llama.cpp GGUF packaging scores.

| model | am MGSM | en MGSM | am MMLU | en holdout | tutoring |
|-------|--------:|--------:|--------:|-----------:|---------:|
| `gemma3_4b_merged_v2` | 0.204 | 0.328 | 0.218 | 0.32 | 0.8811881188118812 |
| `qwen3_1_7b_merged_v2` | 0.036 | 0.356 | 0.0 | 0.34 | 1.0 |

