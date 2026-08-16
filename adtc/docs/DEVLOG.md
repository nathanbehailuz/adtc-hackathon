# Dev log

## 2026-08-16 — Phase 0 Gate 0 (Packaging Truth, Amharic)

### Outcome
**Gate 0 passed** on packaging: download → path match → `adtc-profiler` participant smoke (`--skip-accuracy`) produces TPS / RSS / thermal telemetry. Do not start fine-tuning until this remains green after any packaging edits.

### 0.1 Submission repo
- Public URL: https://github.com/nathanbehailuz/adtc-2026-submission-template (HTTP 200)
- Local remotes: `origin` (team fork), `upstream` (official template)
- `HEAD` aligned with `origin/main` and `upstream/main` (0/0 divergence) at start of Phase 0
- **Note:** Phase 0 `metadata.json` edits are local until pushed; clean-clone packaging test used the local draft metadata + `download_model.sh`

### 0.2 Metadata draft
- Domain: `math_scientific_reasoning`
- Languages: `["en", "am"]`
- `african_alpha_claim: true` (draft; Phase 1 must assign Amharic validator)
- Submitter / `team_id`: `REPLACE_ME_*` placeholders
- Smoke model still SmolLM2-135M Q4_K_M (packaging only)
- PRD language lock updated to Amharic

### 0.3 Tooling freeze
See [`TOOLING.md`](./TOOLING.md).

| Tool | Pin / location |
|------|----------------|
| adtc-profiler | commit `ac2e137dca65ea3b09d997774f17dd8907b489fb` in `adtc/tools/adtc-profiler-venv` |
| llama-bench | llama.cpp release **b10451** at `adtc/tools/llama.cpp/llama-b10451/llama-bench` |
| Python | 3.11.1 |

Host Homebrew install of llama.cpp was skipped (Rosetta/approval friction); workspace binaries used instead.

### 0.4 Participant smoke (working tree)
Command:

```bash
export PATH="$(pwd)/adtc/tools/llama.cpp/llama-b10451:$PATH"
source adtc/tools/adtc-profiler-venv/bin/activate
bash adtc-2026-submission-template/download_model.sh
adtc-profiler run \
  --submission "$(pwd)/adtc-2026-submission-template" \
  --mode participant \
  --skip-accuracy \
  --output adtc/docs/artifacts/phase0_submission_smoke.json
```

Key metrics (`phase0_submission_smoke.json`):

| Metric | Value |
|--------|-------|
| gen TPS | 86.45 |
| TTFT | 1312.71 ms |
| peak RSS | 270.58 MB |
| steady RSS | 220.73 MB |
| throttled | false |

### 0.5 Clean-path verification
- Fresh clone to `/tmp/adtc-phase0-clean-*`
- Copied local Phase 0 `metadata.json` + `download_model.sh` (remote still has pre-Phase-0 template until push)
- `_runtime.model_path` → `model/SmolLM2-135M-Instruct-Q4_K_M.gguf` present after `bash download_model.sh`
- Profiler output: `adtc/docs/artifacts/phase0_clean_clone_smoke.json`

| Metric | Value |
|--------|-------|
| gen TPS | 81.7 |
| TTFT | 707.34 ms |
| peak RSS | 272.39 MB |
| steady RSS | 221.52 MB |
| throttled | false |

### 0.6 Hosting decision
**Final weights:** public Hugging Face GGUF (credential-free). Backup: GitHub Release asset. Documented in TOOLING.md. Phase 0 retains SmolLM2 HF URL.

### Follow-ups before / during Phase 1
1. Push Phase 0 `metadata.json` to `origin` so a pure remote clone matches the Amharic STEM draft
2. Replace `REPLACE_ME_ADTF_TEAM_ID` with real `team_id` (submitter fields filled: Nathan Behailu / nz2212@nyu.edu / nathanbehailuz)
3. Amharic fluent validator: **Nathan Behailu**
4. Keep profiler on pinned SHA for all later scores

---

## 2026-08-16 — Data + fine-tune scaffold (models deferred)

### Schedule shift
- **Phase 2** (GGUF / HF model pulls + baseline screen) deferred to **tomorrow / cloud**
- **Today:** Phase 1 eval freeze scripts, Phase 3 dataset builders, Phase 4 TRL+PEFT QLoRA training code
- Train on **cloud** with HF checkpoints; GGUF only for profiler/submission

### Artifacts added
- `adtc/docs/DATASETS.md` — source catalog
- `adtc/data/`, `adtc/eval/`, `adtc/training/` — pipelines and QLoRA scripts
- PRD updated with current working order + Amharic Phase 1 lock
