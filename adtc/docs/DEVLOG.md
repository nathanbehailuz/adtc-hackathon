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

---

## 2026-08-16 — Phase 1 Gate 1a (Language + eval freeze)

### Outcome
**Gate 1a passed.** Frozen eval under `adtc/data/eval/`; no training data generated in this phase.

### Language (1.1–1.3)
- Doc: [`LANGUAGE.md`](./LANGUAGE.md)
- Amharic / Nathan Behailu; `african_alpha_claim: true` (contingent on pre-submit review)

### Frozen eval counts
| File | Rows |
|------|------|
| `afrimgsm_amh_test_v0.jsonl` | 250 |
| `afrimgsm_eng_test_v0.jsonl` | 250 |
| `afrimmlu_amh_test_v0.jsonl` | 500 |
| `afrixnli_amh_test_v0.jsonl` | 600 |
| `en_stem_holdout_v0.jsonl` | 100 |
| `custom_tutoring_v0.jsonl` | 101 |
| `fertility_parallel_v0.jsonl` | 10 |

Manifest: `eval_manifest_v0.json` (Iroko SHA256). Freeze rules: `adtc/data/eval/FREEZE.md`.

### Amharic spot-check
Nathan should review a stratified sample of Amharic rows in `custom_tutoring_v0.jsonl` (terminology, register, algebra). Log issues here; fix via `v1` if needed—do not rewrite v0 silently.

### Fertility metrics
Deferred to Phase 2: `python eval/fertility.py --tokenizer <HF_ID>` once bases are on cloud.

### Next
Phase 3: build SFT mixes with existing builders; dedup against frozen eval. Phase 2 model screen tomorrow / on cloud.

---

## 2026-08-16 — Pipeline run logging (HPC-ready)

### Why
Train-source download was stopped mid-run (FineWeb2 OK; gated news FAIL; Wikipedia in flight). Without incremental logs the manifest never flushed, so success/fail was unclear.

### What
- Shared logger: `adtc/lib/run_log.py` → `adtc/logs/<stage>/{*.log,*.jsonl,*.summary.json}`
- Wired into: `download_train_sources`, `normalize_cpt_sources`, `normalize_sft_sources`, `mix_sft`, `download_base_models`, `train_sft_qlora`, `train_cpt_qlora`, `merge_lora`
- Download also rewrites `data/raw/download_manifest_v0.json` **after each source** and on Ctrl+C
- Doc: [`RUNLOGS.md`](./RUNLOGS.md); catalog note in [`DATASETS.md`](./DATASETS.md)

### Known from partial local download
| Key | Status |
|-----|--------|
| `fineweb2_amh_100m` | OK (50k snapshot) |
| `amharic_news` | FAIL — gated HF dataset (needs `HF_TOKEN`) |
| later sources | not finished (interrupted) |

Tomorrow on HPC: re-run download (or `--only` remaining keys), then normalize → mix → `training/download_base_models.py` → SFT.

---

## 2026-08-16 — Jubail HPC: Slurm pipeline + Phase 3 data (in progress)

### Outcome
Repo runs from `$SCRATCH` (`/scratch/nz2212/adtc-hackathon`). Slurm jobs under [`adtc/hpc/`](../hpc/); root [`README.md`](../../README.md). Env setup **succeeded**. Train download **partial OK**. Prep job advanced through SFT normalize + EN STEM; **mix still failing** until JSONL reader fix (see below). Model download job ready (`03_download_models.sbatch` → Qwen3-1.7B + 4B).

### Slurm / env
- Added: `setup_env.sbatch`, `01`–`05` (+ `04b`), `env.sh`, `submit_chain.sh`, [`hpc/README.md`](../hpc/README.md)
- Conda: `/scratch/nz2212/adtc-hackathon/adtc/training/.conda-env` (CUDA torch + `training/requirements.txt`)
- Fixes that mattered on Jubail:
  - Use **`SLURM_SUBMIT_DIR`** (not `BASH_SOURCE` — spool copy is not writable)
  - **`set +u` around conda activate** (`MKL_INTERFACE_LAYER` unbound under `set -u`)
  - Explicit `conda activate …/training/.conda-env` in job scripts
  - `#SBATCH -o/-e logs/%x-%j.*` — check **`.out` for OK/FAIL**; `.err` is mostly HF/tqdm stderr

### Job results (selected)

| Job | Result |
|-----|--------|
| `adtc_setup_env-17259662` | **OK** — conda env created |
| `adtc_dl_train-17259751` | **partial** — 8 OK, 2 FAIL, 2 SKIP |
| `adtc_prep_data-17259783` | FAIL — CPT finished then HF datasets **GIL abort** on exit; `set -e` stopped pipeline |
| `adtc_prep_data-17259791` | CPT soft-continued; SFT/EN STEM **OK**; **mix FAIL** |

### Download train (`01`) status
| Key | Status |
|-----|--------|
| `fineweb2_amh_100m` | OK (50k snapshot) |
| `wikipedia_amharic` | OK |
| `afrinllb` | OK |
| `walia` | OK |
| `finetome_am` | OK |
| `r1_multilingual` | OK (2.4k after filter) |
| `dolly_am` | OK |
| `taco_am` | OK |
| `amharic_news` | FAIL — gated (`HF_TOKEN` + Hub access) |
| `afriquellm_gsm8k` | FAIL — missing / inaccessible on Hub |
| `fineweb2_full`, `yoseali` | SKIP (by design) |

Snapshots: `adtc/data/raw/snapshots/*.jsonl`.

### Prep / normalize
- CPT outputs present for fineweb / wikipedia / afrinllb (`amharic_news` skipped/gated)
- SFT sources: `walia_sft_v0` (~122k), `finetome_am_sft_v0` (~83k), `dolly_am`, `taco_am`; `r1_am` empty after Amharic filter
- `en_stem_sft_v0.jsonl` (2000) + stub `am_stem_sft_v0.jsonl` written
- **`sft_mix_v0.jsonl` not written yet**

### Bugs fixed (code)
1. HF `datasets` teardown → `PyGILState_Release` abort after successful CPT — soft-continue in `02_prepare_data.sbatch`; normalize prefers **local snapshots** (no HF re-stream when snapshot exists)
2. `mix_sft.read_jsonl` used `read_text().splitlines()` — Unicode line separators inside Amharic strings → `JSONDecodeError`. Now reads **physical `\n` only** (same for `dedup_against_eval` eval load)

### Next
1. Re-run `sbatch 02_prepare_data.sbatch` (or mix-only wrap) → confirm `data/train/sft_mix_v0.jsonl`
2. `sbatch 03_download_models.sbatch` for Qwen3-1.7B + 4B (after env)
3. Then `04_train_sft_1_7b.sbatch` (A100 / `bf16`)
4. Optional: `HF_TOKEN` for gated `amharic_news`; replace stub MT with real Amharic before serious train
5. Keep updating this DEVLOG after each meaningful HPC/data/train milestone

---

## 2026-08-16 — Base model download OK (Jubail)

### Outcome
Job `adtc_dl_models-17259908` **DONE ok** — both Phase 2/4 HF bases on scratch.

| Key | HF id | Status |
|-----|-------|--------|
| `qwen3_1_7b` | `Qwen/Qwen3-1.7B` | OK |
| `qwen3_4b` | `Qwen/Qwen3-4B` | OK |

Cache: `adtc/data/raw/hf_home/models--Qwen--Qwen3-*`  
Manifest: `adtc/training/model_download_manifest_v0.json`  
Logs: `adtc/hpc/logs/adtc_dl_models-17259908.{out,err}`, `adtc/logs/download_models/`

### Next
1. Finish `sft_mix_v0.jsonl` (`02_prepare_data` / mix-only) if not already written
2. `sbatch 04_train_sft_1_7b.sbatch` (A100)

---

## 2026-08-16 — Expand HF model download list (PRD Phase 2)

### Outcome
`training/download_base_models.py` + `03_download_models.sbatch` now pull the full Phase 2 screen shortlist (not only Qwen3).

| Key | HF id | Role |
|-----|-------|------|
| `qwen3_1_7b` | `Qwen/Qwen3-1.7B` | efficiency (already cached) |
| `qwen3_4b` | `Qwen/Qwen3-4B` | accuracy (already cached) |
| `gemma3_4b` | `google/gemma-3-4b-it` | architecture control (**gated** — needs HF license + `HF_TOKEN`) |
| `qwen25_3b_instruct` | `Qwen/Qwen2.5-3B-Instruct` | middle-size control |
| `qwen35_2b` | `Qwen/Qwen3.5-2B` | optional llama.cpp compat check |
| `qwen35_4b` | `Qwen/Qwen3.5-4B` | optional llama.cpp compat check |

### Next
1. `cd adtc/hpc && sbatch 03_download_models.sbatch` (export `HF_TOKEN` first for Gemma)
2. Confirm `logs/download_models/latest.summary.json` ok for new keys

---

## 2026-08-16 — Load HF_TOKEN from adtc/.env in Slurm jobs

### Outcome
`adtc/.env` holds `HF_TOKEN` (gitignored). [`adtc/hpc/env.sh`](../hpc/env.sh) now sources it and sets `HUGGING_FACE_HUB_TOKEN` so gated Hub pulls (Gemma, news) work in jobs without exporting on the login shell.

### Next
1. Accept Gemma license on HF if not done
2. `sbatch 03_download_models.sbatch` — expect `[env] HF_TOKEN=set` in the `.out`

---

## 2026-08-16 — Model download partial (Gemma gated)

### Outcome
Job `adtc_dl_models-17260204`: **partial** `ok=5 error=1`. Token loaded (`HF_TOKEN=set`).

| Key | Status |
|-----|--------|
| qwen3_1_7b | OK (cache) |
| qwen3_4b | OK (cache) |
| gemma3_4b (`google/gemma-3-4b-it`) | **FAIL 403** — not on authorized list / license not accepted for this HF account |
| qwen25_3b_instruct | OK |
| qwen35_2b | OK |
| qwen35_4b | OK |

### Next
1. On Hugging Face (same account as the token): open https://huggingface.co/google/gemma-3-4b-it → **Acknowledge license / ask for access**
2. Re-run: `sbatch 03_download_models.sbatch` or  
   `python training/download_base_models.py --only gemma3_4b`
3. Proceed with SFT on Qwen3 once `sft_mix_v0.jsonl` exists (Gemma can wait)

---

## 2026-08-16 — Note: cancel Slurm jobs with scancel

`kill <JOBID>` does not cancel queue jobs (no local PID). Use:

```bash
scancel 17260232          # one job
scancel -u $USER          # all your jobs
squeue -u $USER           # confirm gone
```

Pending (`PD`, reason `Priority`) means waiting in the nvidia queue — not running yet.

---

## 2026-08-16 — Single Slurm job to train all SFT configs

### Outcome
Added [`adtc/hpc/04_train_sft_all.sbatch`](../hpc/04_train_sft_all.sbatch): one A100 job, 48h, trains in order:

1. `qlora_qwen3_1_7b.yaml`
2. `qlora_qwen3_4b.yaml`
3. `qlora_qwen25_3b_instruct.yaml` (new)
4. `qlora_gemma3_4b.yaml` (new; needs Gemma download / HF access)

Continues on per-model failure so Gemma gate does not block Qwen runs. Qwen3.5 excluded (compat check only).

`sft_mix_v0.jsonl` is present on scratch (~3.5 MB).

### Next
```bash
cd /scratch/nz2212/adtc-hackathon/adtc/hpc
sbatch 04_train_sft_all.sbatch
# cancel: scancel <jobid>
```

---

## 2026-08-16 — Fix SFTConfig warmup for new transformers/TRL

### Outcome
Job `adtc_sft_all-17260256` failed all 4 configs with:
`TypeError: SFTConfig.__init__() got an unexpected keyword argument 'warmup_ratio'`

Installed transformers uses `warmup_steps` only (float in `[0,1)` = ratio). Updated `train_sft_qlora.py` and `train_cpt_qlora.py` to map `warmup_ratio` from YAML → `warmup_steps`.

Models loaded and data mapped fine before the crash (A100, mix 2000 rows).

### Next
```bash
cd /scratch/nz2212/adtc-hackathon/adtc/hpc
sbatch 04_train_sft_all.sbatch
```

---

## 2026-08-16 — SFT all models OK; merge script covers all four

### Outcome
Job `adtc_sft_all-17260420`: **failed=0 / total=4**. Adapters under `training/runs/*/adapter`:

| Run | Base |
|-----|------|
| `qwen3_1_7b_qlora_v0` | Qwen/Qwen3-1.7B |
| `qwen3_4b_qlora_v0` | Qwen/Qwen3-4B |
| `qwen25_3b_instruct_qlora_v0` | Qwen/Qwen2.5-3B-Instruct |
| `gemma3_4b_qlora_v0` | google/gemma-3-4b-it |

`05_merge_lora.sbatch` previously merged **only** 1.7B; updated to merge all four sequentially on `compute`.

### Next
```bash
cd /scratch/nz2212/adtc-hackathon/adtc/hpc
sbatch 05_merge_lora.sbatch
```

---

## 2026-08-16 — Merge all four OK; plan status check

### Outcome
Job `adtc_merge_all-17260554`: **failed=0 / total=4**. Merged HF folders:

- `runs/qwen3_1_7b_merged_v0`
- `runs/qwen3_4b_merged_v0`
- `runs/qwen25_3b_instruct_merged_v0`
- `runs/gemma3_4b_merged_v0`

### PRD progress (not all done)
| Phase | Status |
|-------|--------|
| 0 Packaging | Done |
| 1 Eval freeze | Done |
| 2 Baseline GGUF screen + fertility + Gate 2 finalists | **Open** (HF bases yes; profiler/GGUF screen no) |
| 3 Bilingual data | **Partial** (mix exists; stub MT; no fluent review / real NLLB) |
| 4 Adapt | **Partial** (QLoRA+merge done; post-adapt eval / CPT decision open) |
| 5 GGUF PTQ + Pareto | **Not started** |
| 6 Ablations | Optional |
| 7 Package / REPORT / demo | **Not started** |

### Next (critical path to Gate 1)
1. Convert merged HF → GGUF → Q8/Q4; profile (Phase 5)
2. Or finish Phase 2 unadapted GGUF screen first if still needed for finalist pick
3. Replace stub Amharic MT before claiming strong am STEM; Nathan review samples
4. Package winner into submission template

---

## 2026-08-16 — ADTC profiler in scratch conda (HPC)

### Outcome
Job `adtc_setup_profiler-17260598` **COMPLETED** (exit 0). Pinned `adtc-profiler` (`ac2e137…`) + `llama-cpp-python` 0.3.34 installed into `adtc/training/.conda-env`. Login-node pip failed earlier (no `g++` / `ninja posix_spawn`); compute job with `gcc/12.2.0` + `cmake/3.31.2` worked. Docs: `hpc/setup_profiler.sbatch`, `TOOLING.md` HPC section.

### Next
1. Install linux llama.cpp **b10451** binaries under `adtc/tools/` and put `llama-bench` on PATH for jobs
2. Run fertility + unadapted GGUF profiler screen on Slurm

---

## 2026-08-16 — Phase 2 screen + Gate 2 (HPC)

### Outcome
Built Jubail-native llama.cpp **b10451** (`setup_build_llama_cpp.sbatch` / job `17260729`). Fertility + unadapted GGUF profiler + translate-test completed on Slurm.

### What
| Item | Result |
|------|--------|
| Fertility | `docs/artifacts/fertility_v0.json` — Qwen3/Qwen2.5 **severe** Amharic fragmentation (`mean_R_am_en≈4.3`); Gemma 3 4B **ok** (`≈2.1`) |
| Unadapted GGUFs | All 8 downloaded (incl. Unsloth Qwen3-1.7B + Qwen3.5-2B) |
| Profiler (participant, skip-accuracy) | job `17260737` — summary `phase2_unadapted_profile_summary_v0.json` |
| Qwen3.5-2B Q4 | **Loaded OK** (TPS≈1.71, RSS≈1.9 GB) — keep as watchlist, not drop |
| Translate-test | `phase2_translate_test_v0.md` — Qwen3-1.7B direct-am **0.02** vs EN translate-test **0.04** (n=50) |

**Unadapted TPS / peak RSS (HPC compute node, not Standard Laptop):**

| key | TPS | peak RSS MB |
|-----|-----|-------------|
| qwen3_1_7b_q4_k_m | 2.28 | 1897 |
| qwen3_1_7b_q6_k | 2.01 | 1553 |
| qwen35_2b_q4_k_m | 1.71 | 1935 |
| qwen3_4b_q4_k_m | 1.55 | 4294 |
| qwen25_3b_q4_k_m | 1.41 | 3466 |
| gemma3_4b_q6_k | 1.35 | 3302 |
| gemma3_4b_q4_k_m | 1.28 | 4090 |
| qwen3_4b_q6_k | 1.26 | 3399 |

### Gate 2 (retrospective)
- **Efficiency finalist:** Qwen3-1.7B (Q4_K_M) — best TPS + lowest RSS among screen.
- **Accuracy finalist:** Gemma 3 4B — best Amharic fertility; later adapted Amharic metrics also lead (see Phase 4).

### Next
1. Finish Phase 5 adapted GGUF Pareto / Gate 5
2. Nathan review `amharic_review_sample_v1.jsonl`

---

## 2026-08-16 — Phase 3 NLLB mix_v1 + Gate 3 notes

### Outcome
Job `adtc_nllb_mix_v1-17260753` **COMPLETED**. Real NLLB-200 machine translation → `data/train/am_stem_sft_nllb_v1.jsonl` → `data/train/sft_mix_v1.jsonl` (no retrain of the four merges in this pass).

### What
| Artifact | Notes |
|----------|--------|
| `sft_mix_v1.jsonl` | Direction counts: en_en=900, am_am=1081, en_am=9, am_en=10 (`sft_mix_v1_direction_counts.json`) |
| `amharic_review_sample_v1.jsonl` | 32 stratified Ethiopic rows for Nathan |
| AfriqueLLM GSM8K | Still missing Hub snapshot — known gap |

**Gate 3:** mix_v1 + review sample logged; frozen eval untouched.

### Next
1. Nathan checklist on review sample
2. Optional later: retrain 1.7B on mix_v1 only if Amharic stays weak after Gate 5 pick

---

## 2026-08-16 — Phase 4 adapted HF eval + CPT decision

### Outcome
Job `adtc_eval_adapted-17260740` **failed=0**. Table: `docs/artifacts/phase4_adapted_eval_v0.md` (limit 50 / suite).

| model | am MGSM | en MGSM | am MMLU | en holdout | tutoring |
|-------|---------|---------|---------|------------|----------|
| gemma3_4b_merged_v0 | 0.18 | 0.32 | 0.26 | 0.32 | 0.70 |
| qwen25_3b_instruct_merged_v0 | 0.00 | 0.34 | 0.26 | 0.34 | 0.85 |
| qwen3_1_7b_merged_v0 | 0.00 | 0.36 | 0.00 | 0.36 | 1.00 |
| qwen3_4b_merged_v0 | 0.06 | 0.30 | 0.00 | 0.30 | 1.00 |

### Gate 4 / CPT
- English STEM holdout ~0.30–0.36; tutoring soft-pass high.
- Amharic MGSM still weak (best Gemma 0.18). Translate-test gap on base was small but absolute Amharic is poor.
- **CPT decision: skip for Gate 1 deadline** (default). Revisit after packaging if Amharic remains the blocker. Tokenizer extension not started despite Qwen fertility “severe” — defer unless post-Gate budget allows.

### Next
1. Gate 5 single GGUF from adapted quants

---

## 2026-08-16 — Phase 5 convert/quantize/profile + Gate 5

### Outcome
- Convert/quantize job `17260741` **failed=0** — 20 GGUFs under `artifacts/gguf/adapted/` (f16 + Q8/Q6/Q5/Q4 × 4 models).
- Profiler job `17260742` **COMPLETED** — 16 quantized candidates profiled (participant, `--skip-accuracy`). Pareto: `docs/artifacts/phase5_pareto_v0.md`.

### Gate 5 winner
**`qwen3_1_7b_merged_v0-Q4_K_M`**  
Path: `adtc/artifacts/gguf/adapted/qwen3_1_7b_merged_v0-Q4_K_M.gguf`  
HPC profiler: TPS≈3.1, peak RSS≈1897 MB, steady≈1802 MB, not throttled.

**Rationale:** Best TPS / RSS among adapted quants on Jubail compute (same heuristic as Phase 2 efficiency finalist). Accuracy arrays empty in these runs (`--skip-accuracy`); Amharic HF eval still favors Gemma 3 4B — if leaderboard accuracy dominates after a full accuracy profiler pass, reconsider Gemma Q4/Q5.

### Next
1. Phase 7: point submission `download_model.sh` at the winner GGUF / public host
2. Optional accuracy-mode profiler pass on winner + Gemma Q4
3. Nathan review of `amharic_review_sample_v1.jsonl`

---

## 2026-08-18 — Datasets report (download + train/test splits)

### Outcome
Wrote [`DATASETS_REPORT.md`](./DATASETS_REPORT.md): Hugging Face sources, Hub splits actually used, download OK/FAIL, freeze counts, mix composition, and an in-file acronyms table. Linked from [`DATASETS.md`](./DATASETS.md).

### What
- Train = Hub **`train`** (capped snapshots + GSM8K train[:2000]); test = Hub **`test`** (AfriMGSM/MMLU/XNLI + GSM8K test[:100]) plus authored tutoring/fertility.
- SFT adapters trained on `sft_mix_v0.jsonl` (2 000: GSM8K 900 / Walia 642 / FineTome 458). `sft_mix_v1` (NLLB) is on disk only.
- Download gaps unchanged: gated Amharic news, missing AfriqueLLM GSM8K; SciQ unused; Dolly/TACO/R1 not in the mix.

### Next
1. If retraining, decide whether to switch configs from `sft_mix_v0` → `sft_mix_v1`.
2. Optional: retry gated news with `HF_TOKEN`; find a replacement for AfriqueLLM GSM8K.

---

## 2026-08-18 — Results report (eval + profiler + GGUF leaderboard)

### Outcome
Wrote [`RESULTS_REPORT.md`](./RESULTS_REPORT.md) from existing artifacts (no new jobs). Linked from root [`README.md`](../../README.md) and [`DATASETS.md`](./DATASETS.md).

### What
- Composite on Jubail compute (`0.5 S_acc + 0.3 S_tps + 0.2 S_mem`): rank 1 **Qwen3-1.7B Q4_K_M** (29.44), rank 2 Gemma 3 4B Q4 (25.96), rank 3 Qwen3-4B Q4 (21.44). Source job `adtc_perf_eval-17261185` (was missing from this log).
- Caveats in the report: frozen accuracy **n=50**, tutoring not in S_acc, HPC not Standard Laptop, Phase 5 Pareto was `--skip-accuracy`.
- Amharic still favors Gemma; 1.7B translate-test gap after SFT is **+0.22** (direct AM 0.00 vs EN 0.22).

### Next
1. Optional: full `LIMIT` frozen eval on the three Q4 GGUFs if the n=50 table is not enough for Gate 1 narrative.
2. Nathan review of `amharic_review_sample_v1.jsonl` before `african_alpha_claim`.
3. Phase 7 packaging (template dir is empty; competition `REPORT.md` is a separate file).

---

## 2026-08-18 — Eval defaults: full frozen suites

### Outcome
Frozen eval no longer caps at 50 items. `--limit` is optional (smoke only). Tutoring uses the full 101 rows, not `min(20, limit)`.

### What
- [`eval/run_hf_eval.py`](../eval/run_hf_eval.py), [`eval/run_gguf_eval.py`](../eval/run_gguf_eval.py), [`eval/run_translate_test.py`](../eval/run_translate_test.py): default `--limit=None` (all rows).
- Slurm: `07_eval_adapted`, `06c_translate_test`, `10_perf_eval` only pass `--limit` if `LIMIT=` is exported. Walltimes 24h / 12h / 48h.
- Profiler ARC-Easy in `10_perf_eval` still uses `ACC_LIMIT=50` (lm-eval, not frozen).
- On-disk RESULTS_REPORT numbers remain the old n=50 subsample until a full re-run.

### Next
1. Re-run `07_eval_adapted.sbatch` and `10_perf_eval.sbatch` for full-set tables.

---

## 2026-08-18 — Interactive GGUF prompt picker

### Outcome
Added [`eval/try_prompt.py`](../eval/try_prompt.py): pick one adapted Q4 GGUF, pick one prompt (tutoring / MGSM EN+AM, or type your own), print the reply.

### What
- Models: Qwen3-1.7B, Gemma 3 4B, Qwen3-4B, Qwen2.5-3B Instruct (adapted Q4_K_M).
- Same llama-cpp chat path as `run_gguf_eval.py`. Keeps the GGUF loaded so you can send more prompts.

### Next
1. Run on a compute node (not login): `python eval/try_prompt.py` after activating `adtc/training/.conda-env`.

---

## 2026-08-18 — Slurm wrapper for try_prompt

### Outcome
Added [`hpc/11_try_prompt.sbatch`](../hpc/11_try_prompt.sbatch) so the GGUF picker runs on `compute` instead of login. Prompt + reply go to `adtc/logs/try_prompt/`.

### What
- Default: model 1 (Qwen3-1.7B Q4), prompt 6 (Amharic `2x+5=13`). Override with `MODEL=` / `PROMPT=` / `PROMPTS=`.
- Logs: `adtc/hpc/logs/adtc_try_prompt-<jobid>.{out,err}` (Slurm) and `adtc/logs/try_prompt/<jobid>_m<model>.log` plus `latest.log`.
- `eval/try_prompt.py` gained `--out` and `--prompts`.

### Next
1. From login: `cd adtc/hpc && sbatch 11_try_prompt.sbatch` then `squeue -u $USER`.

---

## 2026-08-18 — try_prompt looks idle because of load + stdout buffering

### Outcome
Job `17294676` was generating; replies are ~3s once the GGUF is in memory. Wall time is dominated by llama.cpp load, plus this run used **three** prompts (`PROMPTS=1,6,10`). Slurm `.out` stayed empty of Python prints (block-buffered).

### What
- Unbuffered stdout (`PYTHONUNBUFFERED=1`, line buffering) and load timing written into the result log.

### Next
1. Tail `adtc/logs/try_prompt/17294676_m1.log` / wait for job end; next submit will show load progress in `.out`.

---

## 2026-08-18 — try_prompt Amharic system policy

### Outcome
`eval/try_prompt.py` now sends a system prompt: if the user writes in Amharic or explicitly asks for Amharic, explanations go in Amharic; equations stay in standard math notation.

### What
- Chat path: `role=system` + user. If the template rejects system (Gemma), the same text is folded into the user turn.
- Logged as `system=amharic-policy` in the result-file header.

### Next
1. Re-run `MODEL=2 PROMPTS=1,6,10 sbatch 11_try_prompt.sbatch` and check prompt 6 is Amharic prose with `2x + 5 = 13` left intact.

---

## 2026-08-20 — Amharic STEM SFT v2 pipeline

### Outcome
Implemented the Amharic STEM SFT **v2** path (data → train → merge → GGUF → eval) without overwriting v0 adapters/GGUFs. Submitted the Slurm dependency chain from `adtc/hpc/submit_v2_chain.sh`.

### What
- Download: `simonbutt/amharic_gsm8k` fallback + retry `afriquellm_gsm8k` (`02c_download_am_gsm8k.sbatch`).
- Normalize: Ethiopic-only filter on Walia/Dolly/FineTome/TACO; AfriqueLLM + simonbutt → `am_am` solve JSONL.
- Tutoring: [`data/build_tutoring_sft_v2.py`](../data/build_tutoring_sft_v2.py) → `en_stem_sft_v2.jsonl` (2000, problem-specific hints/first_error) + `am_tutoring_sft_v2.jsonl` (authored AM, not eval text).
- NLLB harden: placeholder protect `####` / `<<>>` in [`build_translate_am.py`](../data/build_translate_am.py); [`filter_nllb_stem_v2.py`](../data/filter_nllb_stem_v2.py) kept 4000/6000 from v1 MT.
- Mix: [`mix_sft_v2.py`](../data/mix_sft_v2.py) → `sft_mix_v2.jsonl` (5000); counts in `docs/artifacts/sft_mix_v2_counts.json`. FineTome → **0** Ethiopic rows (Latin-only).
- Train configs: `qlora_gemma3_4b_v2.yaml`, `qlora_qwen3_1_7b_v2.yaml`.
- Slurm: `02d_prepare_mix_v2`, `04c_train_sft_v2`, `05b_merge_lora_v2`, `08b_convert_gguf_v2`, `12_eval_v2`.
- `try_prompt.py` MODELS: indices **5** Gemma v2, **6** Qwen 1.7B v2.

### Next
1. Watch chain: `squeue -u $USER`; after `12_eval_v2`, compare AfriMGSM AM vs Gemma v0 0.12 and `logs/try_prompt/latest_v2_gate.log`.
2. If AfriqueLLM download OK, `02d` rebuilds mix with real AM GSM8K before train.
3. CPT still deferred until after this gate.

---

## 2026-08-20 — v2 GPU HF eval (option 1)

### Outcome
Added [`hpc/12b_eval_v2_hf.sbatch`](../hpc/12b_eval_v2_hf.sbatch): nvidia GPU eval of `gemma3_4b_merged_v2` + `qwen3_1_7b_merged_v2` via `run_hf_eval.py` + `run_translate_test.py --model`. Writes `docs/artifacts/phase4_adapted_eval_v2.{md,json}`.

### What
- Fast development gate only — not a substitute for CPU llama.cpp GGUF (judges).
- CPU job `17345551` already reported Gemma Q4 AfriMGSM AM **0.208** (vs v0 0.12); left running for Qwen GGUF.

### Next
1. Tail `logs/adtc_eval_v2_hf-*.out` for HF table vs Phase 4 v0.

---

## 2026-08-21 — Gemma-only Amharic v3 (on-disk CPT→SFT)

### Outcome
Implemented and submitted the Gemma 3 4B **v3** chain using **only on-disk** corpora (no AfriqueLLM download, no new NLLB, no Qwen, no tokenizer extension). Mixes built locally; Slurm dependency chain queued.

### What
- Remix: [`data/mix_sft_v3.py`](../data/mix_sft_v3.py) → `data/train/sft_mix_v3.jsonl` (**8576** rows); counts [`docs/artifacts/sft_mix_v3_counts.json`](./artifacts/sft_mix_v3_counts.json) — AM solve dominated by simonbutt + NLLB v2(+filtered); tutoring still 49 authored.
- CPT mix: [`data/mix_cpt_v3.py`](../data/mix_cpt_v3.py) → `data/train/cpt_mix_v3.jsonl` (**20000**); [`cpt_mix_v3_counts.json`](./artifacts/cpt_mix_v3_counts.json).
- Review: [`amharic_review_sample_v3.jsonl`](./artifacts/amharic_review_sample_v3.jsonl) (32) + [`amharic_review_v3_checklist.md`](./artifacts/amharic_review_v3_checklist.md).
- Configs: `training/configs/cpt_gemma3_4b_v3.yaml`, `qlora_gemma3_4b_v3.yaml` (2 epochs, base=`runs/gemma3_4b_cpt_merged_v3`).
- Slurm: `02e_prepare_mix_v3`, `03b_train_cpt_v3`, `03c_merge_cpt_v3`, `04d_train_sft_v3`, `05c_merge_lora_v3`, `08c_convert_gguf_v3`, `12c_eval_v3`, `12d_eval_v3_hf`; submitter [`hpc/submit_v3_chain.sh`](../hpc/submit_v3_chain.sh).
- `try_prompt.py` MODEL **7** = Gemma v3 Q4.
- Jobs: `17356059` → `17356060` CPT → `17356061` merge CPT → `17356062` SFT → `17356063` merge → `{17356064 GGUF → 17356065 eval, 17356066 HF}`.

### Next
1. `squeue -u $USER`; after CPT/SFT, compare HF AfriMGSM am vs Gemma v2 **0.204** and translate direct **0.12**.
2. Nathan: review sample v3 + `MODEL=7` live prompts before `african_alpha_claim`.
3. Phase 7 packaging after picking deploy GGUF.

---

## 2026-08-22 — Gemma-only Amharic v4 (AfriqueLLM-first, no CPT)

### Outcome
Started the **v4** improvement path: AfriqueLLM GSM8K as primary AM solve, expanded AM tutoring (authored + derived), **no simonbutt** in the mix, **no CPT** — SFT from stock `google/gemma-3-4b-it`. Slurm chain submitted.

### What
- Download harden: `data/download_train_sources.py` → Hub token + `snapshot_download` fallback for flaky AfriqueLLM.
- Tutoring: `data/build_tutoring_sft_v4.py` (~140 authored: solve/hint/first_error/code_switch), `data/derive_am_tutoring_v4.py` (wrap AfriqueLLM solves).
- Mix: `data/mix_sft_v4.py` (40% AM stem / 25% tutoring / 20% EN / 10% instruct / 5% replay); prep `hpc/02f_prepare_mix_v4.sbatch` aborts if AfriqueLLM AM rows &lt; 1500.
- Train: `qlora_gemma3_4b_v4.yaml` → merge/GGUF/eval (`04e`/`05d`/`08d`/`12e`/`12f`); `hpc/submit_v4_chain.sh`.
- `try_prompt.py` MODEL **8** = Gemma v4 Q4.

- Jobs: `17365092` prep → `17365093` SFT → `17365094` merge → `{17365095 GGUF → 17365096 eval, 17365097 HF}`.

### Next
1. Confirm prep job lands AfriqueLLM ≥1500 and writes `sft_mix_v4.jsonl`.
2. After HF eval, compare am MGSM vs v3 **0.232** and translate vs v3 direct **0.10**.
3. Re-run try_prompt 8/10 (first_error + code_switch) on MODEL 8.

---

## 2026-08-22 — v4 prep FAIL (AfriqueLLM 404); fallback patched

### Outcome
Prep job **`17365092`** failed (~38s): Hub returns **404** for `peterlu02/afriquellm-coldstart-gsm8k-11lang` (even with `HF_TOKEN=set`). Downstream jobs show **`DependencyNeverSatisfied`** because prep did not complete.

### What
- Log: [`hpc/logs/adtc_prep_mix_v4-17365092.out`](../hpc/logs/adtc_prep_mix_v4-17365092.out) — download error + missing `afriquellm_gsm8k_am_sft_v0.jsonl`.
- On-disk fallback only: NLLB filtered **657** good `am_am` solve + simonbutt **7473** (cap 2500 in pool).
- Patched: `data/build_am_stem_pool_v4.py`, `02f_prepare_mix_v4.sbatch` (no hard abort on AfriqueLLM; tutoring derive from pool).

### Next
1. `scancel 17365093 17365094 17365095 17365096 17365097` then `cd adtc/hpc && bash submit_v4_chain.sh`.
2. If AfriqueLLM becomes reachable, pool auto-selects it when ≥1500 rows.
3. After train, compare tutoring-heavy v4 vs v3 on try_prompt 8/10.

---

## 2026-08-22 — Added GEMMA_V4_CONTEXT.md for external-LLM briefing

### Outcome
Wrote a self-contained project briefing for GPT/Claude covering architecture, v4 data mix (AfriqueLLM fallback), QLoRA hypers, full-suite eval vs v3, caveats, and recommended next steps.

### What
- New: [`docs/GEMMA_V4_CONTEXT.md`](./GEMMA_V4_CONTEXT.md)
- Linked from repo README docs index

### Next
1. Use the briefing for deploy pick (Gemma v3 vs v4 vs Qwen 1.7B) and Phase 7 packaging.
2. Nathan Amharic review + profiler re-aggregate on chosen GGUF.

---

## 2026-08-23 — ADTC v5 Milestone A (AfriqueQwen3.5 infra; no training yet)

### Outcome
Scaffolded non-destructive v5 pipeline for `McGill-NLP/AfriqueQwen3.5-4B-ExtendedCM` + `AfriqueQwen3.5-4B` (+ `AfriqueQwen-4B` fallback). Built CPT/SFT mixes, HF config preflight, Slurm chain scripts. **Did not launch CPT/SFT.** GGUF/HF-4bit preflight jobs queued.

### What
- Manifest: `training/configs/v5_models.yaml`
- Loader / LoRA inspect / HF preflight: `training/model_loader.py`, `inspect_lora_targets.py`, `preflight_v5_model.py`
- Mixes: `cpt_mix_v5.jsonl` (21500), `sft_mix_v5.jsonl` (8086, simonbutt_frac=0, am_en=800 / en_am=2142), `xling_sft_v5.jsonl` (1600)
- Configs: `cpt_qwen35_*_v5.yaml`, `qlora_qwen35_*_v5.yaml`, fallback + M0→SFT ablation
- HPC: `02g`/`03d`/`03e`/`04f`/`05e`/`08e`/`08f`/`08g`/`12g`/`12h` + `submit_v5_preflight.sh` / `submit_v5_chain.sh`
- Report: [`docs/artifacts/v5/MILESTONE_A_REPORT.md`](./artifacts/v5/MILESTONE_A_REPORT.md)
- Jobs: GGUF preflight `17375335–17375337`, HF GPU preflight `17375338`

### Architecture note
Qwen3.5 Afrique checkpoints are `Qwen3_5ForConditionalGeneration` with **vision_config** — GGUF text-only single-file gate is the critical next check before training.

### Next
1. Wait for `docs/artifacts/v5/gguf_preflight_*.json` (+ GPU HF 4bit fields in `model_preflight.json`).
2. If Qwen3.5 text GGUF fails / needs mmproj → train `qwen3_afrique` instead.
3. On your OK: `V5_ALIAS=qwen35_extcm bash submit_v5_chain.sh` (Milestone B).

---

## 2026-08-23 — Fix v5 GGUF preflight `ALIAS` KeyError

### Outcome
GGUF gate jobs (`17375335–37`, `17375498–500`) failed in ~1–2s with `KeyError: 'ALIAS'` before convert. Root cause: `08f` ran a Python heredoc that read `os.environ["ALIAS"]` before `export ALIAS`. HF preflight was already OK; all three Hub weights are on disk (~8–10G each).

### What
- Fixed [`hpc/08f_preflight_gguf_v5.sbatch`](../hpc/08f_preflight_gguf_v5.sbatch): `export ALIAS` first, removed dead `/tmp` HF_ID block, require snapshot to contain `*.safetensors` (avoid tokenizer-only stubs).

### Next
1. Resubmit GGUF-only: `cd adtc/hpc && for a in qwen35_extcm qwen35_afrique qwen3_afrique; do sbatch --export=ALL,V5_ALIAS=$a 08f_preflight_gguf_v5.sbatch; done`
2. Review `docs/artifacts/v5/gguf_preflight_*.json`.
3. Then approve Milestone B train chain if gate passes.

---

## 2026-08-24 — try_prompt MODEL=9 HF backend for v5

### Outcome
`llama_cpp` cannot load `qwen35` GGUF (arch unsupported). Added HF transformers path for model 9 (`qwen35_extcm_merged_v5`) and updated `11_try_prompt.sbatch` to require A100 for MODEL=9.

### What
- `eval/try_prompt.py`: model 9 → `hf_path`; CUDA when available
- `hpc/11_try_prompt.sbatch`: docs + GPU guard for MODEL=9

### Next
1. `MODEL=9 PROMPTS=all sbatch -p nvidia --gres=gpu:a100:1 --mem=64G -t 4:00:00 11_try_prompt.sbatch`
2. Compare qualitative replies vs Gemma v4 (model 8)

---

## 2026-08-24 — Split backlog into phased git commits

### Outcome
Committed the long uncommitted backlog (since `installed llama.cpp`) as seven focused commits on `main`, authored as Nathan Behailu. Left secrets / train JSONL / HF caches untracked per `.gitignore`.

### What
| Commit | Topic |
|--------|--------|
| `00a998d` | Full-suite eval + perf leaderboard + Gate 2–5 v0 artifacts |
| `fe7c839` | `try_prompt` + `11_try_prompt.sbatch` |
| `debca6d` | Amharic SFT **v2** pipeline |
| `922c92c` | Gemma **v3** CPT→SFT |
| `405b96b` | Gemma **v4** SFT + briefing |
| `cf883c2` | AfriqueQwen **v5** infra / preflight |
| `65cf22b` | Docs index / PRD / HPC README / DEVLOG catch-up |

### Next
1. Optional: `git push origin main` when ready.
2. Continue try_prompt MODEL=9 qualitative check vs Gemma v4.

---

## 2026-08-24 — ADTC v6 English-only Qwen3-1.7B scaffold

### Outcome
Scaffolded non-destructive **v6** track: English-only SFT on `Qwen/Qwen3-1.7B` (no Afrique / Amharic CPT), mix built (**10473** rows), train→GGUF→eval→profiler chain scripts ready. **Did not launch train.**

### What
- Mix: `data/mix_sft_v6.py` → `data/train/sft_mix_v6.jsonl` (GSM8K 7473 + SciQ 3000; dedup 0)
- Config / HPC: `qlora_qwen3_1_7b_v6.yaml`, `02h`/`04g`/`05f`/`08h`/`12i`/`12j`/`09b`, `submit_v6_chain.sh`
- Thinking off for scoring: HF `enable_thinking=False`; GGUF `/no_think`
- `try_prompt` model **10** = `qwen3_1_7b_merged_v6-Q4_K_M.gguf`
- Report: [`docs/artifacts/v6/MILESTONE_REPORT.md`](./artifacts/v6/MILESTONE_REPORT.md)

### Next
1. `cd adtc/hpc && bash submit_v6_chain.sh`
2. Review `docs/artifacts/v6/*` vs v0 Qwen Q4 Gate 5 baseline.
3. Pick Q4/Q5/Q6 via `phase5_gate5_winner_v6.json`.

---

## 2026-08-25 — Submission cleanup: v6 English-only only

### Outcome
Stripped the repo to the **v6 English-only** pipeline for Gate 1 submission. Removed v0–v5 Amharic/Afrique/multilingual experiment code, Slurm chains, configs, docs artifacts, and ~365G of obsolete checkpoints/GGUFs/HF caches. Repo `adtc/` now ~23G (v6 runs + GGUF quants + Qwen3-1.7B base cache).

### What
- **Kept:** `submit_v6_chain.sh`, v6 sbatch scripts, `mix_sft_v6.py`, `qlora_qwen3_1_7b_v6.yaml`, eval harness, `docs/artifacts/v6/`
- **Deleted:** ~50 legacy Slurm scripts, v0–v5 data builders/train configs, `docs/artifacts/{v5,perf,phase*}`, non-v6 `training/runs/`, non-v6 GGUFs (~193G), `profiler_stage/{adapted,unadapted}`, obsolete HF bases (Gemma, AfriqueQwen, NLLB, etc.)
- **Slimmed:** `try_prompt.py` → single Q5 model; `download_base_models.py` → Qwen3-1.7B only; docs/READMEs rewritten for EN-only v6
- **Staged:** `adtc-2026-submission-template/` with `qwen3_1_7b_merged_v6-Q5_K_M.gguf` + `metadata.json` + `download_model.sh`

### Next
1. Fill submission `REPORT.md` + demo video on laptop profiler numbers.
2. Optional: `git push` when ready.

---

## 2026-08-25 — Rename HPC scripts + PIPELINE.md

### Outcome
Renamed all numbered Slurm scripts in `adtc/hpc/` to descriptive names (no leading digits). Added [`docs/PIPELINE.md`](./PIPELINE.md) as the end-to-end data → model → training → results guide.

### What
- `02h_prepare_mix_v6.sbatch` → `prepare_mix.sbatch`, `03_download_models` → `download_models`, `04g_train_sft_v6` → `train_sft`, `05f_merge_lora_v6` → `merge_lora`, `08h_convert_gguf_v6` → `convert_gguf`, `09b_profile_gguf_v6` → `profile_gguf`, `11_try_prompt` → `try_prompt`, `12i_eval_v6_hf` → `eval_hf`, `12j_eval_v6_gguf` → `eval_gguf`, `submit_v6_chain.sh` → `submit_chain.sh`
- Updated `submit_chain.sh` dependencies + READMEs to match

### Next
1. Use `bash submit_chain.sh` for future runs.

---

## 2026-08-25 — Submission template chat demo

### Outcome
Aligned `adtc-2026-submission-template/` with the official ADTC layout and added a self-contained terminal chat demo (venv + `download_model.sh` + `chat.py`). Root README refreshed for TebebAI v6 and the submission demo path.

### What
- **Removed:** `compile_report.sh`, `REPORT.tex` (not in official template)
- **Added:** `.gitignore`, `LICENSE` (GPL-3.0), `README.md`, `chat.py`, `requirements.txt` (`llama-cpp-python`)
- **chat.py:** checks GGUF via `metadata.json` `_runtime.model_path`, multi-turn history, strips `<think>` / `<<>>`
- **Root README:** TebebAI headline metrics + pointer to submission `chat.py`

### Next
1. Point `download_model.sh` at a public URL before Gate submission (currently local-stage stub).
2. Demo: `cd adtc-2026-submission-template && python3 -m venv .venv && … && python chat.py`

---

## 2026-08-25 — REPORT.md rewritten from adtc pipeline

### Outcome
Rewrote [`adtc-2026-submission-template/REPORT.md`](../../adtc-2026-submission-template/REPORT.md) as a technical ADTC writeup (Problem / Design / Constraints / Benchmarks) grounded in the v6 `adtc/` pipeline, Gate 5 Pareto table, HF frozen eval, and profiler scores.

### What
- Base Qwen3-1.7B, QLoRA config highlights, `sft_mix_v6` 10473 rows, Slurm chain, Q4/Q5/Q6 table, S_tps/S_mem, Jubail caveat
- Points to `chat.py` local demo and `adtc/docs` artifacts

### Next
1. Fill real `team_id` / submitter fields in `metadata.json` before public submit.
2. Host GGUF + fix `download_model.sh` for credential-free fetch.

---

## 2026-08-25 — HF download_model + model_path rename

### Outcome
`download_model.sh` now fetches `tebeb_tutor_1.7b.gguf` from Hugging Face `nz2212/tebeb_tutor_1.7b` (idempotent). `metadata.json` `_runtime.model_path` is `model/tebeb_tutor_1.7b.gguf`. Local `model/` already has the full GGUF; no full re-download needed — HF URL was confirmed reachable earlier (resolve → CDN).

### What
- Updated `chat.py` default, README/REPORT path strings
- Cleaned interrupted `.partial` / `.localbak` after a cancelled full fetch

### Next
1. Submit when Devpost fields + public repo are ready.

