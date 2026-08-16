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
