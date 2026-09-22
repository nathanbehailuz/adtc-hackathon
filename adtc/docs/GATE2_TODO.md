# Gate 2 (Semifinal) — ordered TODO

**Deadline:** 22 Sep 2026 (today). Organizers recommend submit **≥12 hours early** — treat remaining time as critical path.  
**Submission repo (fork):** [`adtc-2026-submission-template/`](../../adtc-2026-submission-template/) → https://github.com/nathanbehailuz/adtc-2026-submission-template  
**Upstream template:** https://github.com/Africa-Deep-Tech-Foundation/adtc-2026-submission-template/  
**Guidelines:** Gate 2 Semifinalist Round Submission Guidelines (email PDF).  
**Placement audit (pass/fail vs official tree):** [`GATE2_PLACEMENT_CHECKLIST.md`](GATE2_PLACEMENT_CHECKLIST.md)

Work **parallel tracks**: (A) wait for Shadeform v7 artifacts, (B) package everything that does not need the new GGUF.

---

## Track A — waiting on Shadeform (do not block Track B)

| # | Task | Done when |
|---|------|-----------|
| A1 | Q4 GGUF frozen eval finishes | `docs/artifacts/v7/qwen3_1_7b_merged_v7-Q4_K_M_eval.json` exists (**done**; included AM — ignore AM cols) |
| A2 | Q5 GGUF frozen eval (**EN-only** after syncing `eval/run_gguf_eval.py`) | `…-Q5_K_M_eval.json` — EN MGSM + holdout + tutoring only (**done on Shadeform**) |
| A3 | `judge_smoke` (format leak + multi-part tutoring) | `docs/artifacts/v7/judge_smoke_*.json` — **no `####` / `<<>>`** |
| A4 | `profile_gguf` Gate 5 pick | `phase5_gate5_winner_v7.json` (Q4 vs Q5 vs Q6) |
| A5 | Copy winner GGUF off instance | local + File Storage backup |
| A6 | Shut down Shadeform when idle | stop GPU billing |

**Laptop pull** (copy-paste entire block into Terminal on your Mac):

```bash
KEY=~/.ssh/agh-a6000.pem
HOST=shadeform@216.81.248.135
REMOTE=~/adtc-hackathon/adtc
LOCAL=~/Desktop/adtc-hackathon/adtc

mkdir -p \
  "$LOCAL/artifacts/gguf/adapted" \
  "$LOCAL/training/runs/qwen3_1_7b_qlora_v7" \
  "$LOCAL/data/train" \
  "$LOCAL/docs/artifacts/v7" \
  "$LOCAL/logs"

scp -i "$KEY" \
  "$HOST:$REMOTE/artifacts/gguf/adapted/qwen3_1_7b_merged_v7-Q4_K_M.gguf" \
  "$LOCAL/artifacts/gguf/adapted/"

scp -i "$KEY" \
  "$HOST:$REMOTE/artifacts/gguf/adapted/qwen3_1_7b_merged_v7-Q5_K_M.gguf" \
  "$LOCAL/artifacts/gguf/adapted/"

scp -i "$KEY" -r \
  "$HOST:$REMOTE/training/runs/qwen3_1_7b_qlora_v7/adapter" \
  "$LOCAL/training/runs/qwen3_1_7b_qlora_v7/"

scp -i "$KEY" \
  "$HOST:$REMOTE/data/train/sft_mix_v7.jsonl" \
  "$LOCAL/data/train/"

scp -i "$KEY" \
  "$HOST:$REMOTE/docs/artifacts/v7/qwen3_1_7b_merged_v7_hf_eval.json" \
  "$HOST:$REMOTE/docs/artifacts/v7/qwen3_1_7b_merged_v7-Q4_K_M_eval.json" \
  "$HOST:$REMOTE/docs/artifacts/v7/qwen3_1_7b_merged_v7-Q5_K_M_eval.json" \
  "$HOST:$REMOTE/docs/artifacts/v7/gguf_manifest.json" \
  "$LOCAL/docs/artifacts/v7/"

scp -i "$KEY" -r \
  "$HOST:$REMOTE/logs/train_sft" \
  "$LOCAL/logs/" || true

echo "Done. Check:"
ls -lh "$LOCAL/artifacts/gguf/adapted/"*.gguf
ls -lh "$LOCAL/training/runs/qwen3_1_7b_qlora_v7/adapter/"
ls -lh "$LOCAL/data/train/sft_mix_v7.jsonl"
ls -lh "$LOCAL/docs/artifacts/v7/"
```

**On instance (tmux) — remaining A3+A4:**

```bash
cd ~/adtc-hackathon/adtc/agh && source env.sh
unset LD_LIBRARY_PATH
V7_QUANT=Q4_K_M bash judge_smoke.sh
# optional: bash profile_gguf.sh   # re-sets LD_LIBRARY_PATH for llama-bench
```

---

## Track B — submission packaging (start now)

### 1. Sync official template (first)

1. Pull latest **upstream** Gate 2 template into the fork (provenance + placeholder `download_model.sh`).
2. Confirm fork has: `metadata.json` (with `provenance`), `download_model.sh` placeholders, `REPORT.md` Model Provenance section, `model/` gitignored.
3. Do **not** invent a new download script — only replace the two placeholders later.

### 2. Identity / metadata (no GGUF needed)

Edit `adtc-2026-submission-template/metadata.json`:

- [x] Real `team_id`, submitter name / email / GitHub
- [x] `domain`: `math_scientific_reasoning` (not coding template default)
- [x] `language_scope`, `african_alpha_claim`, cross-disciplinary fields
- [x] Exactly **2** `test_prompts` (tutoring / STEM; match Round-1 strength)
- [x] `model.name`, `quantization`, `parameters_estimate` for the **winner** GGUF — **Q4_K_M**
- [x] `_runtime.model_path` = `model/tebeb_tutor_1.7b-Q4_K_M.gguf`
- [x] `provenance.base_model_source` = `huggingface:Qwen/Qwen3-1.7B`
- [x] `provenance.base_model_commit_sha` = **exact Hub revision SHA** used for train (not “latest”)
- [x] `provenance.fine_tuning_method` = `qlora`
- [x] `provenance.training_datasets` = GSM8K / SciQ / authored tutoring v7 (names + sources)

### 3. `provenance/` proof-of-training (mandatory for QLoRA)

Create `adtc-2026-submission-template/provenance/`:

| File / content | Source |
|----------------|--------|
| Adapter weights `adapter_model.safetensors` + `adapter_config.json` | Shadeform `training/runs/qwen3_1_7b_qlora_v7/adapter/` |
| Train config | copy `adtc/training/configs/qlora_qwen3_1_7b_v7.yaml` |
| Train / merge scripts | `train_sft_qlora.py`, `merge_lora.py` (or thin copies + path notes) |
| Loss / run logs | `adtc/logs/train_sft/` (or trainer `trainer_state.json` / log) |
| Dataset description + sample + checksums | mix report + sample of `sft_mix_v7.jsonl` + SHA256 of mix file |
| Merge → GGUF notes / script refs | `merge_lora.py` + `agh/convert_gguf.sh` |
| SHA256 list | base HF revision, adapter, **final GGUF** (fill GGUF hash after upload) |

### 4. `REPORT.md` rewrite (TebebAI, not template stubs)

Ordered sections:

1. Problem (EN STEM tutor, offline Africa / laptop)
2. Design decisions (Qwen3-1.7B, Q4/Q5 pick, alternatives)
3. **Model Provenance** (must match `metadata.json`; include **≥2 before/after** base vs fine-tuned prompts)
4. Constraints (8 GB, CPU llama.cpp, offline)
5. Benchmarks (use **reproducible** profiler numbers — update after A4; do not invent TPS)

Call out Round-1 fixes: stripped `####`/`<<>>`, richer scaffolding, authored tutoring, r=32/α=64, 2 epochs.

### 5. Host GGUF + pin `download_model.sh` (needs A5)

1. Upload winner GGUF to public HF (e.g. `nz2212/tebeb_tutor_1.7b` or v7 repo).
2. Copy the file’s **commit SHA** (not `main`).
3. Edit **only**:

```bash
MODEL_FILE="$MODEL_DIR/[YOUR_MODEL_FILE_NAME].gguf"
MODEL_URL="[YOUR_MODEL_URL]"
```

`MODEL_URL` must be a **static** resolve URL with commit SHA (no `${VAR}`, no branch `main`).  
4. Align `_runtime.model_path` / `MODEL_FILE` names exactly.

### 6. Local packaging smoke (before Devpost)

```bash
cd adtc-2026-submission-template
rm -f model/*.gguf    # force re-download
bash download_model.sh
# confirm path matches metadata _runtime.model_path
# optional: python chat.py / adtc-profiler participant smoke
```

### 7. Demo video (max **2:00**, enforced)

| Segment | Time | Content |
|---------|------|---------|
| Elevator pitch | ~30s | Problem + offline tutor |
| Demo | ~60s | Live hint / diagnose / no answer dump |
| Innovation | ~30s | QLoRA + GGUF + provenance / laptop LLM |

Upload to Devpost (and YouTube/unlisted if useful).

### 8. Process / ops (parallel anytime)

- [ ] Book **mandatory** 15-min diligence call: https://calendar.app.google/3xcUM2LYJp5Wu7Mr8 — **all** team members, cameras on
- [ ] Mentor intro email — respond when it arrives
- [ ] AGH `$50` credit: confirm account email sent to organizers (if not already)
- [ ] Devpost project reopened — update description ([`DEVPOST.md`](DEVPOST.md)), links, video, repo URL
- [x] Repo **public**; no secrets; `model/` gitignored

### 9. Final Gate 2 checklist (official)

- [x] Template-compliant repo (push remaining metadata/REPORT/provenance tidy)
- [x] Model Provenance in REPORT + `metadata.json`
- [x] `provenance/` complete (adapter + scripts + loss mean + licenses + GGUF SHA)
- [x] `download_model.sh` static URL, placeholders only edited
- [x] Originality / citations OK (licenses listed)
- [x] Benchmarks honest (no invented Standard Laptop TPS)
- [x] Model still useful (not gamed for TPS only)
- [ ] Updated ≤2 min video
- [ ] Eligibility unchanged
- [ ] **Submit on Devpost** (aim ASAP today)

---

## Suggested order today (critical path)

```
1. metadata.json identity + domain + 2 prompts + provenance fields (draft SHA)
2. Start provenance/ folder (adapter + config + scripts + logs) — scp from Shadeform
3. Draft REPORT.md (leave Benchmarks table TBD)
4. When A4/A5 land → upload GGUF → pin download_model.sh → SHA256 → fill REPORT benchmarks
5. Packaging smoke (download_model.sh clean clone)
6. Record / upload video
7. Push fork → Devpost submit
8. Book diligence call if not booked
```

## Out of scope for this file

- Re-training another mix (only if judge_smoke fails markup / scaffolding).
- Changing `download_model.sh` beyond the two placeholders.

## Status snapshot (local, 22 Sep 2026)

| Item | Status |
|------|--------|
| v7 train / merge / GGUF convert | Done on Shadeform |
| HF frozen eval v7 | Done (EN MGSM 0.444, EN STEM 0.47) |
| Q4 GGUF eval | Done; Q4 is deploy pick |
| Q5 / judge_smoke | Done (Q5 eval + judge_smoke JSON in `docs/artifacts/v7/`) |
| Submission template (local) | shorter `test_prompts` (triangle + Betty); GitHub `d621257` |
| `provenance/` | On GitHub `9caa7ca`; this revision adds licenses, `train_loss=0.6026`, mixed sample, GGUF SHA (no TBD) |
| GitHub fork | Pushed `b3ca790` — on-policy test_prompts + provenance tidy |
| Remaining ops | Video, Devpost paste, diligence call, optional HF re-download smoke |
| Placement checklist | [`GATE2_PLACEMENT_CHECKLIST.md`](GATE2_PLACEMENT_CHECKLIST.md) |
