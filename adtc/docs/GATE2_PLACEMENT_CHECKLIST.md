# Gate 2 — submission placement checklist

**Re-audited:** 22 Sep 2026 (after GitHub push `9caa7ca` + local packaging tidy).  
**Submit repo:** https://github.com/nathanbehailuz/adtc-2026-submission-template  
**Sources:** [submission template README](../../adtc-2026-submission-template/README.md), Gate 2 Semifinalist Round Submission Guidelines, Round 1 TebebAI results (`ADTC2026_755`).

Legend: **PASS** in place · **GAP** present but incomplete · **OPS** not a file, still required to submit.

---

## Required tree (official)

```
adtc-2026-submission-template/
├── metadata.json
├── download_model.sh
├── REPORT.md
├── chat.py / requirements.txt     ← optional local demo (already on GitHub)
├── model/                         ← gitignored
├── provenance/                    ← QLoRA proof
└── .gitignore
```

---

## 1. Required files — on disk vs GitHub

| Item | Status |
|------|--------|
| `metadata.json` TebebAI (domain, provenance, 2 test_prompts) | **PASS** on GitHub `b3ca790` (triangle + Betty) |
| `download_model.sh` pinned HF URL | **PASS** on GitHub `9caa7ca` |
| `REPORT.md` TebebAI + Model Provenance | **PASS** (updated with licenses + laptop tutoring traces) |
| `model/` gitignored; no `.gguf` in git | **PASS** |
| Repo public | **PASS** |
| `chat.py` + `requirements.txt` | **PASS** on GitHub `9caa7ca` |
| `provenance/` adapter + scripts | **PASS** on GitHub `9caa7ca` |

**Do not commit:** `model/*.gguf`, `.venv/`, `submission.json`, `audit.json`.

---

## 2. `metadata.json`

| Check | Status | Notes |
|-------|--------|-------|
| Identity / domain / submitter | **PASS** | TebebAI / math_scientific_reasoning |
| Exactly **2** `test_prompts` | **PASS** | tp_001 triangle first-error; tp_002 Betty hint (on-policy laptop smoke) |
| `model.runtime` = `llama.cpp` | **PASS** | |
| `_runtime.model_path` | **PASS** | `model/tebeb_tutor_1.7b-Q4_K_M.gguf` |
| **No** `git_commit_sha` key | **PASS** | |
| Provenance object | **PASS** | Qwen3-1.7B @ `70d244cc…`, qlora, GSM8K/SciQ/authored |

---

## 3. `download_model.sh`

| Check | Status | Notes |
|-------|--------|-------|
| Only `MODEL_FILE` / `MODEL_URL` edited | **PASS** | |
| Static URL, commit pin, no `main` | **PASS** | `58347614c3c7860c126f62fc7bbdb3cd1d15dd65` |
| HF file public | **PASS** | `nz2212/tebebAIv2` Q4 GGUF |
| Path rename to `_runtime.model_path` | **PASS** | |
| Clean-clone `bash download_model.sh` | **GAP** | Local `model/` is still a **symlink** to the hackathon GGUF |

---

## 4. `REPORT.md`

| Section | Status |
|---------|--------|
| Problem / Design / Constraints | **PASS** |
| Model Provenance table + licenses | **PASS** |
| ≥2 tutoring prompt traces (v7 Q4) | **PASS** (triangle + Betty). Stock Qwen3-1.7B was **not** re-inferred on this laptop; accuracy table is v6 vs v7 |
| Benchmarks | **GAP** — official Standard Laptop TPS/RSS still “measured by profiler”; do not invent |

---

## 5. `provenance/`

| Required | Status |
|----------|--------|
| Adapter + config + train YAML + scripts | **PASS** (on GitHub) |
| `training_log.txt` | **PASS** — mean `train_loss=0.6026` (no step-level `trainer_state.json`) |
| Dataset info + licenses + mixed sample | **PASS** |
| SHA256 list | **PASS** — GGUF `0c00f7a5…`; no `TBD_AFTER_UPLOAD`; Hub revision labeled as git SHA |

---

## 6. Official Gate 2 checklist (guidelines §4)

| # | Item | Status |
|---|------|--------|
| 1 | Template-compliant public repo | **PASS** (`b3ca790`) |
| 2 | Provenance in REPORT + metadata | **PASS** |
| 3 | `provenance/` complete | **PASS** for required artifacts |
| 4 | `download_model.sh` static URL | **PASS** |
| 5 | Originality / citations | **PASS** — licenses in `dataset_info.md` + REPORT |
| 6 | Benchmarks honest | **GAP** — no Standard Laptop profiler numbers |
| 7 | Model still useful | **PASS** |
| 8 | Video ≤ 2:00 | **OPS** |
| 9 | Eligibility | **OPS** |
| 10 | Devpost submit | **OPS** |

---

## 7. Still to do (human / ops)

1. Record ≤2 min video; paste [`DEVPOST.md`](DEVPOST.md); book diligence call.
2. Optional: delete local GGUF symlink and `bash download_model.sh` (~1.1 GB).
3. Official TPS/RSS only from ADTC profiler.

Judges clone **only** the GitHub submission URL. They run `download_model.sh` + `adtc-profiler` + llama.cpp — **not** `chat.py`.
