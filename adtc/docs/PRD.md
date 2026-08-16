# PRD + Step-by-Step Plan — ADTC Offline Multilingual STEM Tutor

**Status:** Draft for Gate 1 (deadline **25 Aug 2026**)  
**Source:** `[idea.md](./idea.md)`  
**Domain:** `math_scientific_reasoning`  
**Artifact:** one GGUF + llama.cpp (no multi-model runtime)

---

## 1. Product summary

### Problem

Students and teachers in low-connectivity settings need a **STEM tutor that runs fully offline** on an 8 GB budget laptop, in **English + Amharic**, with tutoring behaviors (explain, hint, diagnose error)—not just MCQ answers.

### Goal

Ship a **single dense 1.7–4B specialist GGUF** that maximizes the ADTC score:


| Weight | Metric     | Notes                                                    |
| ------ | ---------- | -------------------------------------------------------- |
| 50%    | Accuracy   | Domain prompts (EN + target language)                    |
| 30%    | Throughput | Track raw TPS; profiler refs ~15 tok/s                   |
| 20%    | Memory     | S_{\mathrm{eff}} = 100 \times (7 - \mathrm{PeakRAM}) / 7 |


Hard constraints: peak RSS ≤ 7 GB, thermal penalty above 85°C / throttle, OOM = fail.

### Non-goals (Gate 1)

- Multi-LLM routers / runtime translation sandwich
- Starting from 7B+ and shrinking
- Healthcare / AfriMed-QA as primary track
- QAT, tokenizer surgery, or heavy CPT unless diagnostics force it
- Qwen3.5 as main path until profiler proves llama.cpp compatibility



### Central hypothesis

A **1.7–4B** model, adapted with **native X + translated STEM + English reasoning replay**, then quantized, can beat a larger generic model on the ADTC objective on CPU.

### Primary bets


| Role                  | Model               | Likely quants    |
| --------------------- | ------------------- | ---------------- |
| Efficiency challenger | Qwen3-1.7B          | Q4_K_M / Q6_K    |
| Accuracy challenger   | Qwen3-4B            | Q4_K_M / Q5_K_M  |
| Architecture control  | Gemma 3 4B          | Q4-class         |
| Middle-size control   | Qwen2.5-3B-Instruct | Q4_K_M (one run) |


---



## 2. Success criteria



### Must ship (Gate 1)

- [ ] Public GitHub repo from official template
- [ ] `metadata.json` complete (`domain: math_scientific_reasoning`, `budget_laptop_claim: true`)
- [ ] Exactly **2** domain `test_prompts` (assume **≥3** hidden prompts—do not overfit demos)
- [ ] Idempotent, credential-free `download_model.sh` → path matches `_runtime.model_path`
- [ ] One valid `.gguf`, offline inference only
- [ ] `REPORT.md` with measured design narrative
- [ ] 2-minute demo video (laptop specs, offline, EN + X STEM tutoring, profiler numbers)



### Performance targets (local / profiler)

- Peak RSS comfortably under 7 GB with headroom for S_{\mathrm{eff}}
- Sustained generation without thermal fail
- Measurable gains on: EN-STEM, X-STEM, X-general, custom tutoring set
- Final pick = **Pareto winner** on accuracy × TPS × RSS, not “largest model that fits”



### African-language claim

Set `african_alpha_claim: true` only if a fluent validator reviewed outputs/data and language quality is real—not because a benchmark exists.

---



## 3. Architecture (locked decisions)

```
HF base/instruct
  → LoRA/QLoRA SFT (cloud GPU)
  → merge adapters
  → high-precision GGUF
  → PTQ sweep Q8 → Q6 → Q5 → Q4
  → ADTC profiler + held-out eval
  → one final GGUF
```

**Many models in training (teachers, MT, data gen); one model at inference.**

**Conditional only:**

- Mixed CPT → SFT if direct-X still weak after bilingual SFT
- Tokenizer extension if fertility diagnostics are severe
- Distillation if 4B ≫ 1.7B accuracy **and** 1.7B ≫ 4B hardware score
- QAT only if Q4 accuracy collapses while Q4 still wins on hardware

**Default inference:** direct multilingual model (not EN↔X MT at runtime).

---



## 4. Language & evaluation freeze



### Language selection rule (do this before training data)

1. Teammate/collaborator can **fluently validate** outputs → essential
2. Covered by IrokoBench / AfriQA / AfroBench → very high
3. Native text + MT resources exist → high
4. Tokenizer not catastrophically fragmented → high
5. “Model already speaks X a bit” → helpful, not decisive

**Locked for this project:** target language **Amharic (`am`)**, with English (`en`) as primary evaluation language. Validate scientific terminology and register in Amharic before claiming alpha quality.

### Freeze before any synthetic data


| Suite                               | Purpose                                |
| ----------------------------------- | -------------------------------------- |
| AfriMGSM (X)                        | Multilingual math                      |
| AfriMMLU (X)                        | Knowledge / STEM                       |
| AfriXNLI (X)                        | Language understanding (not just math) |
| AfriQA (optional)                   | Cross-lingual QA                       |
| English STEM held-out               | Forgetting check                       |
| Custom bilingual tutoring (100–300) | Product behavior                       |
| Tokenizer fertility set             | Runtime / language diagnostic          |


**Never train on these test sets.** Dedup/hash against held-out prompts where practical.

### Per-run metrics to log

- Accuracy: A_{\text{EN-STEM}}, A_{\text{X-STEM}}, A_{\text{X-general}}, \Delta_{\text{forget}}
- Systems: gen TPS, prompt TPS, TTFT, peak RSS, steady RSS, T_{\text{peak}}, throttle flag
- Contexts: **2K and 4K** + one sustained run (avoid advertising huge context you don’t need)



### Human review sample

Fluent speaker blind-reviews stratified outputs for: grammar, terminology, facts, language consistency, pedagogical clarity.

---



## 5. Step-by-step todo list

Work in order. **Do not skip decision gates.** Dates assume ~9 days to Gate 1 (16→25 Aug)—compress later phases if early gates slip.

### Current status / working order (as of 16 Aug 2026)

| Status | Phase | Notes |
|--------|-------|--------|
| **Done** | Phase 0 Gate 0 | Packaging smoke passed (SmolLM2 + pinned profiler). See `DEVLOG.md` / `TOOLING.md`. |
| **Done** | Phase 1 Gate 1a | Language lock + frozen eval under `adtc/data/eval/` — see `LANGUAGE.md`, `FREEZE.md`. |
| **Now** | Phase 3 data build | SFT builders exist; build/review train mixes **without** touching frozen eval. |
| **Tomorrow** | Phase 2 | Download/profile Must GGUFs (or pull HF bases on **cloud** for training). Hardware + accuracy screen. |
| **Then** | Phase 4 run | QLoRA on cloud GPU using HF checkpoints; GGUF only for profiler/submission. |

**Code locations:** datasets → [`adtc/docs/DATASETS.md`](./DATASETS.md), `adtc/data/`; eval scripts → `adtc/eval/`; fine-tune → `adtc/training/` (TRL + PEFT QLoRA).

**Cloud vs laptop:** Train on cloud with Hugging Face (HF) weights. Convert merged checkpoint → GGUF for ADTC profiler / Gate 1. Laptop does not need every base model locally.

---

### Phase 0 — Packaging truth (Day 0–1) — BLOCKING — **DONE (Gate 0)**

Goal: prove the submission loop works with a vanilla GGUF before any training.


| #   | Task                                                                                               | Done when                                 |
| --- | -------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| 0.1 | Fork official ADTC submission template into team public repo                                       | Repo exists, team can push                |
| 0.2 | Fill draft `metadata.json`: domain `math_scientific_reasoning`, languages `en`+`am`, placeholders marked | Valid JSON; no silent wrong domain        |
| 0.3 | Install and **freeze** ADTC profiler version; document hash/version in repo notes                  | Same profiler used for all later runs     |
| 0.4 | Run participant mode on a tiny public GGUF (template default OK)                                   | Throughput + RSS + thermal numbers appear |
| 0.5 | Confirm `download_model.sh` → `model/*.gguf` matches `_runtime.model_path` from clean clone        | Fresh machine: clone → bash script → load |
| 0.6 | Decide hosting for final weights (HF public / release asset)                                       | Public URL strategy chosen                |


**Gate 0:** Clean checkout downloads and profiles. If this fails, stop—do not fine-tune. **Passed 16 Aug 2026.**

---



### Phase 1 — Language + eval freeze — **DONE (Gate 1a)**

**Amharic lock** — see [`LANGUAGE.md`](./LANGUAGE.md)
- Target language: Amharic (`am` / Iroko HF configs often `amh`)
- Fluent validator / owner: **Nathan Behailu**
- `african_alpha_claim: true` (contingent on validation samples before Gate 1 submit)
- Coverage: Amharic in IrokoBench (AfriMGSM / AfriMMLU / AfriXNLI)
- Frozen files: [`adtc/data/eval/FREEZE.md`](../data/eval/FREEZE.md)


| #   | Task                                                                                                                       | Done when                                       |
| --- | -------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| 1.1 | List languages someone can validate                                                                                        | **Done** — Nathan / Amharic                     |
| 1.2 | Cross-check Iroko / AfriQA / AfroBench coverage for that language                                                          | **Done** — `LANGUAGE.md`                        |
| 1.3 | Decide `african_alpha_claim` yes/no                                                                                        | **Done** — `true`, contingent on review         |
| 1.4 | Build English STEM held-out set                                                                                            | **Done** — `en_stem_holdout_v0.jsonl` (100)     |
| 1.5 | Pull / subset AfriMGSM, AfriMMLU, AfriXNLI for Amharic (`amh`)                                                             | **Done** — frozen JSONL + `eval_manifest_v0.json` |
| 1.6 | Write custom tutoring items (EN↔am); target 100–300                                                                        | **Done** — `custom_tutoring_v0.jsonl` (~100); expand to v1 before heavy train if needed |
| 1.7 | Build tokenizer fertility mini-set (parallel EN/am sentences)                                                              | **Done** — set + script; metrics deferred to Phase 2 |


**Gate 1a:** Language + frozen eval exist. No synthetic train data yet. **Passed 16 Aug 2026.**

---



### Phase 2 — Baseline screening — **DEFERRED (model-setup day / tomorrow)**

Run **unadapted** models. Prefer official GGUFs on the profiler laptop **or** profile later; HF bases for training live on the **cloud**. Measure EN-STEM, Amharic-STEM, TPS, RSS at Q4 and Q6.


| #   | Config                 | Priority                                                   |
| --- | ---------------------- | ---------------------------------------------------------- |
| 2.1 | Qwen3-1.7B Q4 + Q6     | Must                                                       |
| 2.2 | Qwen3-4B Q4 + Q6       | Must                                                       |
| 2.3 | Gemma 3 4B Q4 + Q6     | Must                                                       |
| 2.4 | Qwen2.5-3B-Instruct Q4 | If time                                                    |
| 2.5 | Qwen3.5-2B/4B GGUF     | **1-hour compatibility check only**—drop if profiler fails |


Also run:


| #   | Task                                            | Done when                                      |
| --- | ----------------------------------------------- | ---------------------------------------------- |
| 2.6 | Direct-X vs translate-test (X→EN) on same items | Diagnosis: language interface vs reasoning     |
| 2.7 | Token fertility for each family on X            | Fragmentation flag yes/no                      |
| 2.8 | Rank by ADTC-shaped score + note Pareto         | **Top efficiency** + **top accuracy** retained |


**Gate 2:** Two finalists named. Everything else parked.

---



### Phase 3 — Bilingual STEM data (Day 2–4) — **scaffolding now**

Build SFT data **after** eval freeze. Prefer educational STEM sources over bulk web crawl.

**Code:** catalog [`DATASETS.md`](./DATASETS.md); builders in `adtc/data/`; dedup in `adtc/eval/dedup_against_eval.py`.


| #   | Task                                                                                       | Done when                         |
| --- | ------------------------------------------------------------------------------------------ | --------------------------------- |
| 3.1 | Collect native Amharic text pool (grammar/register)—keep **separate** for ablation         | Versioned corpus A                |
| 3.2 | Collect / generate high-quality EN STEM tutoring examples                                  | Versioned corpus B                |
| 3.3 | Translate EN STEM → Amharic (MT); keep originals                                           | Parallel corpus                   |
| 3.4 | Fluent review of stratified translated samples (terms, algebra, units, negation, register) | Reject log + clean train set      |
| 3.5 | Format instruction data in four directions: EN→EN, am→am, EN→am, am→EN                     | Balanced mix documented           |
| 3.6 | Cover tutoring behaviors (not only final answers)                                          | Checklist covered in sample audit |
| 3.7 | Dedup against frozen eval prompts                                                          | Zero exact/near overlap           |
| 3.8 | Optional: teacher LLM (larger) generates/corrects examples—teacher **not** in submission   | Student-only train set            |


**Suggested CPT mixture ranges (only if Phase 4 triggers CPT):**


| Pool              | Search range |
| ----------------- | ------------ |
| Native Amharic    | 25–40%       |
| Translated am STEM | 25–35%      |
| EN math/science   | 15–25%       |
| EN/general replay | 5–15%        |
| Code/structured   | 5–15%        |


Start hypothesis near 35 / 25 / 20 / 10 / 10 if CPT runs.

**Gate 3:** Clean bilingual SFT set reviewed; eval untouched.

---



### Phase 4 — Adapt finalists — cloud GPU — **training code now; run after Phase 2**

**Code:** `adtc/training/` — TRL + PEFT QLoRA (`train_sft_qlora.py`, `merge_lora.py`, YAML configs). Pull HF bases on the **server**; do not require local GGUF for training.


| #   | Task                                                                                                                         | Done when                           |
| --- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| 4.1 | QLoRA/SFT **efficiency** finalist first (likely Qwen3-1.7B) on same data recipe                                              | Merged + unmerged checkpoints saved |
| 4.2 | Eval: EN-STEM, am-STEM, am-general, \Delta_{\text{forget}}, human sample                                                     | Numbers in results table            |
| 4.3 | Same recipe on **accuracy** finalist (4B Qwen or Gemma)                                                                      | Fair compare                        |
| 4.4 | **Decision:** Is direct-Amharic competence still weak? (large direct-am vs translate-test gap, poor prose, morphology, terminology) | Yes → 4.5; No → skip CPT     |
| 4.5 | (Conditional) Mixed CPT → SFT on the model that needs it                                                                     | CPT justified in report             |
| 4.6 | (Conditional) Tokenizer extension **only** if fertility severe                                                               | Otherwise skip                      |
| 4.7 | Merge adapters for deployment candidate(s); keep unmerged for reproducibility                                                | One merge per finalist              |


**Gate 4:** Adapted checkpoints measured. CPT/tokenizer only if triggered.

---



### Phase 5 — Quantization + Pareto (Day 5–6)

For **top adapted** checkpoint(s) only:


| #   | Task                                               | Done when                              |
| --- | -------------------------------------------------- | -------------------------------------- |
| 5.1 | Convert merged HF → high-precision GGUF            | Conversion verified loads in llama.cpp |
| 5.2 | PTQ: Q8, Q6, Q5, Q4 (same prompts/settings)        | Per-quant accuracy + TPS + RSS         |
| 5.3 | Compute \Delta A_q vs hardware deltas              | Table ready                            |
| 5.4 | Sustained profiler run (thermal)                   | No surprise throttle/OOM               |
| 5.5 | Plot accuracy vs TPS vs RSS; drop dominated points | Pareto set                             |
| 5.6 | Pick single submission GGUF                        | Winner named with score rationale      |


**Break-even reminder:** larger model must gain enough S_{\mathrm{acc}} to offset losses in perf + eff (≈ 0.6\Delta S_{\mathrm{perf}} + 0.4\Delta S_{\mathrm{eff}} required).

**Gate 5:** One GGUF chosen.

---



### Phase 6 — Optional science ablations (Day 6–7, if time)

Do **not** block submission on these; they strengthen the report.


| #   | Task                                                                             | Trigger                                                           |
| --- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 6.1 | Data-mixture ablation on one model: native → +trans STEM → +EN reasoning → +code | AfriqueLLM story                                                  |
| 6.2 | Distill 4B → 1.7B via teacher-generated tutoring data                            | 4B much smarter, 1.7B much faster                                 |
| 6.3 | QAT                                                                              | Q4 accuracy ≪ higher precision **and** Q4 still hardware-dominant |
| 6.4 | AfriqueLLM checkpoint as CPT-pre-adapted control                                 | Language covered + GGUF/profiler OK                               |


---



### Phase 7 — Package, report, demo (Day 7–9)


| #   | Task                                                                                                 | Done when                    |
| --- | ---------------------------------------------------------------------------------------------------- | ---------------------------- |
| 7.1 | Upload final GGUF to public host                                                                     | Stable URL                   |
| 7.2 | Update `download_model.sh` (idempotent, no creds)                                                    | Path = `_runtime.model_path` |
| 7.3 | Finalize `metadata.json` (team_id, languages, 2 prompts, claims, model fields)                       | No placeholders              |
| 7.4 | Write `REPORT.md` around **measured decisions** (screen → diagnose → adapt → quant → Pareto)         | Narrative matches tables     |
| 7.5 | Clean-clone dry run: download → profile → sample EN + X tutoring                                     | Pass                         |
| 7.6 | Record ≤2 min video: specs, airplane mode, same problem EN+X, tutoring not just QA, profiler RAM/TPS | Uploaded                     |
| 7.7 | Submission checklist from template README                                                            | All boxes                    |
| 7.8 | Submit repo URL on Devpost                                                                           | Confirmation saved           |


**Report thesis (copy into REPORT.md):**  
We evaluated compact architectures, diagnosed language vs reasoning bottlenecks, measured translated STEM + EN replay, swept quantizations, and selected the accuracy–throughput–memory frontier—not the largest model that fits.

---



## 6. Experiment tournament (do not Cartesian-explode)

```
Screen (8 cheap configs) → keep top efficiency + top accuracy
  → base vs bilingual SFT
    → CPT only if language still weak
      → quant sweep on winner(s)
        → distill only if 4B/1.7B tradeoff demands it
```

---



## 7. Roles & ownership (fill in)


| Role                              | Owner           | Notes                          |
| --------------------------------- | --------------- | ------------------------------ |
| Submission / packaging / profiler | Nathan Behailu  | Phase 0 + 7                    |
| Language validation               | Nathan Behailu  | Amharic                        |
| Data + translation QC             |                 |                                |
| Training (QLoRA / CPT)            |                 | Cloud GPU                      |
| Eval harness + tables             |                 |                                |
| Report + video                    |                 |                                |


---



## 8. Risks and mitigations


| Risk                           | Mitigation                                                |
| ------------------------------ | --------------------------------------------------------- |
| Packaging fails late           | Phase 0 before training                                   |
| Overfit 2 demo prompts         | Broad STEM competence; assume ≥3 hidden prompts           |
| No one can validate X          | Change language or add collaborator before claiming alpha |
| CPT burns GPU budget           | Conditional only; SFT-first                               |
| Qwen3.5 shiny but incompatible | Profiler gate before adaptation                           |
| Thermal throttle on sustained  | Sustained profiler runs before lock                       |
| Q4 hurts small models more     | Always measure Q8→Q4 drop on EN and X                     |


---



## 9. Daily standup questions

1. Did packaging still work yesterday?
2. Which gate are we on, and what measurement unlocks the next?
3. What are we **not** doing this week (CPT / distill / QAT / tokenizer)?

---



## 10. Quick reference — ADTC packaging

```
submission/
├── metadata.json       # domain, 2 prompts, _runtime.model_path
├── download_model.sh   # public URL → model/*.gguf
├── REPORT.md
├── model/              # gitignored
│   └── *.gguf
└── .gitignore
```

Runtime: **llama.cpp only**. Judges talk to the model, not an app stack.