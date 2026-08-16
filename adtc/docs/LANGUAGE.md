# Language lock — English + Amharic

**Status:** Phase 1 decisions frozen (16 Aug 2026)  
**Do not change target language without a new `LANGUAGE.md` version and PRD update.**

---

## 1.1 Validator

| Role | Person | Language |
|------|--------|----------|
| Fluent validator / owner | **Nathan Behailu** | Amharic |
| Primary evaluation language | — | English |

Nathan reviews scientific terminology, register, algebra phrasing, and tutoring quality in Amharic before Gate 1 submission.

---

## 1.2 Benchmark coverage

| Suite | Amharic covered? | Role for this project |
|-------|------------------|------------------------|
| **IrokoBench** AfriMGSM | Yes (`amh`) | Primary multilingual math |
| **IrokoBench** AfriMMLU | Yes (Masakhane / Iroko) | Knowledge / STEM |
| **IrokoBench** AfriXNLI | Yes | General language understanding |
| AfriQA | Check if Amharic present | Optional cross-lingual QA |
| AfroBench | Broad African multi-task | Optional sanity check |

**Rationale:** Amharic is East African, Ge’ez script, and appears in IrokoBench—so we can evaluate math and knowledge without inventing a private benchmark. Validator capacity (Nathan) outweighs “dataset fame” languages nobody on the team can check.

**Codes**

| Context | Code |
|---------|------|
| `metadata.json` `language_scope` | `am`, `en` (BCP-47) |
| Masakhane / Iroko HF configs | often `amh`, `eng` |

---

## 1.3 African alpha claim

- Draft `metadata.json`: **`african_alpha_claim: true`**
- **Justification:** fluent Amharic validation exists on the team; product is bilingual EN+Amharic STEM tutoring for African users offline.
- **Contingent:** claim stays only if Nathan reviews a stratified sample of (a) custom tutoring Amharic rows and (b) model outputs before final Gate 1 submit. If quality is inadequate, flip to `false` rather than ship a cosmetic claim.

---

## Fertility (1.7)

- Parallel set: `adtc/data/eval/fertility_parallel_v0.jsonl`
- Script: `adtc/eval/fertility.py`
- **Metrics** (`F_am`, `R_am/en`): run in Phase 2 when an HF tokenizer is available on cloud — not required to close Gate 1a.

```bash
python eval/fertility.py --tokenizer Qwen/Qwen3-1.7B
```

---

## Eval freeze pointer

See `adtc/data/eval/FREEZE.md` for frozen file list and version rules. Never train on these files.
