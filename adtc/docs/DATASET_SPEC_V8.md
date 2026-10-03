# TebebAI v8 Dataset Specification

**Status:** proposed implementation specification  
**Scope:** English secondary-school STEM tutoring SFT  
**Primary objectives:** factual/reasoning accuracy and pedagogically useful tutoring behavior  
**Out of scope:** adding data solely to improve runtime, memory use, or tokens/second

This document defines the next training dataset for TebebAI. It replaces source-count-driven mixing with a curriculum matrix, preserves provenance and licensing at row level, and adopts organization methods used in MathDial, Bridge, LearnLM, FEAT, MRBench, and MathTutorBench.

The executable training export remains ordinary chat JSONL. A richer canonical dataset is retained separately so that every answer can be audited, filtered, re-rendered, and evaluated by subject, behavior, error type, and source.

## 1. Why v8 is needed

The v7 mix contains 10,636 examples:

| Component | Rows | Approximate share | Main limitation |
|---|---:|---:|---|
| GSM8K | 7,473 | 70.3% | Mostly grade-school mathematics |
| SciQ | 3,000 | 28.2% | Science coverage, but shallow conversion |
| Authored tutoring | 163 | 1.5% | Too small to determine model behavior |

The mix teaches substantially more mathematical solution imitation than scientific reasoning or tutoring. The current SciQ conversion also creates generic hints such as asking the learner to reread the question; it does not consistently encode the scientific fact, misconception, next reasoning step, or answer-withholding constraint that makes a hint useful.

The v8 dataset must address four separate capabilities:

1. **Subject knowledge:** mathematics, physics, chemistry, biology, and Earth science as five separately measured subjects.
2. **Reasoning correctness:** valid steps, calculations, units, causal explanations, and final answers.
3. **Student-state diagnosis:** recognizing the first error or underlying misconception in a student's attempt.
4. **Pedagogical action:** choosing whether to focus attention, probe, explain, or withhold the answer and give a hint.

More data is not automatically better. Every included row must have a known license, traceable origin, isolated evaluation relationship, and an independently checkable target.

## 2. Research-backed design principles

### 2.1 Represent the tutoring decision, not only the final reply

[Bridge](https://aclanthology.org/2024.naacl-long.120/) decomposes a tutor turn into three decisions: identify the student error, select a remediation strategy, and state the pedagogical intention before generating the response. Expert decisions improved tutor responses, whereas random decisions degraded them. TebebAI therefore records `error_type`, `misconception`, `teacher_move`, and `intent` as separate fields.

### 2.2 Use a stable taxonomy of teacher moves

[MathDial](https://aclanthology.org/2023.findings-emnlp.372/) organizes teacher responses into four broad moves—Focus, Probing, Telling, and Generic—and finer intentions such as guiding attention, requesting an explanation, seeking self-correction, recalling relevant information, and revealing a strategy. TebebAI uses this hierarchy rather than treating every instructional answer as undifferentiated prose.

MathDial also distinguishes conceptual errors from arithmetic errors. Its dialogues contain substantially more conceptual confusions than arithmetic ones. For diagnosis examples, v8 adopts an initial 80/20 conceptual-versus-procedural target, then measures whether the distribution improves held-out results rather than treating it as permanent.

### 2.3 Give the model explicit pedagogical instructions

[LearnLM](https://arxiv.org/abs/2412.16429) reports that pedagogical instruction following benefits from specific system-level instructions and from co-training pedagogical data with general post-training data. Vague instructions are less useful. Each v8 example therefore states the desired behavior precisely—for example, “identify the first incorrect step, ask one question that helps the student repair it, and do not reveal the final numerical answer.”

The applicable learning principles are stored as labels: active learning, managing cognitive load, adapting to the learner, stimulating curiosity, and deepening metacognition. They are not required to appear as jargon in the generated response.

### 2.4 Use a stronger answer-expansion model and a separate verifier

[FEAT](https://aclanthology.org/2025.acl-short.45/) shows the value of mixing a smaller, higher-quality tier with generated data. TebebAI adapts that organization method to an AI-curated workflow: after the pipeline assigns a source row to a `solve`, `explain`, `hint`, or `diagnose` bucket, a pinned frontier model expands the row into the required answer format, and a separate pinned model pass verifies the result. Because these examples are not human-reviewed, this specification calls them **model-curated**, not “human gold.”

All rows receive source-grounded answer expansion and separately prompted verification. In addition, 10% form a **high-assurance reference tier** with an additional adjudication pass. The build defines three model roles—generator, verifier, and adjudicator—but does not hard-code unverified marketing names or Cursor-internal model slugs. Before the QA pilot, an API preflight must prove that each configured ID is callable with the project key; the exact provider, callable model ID, snapshot/version when available, endpoint, and prompt version are then pinned in `build_manifest.json`. A model-list preflight that cannot run because the key lacks model-read scope is not evidence that an ID exists; in that case a minimal non-production request must validate each configured ID.

When all three roles use OpenAI models, reports must label the setup **same-provider, reduced independence**. A cross-provider verifier may replace a role only when its exact callable model ID and terms are pinned in the manifest. Deterministic solvers remain hard gates and override model agreement. The v7-versus-v8 pairwise judge is also labeled same-provider whenever an OpenAI judge evaluates OpenAI-generated training data.

### 2.5 Separate subject expertise, student understanding, and teaching behavior

[MathTutorBench](https://aclanthology.org/2025.emnlp-main.11/) evaluates subject expertise, understanding of a student's solution, and response generation/scaffolding as distinct capability groups. TebebAI keeps `solve`, `explain`, `diagnose`, and `hint` as distinct behavior labels and reports each separately. A high aggregate score cannot hide a model that solves problems but fails as a tutor.

### 2.6 Preserve hard tutoring benchmarks for evaluation

[MRBench](https://aclanthology.org/2025.naacl-long.57/) and the [BEA 2025 Shared Task](https://aclanthology.org/2025.bea-1.77/) provide useful dimensions such as mistake identification, mistake location, guidance, and actionability. Their taxonomies should inform the schema and rubric, but their benchmark examples should remain evaluation-only.

## 3. Canonical row schema

Canonical records are stored as JSONL in `data/canonical/v8/`. Required fields are marked **R**; conditionally required fields are marked **C**.

```json
{
  "schema_version": "tebeb-v8.1",
  "id": "detstem_stoich_004219__diagnose__v1",
  "split": "train",
  "source": {
    "dataset": "tebeb/deterministic-stem-v1",
    "revision": "PINNED_COMMIT_OR_DATA_HASH",
    "config": "default",
    "source_split": "train",
    "original_id": "stoich_limiting_reactant_004219",
    "url": "repo:data/generators/stem_v1",
    "license": "project-generated",
    "license_class": "permissive",
    "export_profile": "permissive",
    "transformation": "detstem_controlled_error_diagnose_v1"
  },
  "curriculum": {
    "subject": "chemistry",
    "secondary_subjects": ["math"],
    "topic": "stoichiometry",
    "subtopic": "limiting_reactant",
    "grade_band": "secondary_9_12",
    "difficulty": "medium",
    "learning_objective": "Use mole ratios to identify the limiting reactant."
  },
  "pedagogy": {
    "behavior": "diagnose",
    "teacher_move": "probe",
    "intent": "seek_self_correction",
    "principles": ["active_learning", "manage_cognitive_load"],
    "answer_policy": "withhold_final",
    "question_count": 1,
    "interaction_mode": "single_turn"
  },
  "grounding": {
    "problem": "...",
    "canonical_answer": "33 g CO2",
    "solution_steps": ["...", "..."],
    "supporting_facts": ["..."],
    "equations": ["2 CO + O2 -> 2 CO2"],
    "units": ["mol", "g"],
    "accepted_variants": ["33 grams", "33 g"]
  },
  "student_state": {
    "attempt": "...",
    "first_error_step": 3,
    "error_type": "ratio_error",
    "error_class": "conceptual",
    "error_class_rationale": "The mole relationship, rather than the arithmetic operation, is misunderstood.",
    "misconception": "The student used a 2:2 ratio for O2 to CO2."
  },
  "constraints": {
    "items": [
      {"id": "identify_first_error", "check": "semantic_verifier"},
      {"id": "explain_why_wrong", "check": "semantic_verifier"},
      {"id": "give_one_corrective_hint", "check": "semantic_verifier"},
      {"id": "exactly_one_followup_question", "check": "question_count"},
      {"id": "no_final_numeric_answer", "check": "numeric_leak_v2"}
    ],
    "forbidden_answers": ["33 g CO2"],
    "constraint_results": {
      "identify_first_error": {"passed": true},
      "explain_why_wrong": {"passed": true},
      "give_one_corrective_hint": {"passed": true},
      "exactly_one_followup_question": {"passed": true, "observed": 1},
      "no_final_numeric_answer": {"passed": true, "matched_variants": []}
    },
    "all_constraints_pass": true
  },
  "messages": [
    {"role": "system", "content": "Act as a secondary-school chemistry tutor. Identify the first incorrect step, ask exactly one focused question, and do not reveal the final answer."},
    {"role": "user", "content": "...student problem and attempt..."},
    {"role": "assistant", "content": "...verified tutor response..."}
  ],
  "quality": {
    "generator": "provider/CALLABLE_GENERATOR_ID@prompt-v8.1",
    "verifiers": ["deterministic_v1", "provider/CALLABLE_VERIFIER_ID@verify-v8.1"],
    "factual_score": 1.0,
    "pedagogy_score": 1.0,
    "constraint_pass": true,
    "curator_review": "accepted",
    "curator_model": "provider/CALLABLE_ADJUDICATOR_ID@adjudicate-v8.1",
    "model_independence": "same-provider-reduced-independence",
    "review_passes": 2,
    "assurance_tier": "high",
    "selection": {
      "decision": "accept",
      "subject_method": "source_metadata",
      "subject_distribution": {
        "math": 0.01,
        "physics": 0.01,
        "chemistry": 0.96,
        "biology": 0.01,
        "earth_science": 0.00,
        "no_match": 0.01
      },
      "subject_choice_confidence": 0.95,
      "behavior_eligibility": {
        "solve": {"eligible": true, "p_yes": 0.99},
        "explain": {"eligible": false, "p_yes": 0.18},
        "hint": {"eligible": true, "p_yes": 0.94},
        "diagnose": {"eligible": true, "p_yes": 0.90}
      },
      "behavior_classifier": "typesafe/jev-1.13.0@policy-v8.1",
      "typesafe_sdk": "typesafe-sdk==0.7.2",
      "selected_behavior": "diagnose",
      "scores": {
        "grounding": 2,
        "answer_verifiability": 2,
        "curriculum_fit": 2,
        "behavior_affordance": 2,
        "clarity": 2,
        "diversity_value": 1
      },
      "total": 11,
      "rejection_reasons": []
    }
  },
  "identity": {
    "source_key": "tebeb/deterministic-stem-v1@PINNED_COMMIT/train/stoich_limiting_reactant_004219",
    "family_id": "sha256:...",
    "row_id": "sha256:...",
    "record_version": 1,
    "parent_row_ids": []
  },
  "dedup": {
    "normalization_version": "norm-v1",
    "canonical_content_hash": "sha256:...",
    "prompt_hash": "sha256:...",
    "assistant_hash": "sha256:...",
    "constraint_signature": "sha256:...",
    "template_fingerprint": "sha256:...",
    "semantic_cluster_id": "cluster_...",
    "nearest_eval_similarity": 0.31,
    "duplicate_of": null,
    "decision": "keep"
  }
}
```

### 3.1 Field requirements

| Field | Requirement | Purpose |
|---|---|---|
| `schema_version` | **R** | Makes future migrations explicit. |
| `id` | **R**, globally unique | Stable identity for auditing and exclusions. |
| `split` | **R** | One of `train`, `validation`, `test`, `external_eval`. |
| `source.*` | **R** | Provenance, license, pinned revision, and transformation recipe. |
| `curriculum.*` | **R** except `subtopic` and `secondary_subjects` | Coverage reports and family-level splitting. |
| `pedagogy.*` | **R** | Behavior-conditional training and evaluation. |
| `grounding.*` | **R** | Correctness verification; `canonical_answer` may be null for Explain only when verified supporting claims and a reference explanation are present. It is not necessarily exposed to the model. |
| `student_state.*` | **R** for `diagnose`; **C** for `hint` | Every Diagnose row includes an attempt. For the 10–15% whose attempt is correct, use `first_error_step: null`, `error_type: "none"`, `error_class: null`, and an explicit no-error rationale; Hint uses student state when the task is attempt-conditioned. |
| `constraints.*` | **R** | Enables deterministic instruction-following checks. |
| `messages` | **R** | The chat-format training target. |
| `quality.*` | **R** | Makes generation, routing, scoring, and acceptance reproducible. |
| `identity.*` | **R** | Gives each source family and generated variant a stable identity across pilot and full builds. |
| `dedup.*` | **R** | Detects exact, templated, cross-source, and semantic repetition and keeps related rows in one split. |

The training renderer exports only `messages` plus minimal identifiers. Canonical answers, verifier notes, and hidden rubrics remain in metadata. In hint and diagnosis examples, the canonical answer must not be inserted into the user-visible system or user prompt unless the exercise explicitly asks the tutor to critique an already revealed answer.

### 3.2 Atomic constraint library

Multi-part requirements come from a versioned library rather than free-form generation. Each constraint entry defines a stable ID, 5–10 paraphrased renderings, compatible behaviors, required/incompatible constraints, parameter schema, and a deterministic or semantic checker. Core groups are withholding, structure, diagnosis, content, audience, and local context.

Examples include `no_final_numeric_answer`, `give_one_corrective_hint`, `identify_first_error`, `explain_why_wrong`, `exactly_one_followup_question`, `end_with_two_check_questions`, `audience_age_15`, `distinguish_named_concepts`, and `use_verified_african_context_analogy`. Compatibility rules forbid combinations such as `give_full_solution` plus `no_final_numeric_answer` and require withholding to co-occur with useful scaffolding.

The renderer varies numbered lists, prose requirements, system/user splits, no-system prompts, and embedded student attempts while retaining the same constraint IDs. A row is accepted only when every `constraint_results` entry passes. Order is enforced only when a constraint explicitly requests ordering.

Forbidden-answer checks are number-aware rather than substring-based. They tokenize numerical expressions and detect equivalent forms such as `9`, `9.0`, `nine`, and `18/2` using exact numeric/symbolic equivalence plus unit normalization. The string `33` must not match `133` or an unrelated identifier merely by substring.

African-context analogies are selected from a structured, AI-generated, independently verified context bank. The expansion model may adapt a selected context row but may not invent local factual claims without verification.

### 3.3 Controlled vocabularies

**Subjects**

- `math`
- `physics`
- `chemistry`
- `biology`
- `earth_science`

**Behaviors**

- `solve`: produce a correct solution with justified steps.
- `explain`: explain a concept, mechanism, distinction, or worked solution at the requested level.
- `hint`: provide the smallest useful next step while respecting the answer policy.
- `diagnose`: identify the first incorrect step or misconception and guide repair.

**Teacher moves**

- `focus`: direct attention to relevant information or the location of an error.
- `probe`: elicit reasoning, explanation, recall, or self-correction.
- `tell`: reveal a fact, strategy, worked step, or answer when permitted.
- `generic`: acknowledge, encourage, or manage the interaction without adding content.

**Fine-grained intentions**

- `guide_focus`
- `seek_strategy`
- `seek_explanation`
- `seek_self_correction`
- `recall_relevant_fact`
- `perturb_problem`
- `reveal_strategy`
- `reveal_step`
- `reveal_answer`
- `encourage`
- `check_understanding`
- `greet_or_close`
- `general_inquiry`

**Move-to-intent validation**

| Teacher move | Allowed intentions |
|---|---|
| `focus` | `seek_strategy`, `guide_focus` |
| `probe` | `recall_relevant_fact`, `seek_explanation`, `seek_self_correction`, `perturb_problem`, `check_understanding` |
| `tell` | `reveal_strategy`, `reveal_step`, `reveal_answer` |
| `generic` | `encourage`, `greet_or_close`, `general_inquiry` |

This mapping follows MathDial's taxonomy; in particular, `seek_self_correction` is a Probing intent, not a Focus intent. The schema validator rejects invalid move-intent pairs.

**Error types**

- No error: `none` (valid only when the student's attempt is verified correct).
- Cross-domain: `concept_confusion`, `wrong_assumption`, `unsupported_claim`, `instruction_miss`, `premature_conclusion`.
- Quantitative: `arithmetic_error`, `algebra_error`, `ratio_error`, `unit_error`, `sign_error`, `sign_convention_error`, `coordinate_choice_error`, `formula_selection_error`, `transcription_error`.
- Science: `causal_reversal`, `category_error`, `scale_error`, `conservation_error`, `equilibrium_error`, `representation_error`.

Every erroneous attempt also has an explicit `error_class` of `conceptual` or `procedural` plus a short rationale. `arithmetic_error` and `transcription_error` default to procedural. `concept_confusion`, `wrong_assumption`, `causal_reversal`, `category_error`, `conservation_error`, `equilibrium_error`, `formula_selection_error`, `sign_convention_error`, and `coordinate_choice_error` default to conceptual. `algebra_error`, `ratio_error`, `unit_error`, `sign_error`, `scale_error`, and `representation_error` are context-dependent and require verifier adjudication rather than an automatic class. Correct-attempt Diagnose rows use `error_type: "none"` and `error_class: null`; they are reported separately and excluded from the 80/20 denominator. The 80/20 target is computed from non-null `error_class`, not inferred from `error_type`.

## 4. Coverage and mixture

### 4.1 Balance cells, not source datasets

The first v8 release targets 10,000 accepted rows arranged as a 5 × 4 curriculum matrix. Each subject-behavior cell contains 500 rows.

| Subject | Solve | Explain | Hint | Diagnose | Total |
|---|---:|---:|---:|---:|---:|
| Mathematics | 500 | 500 | 500 | 500 | 2,000 |
| Physics | 500 | 500 | 500 | 500 | 2,000 |
| Chemistry | 500 | 500 | 500 | 500 | 2,000 |
| Biology | 500 | 500 | 500 | 500 | 2,000 |
| Earth science | 500 | 500 | 500 | 500 | 2,000 |
| **Total** | **2,500** | **2,500** | **2,500** | **2,500** | **10,000** |

This corrects v7's math dominance and lets us attribute an improvement or regression to a subject and behavior. It does **not** require equal row counts from each external dataset. MathDial and Bridge are retained for tutoring structure, while the filtered STEM and EXAMS sources supply high-school content breadth and deterministic generators supply checkable quantitative depth. Source contributions follow quality, level, provenance, and licensing, while the final learning objectives remain balanced.

Biology and Earth-science supply is a blocking uncertainty, not an assumption. STEM lacks trusted worked explanations, EXAMS requires verified translation, EduAdapt is not approved by default, and the initial deterministic generators cover only Mathematics, Physics, and Chemistry. Before any bulk model spend, the build must complete the supply census in Section 11 and report hard-gate-eligible source families for every subject × behavior cell. If Biology or Earth science cannot support the balanced minimum, the team must make a separate, documented go/no-go decision to admit a licensed grounding-only reference corpus, implement solver-backed generators for suitable topics, or revise the curriculum matrix. The quota controller must not fill those cells with circular model-written explanations verified only by the same provider.

### 4.2 Model-based answer expansion and high-assurance allocation

Each cell initially contains:

- 500 source-grounded rows expanded by the pinned generator after bucket assignment and accepted by the pinned verifier;
- within those 500, 50 high-assurance rows adjudicated by an additional frontier model, preferably from a different model family.

This yields 10,000 model-curated rows, including 1,000 high-assurance rows. High assurance requires independent review of the question, canonical solution, student error, tutor strategy, and final response—not merely approval of fluent wording. Chemistry and physics quantitative rows additionally require deterministic calculation, unit, and equation checks because agreement between two language models is not proof of scientific correctness.

The model stack is the API-preflighted role configuration described in Section 2.4. If no cross-provider verifier is available, the build may proceed with pinned same-provider models, but `model_independence` must say `same-provider-reduced-independence` in every affected row and report. Marketing-family names without a callable model ID are not reproducible configurations.

### 4.3 Difficulty and error distribution

Within each subject-behavior cell:

- 30% easy, 50% medium, 20% hard as an initial curriculum target;
- no topic contributes more than 15% of a cell;
- no transformation template contributes more than 20% of a cell;
- diagnosis rows target roughly 80% conceptual and 20% arithmetic/procedural errors;
- every `diagnose` row contains an explicit student attempt, and 10–15% of those attempts are actually correct so the tutor must say that no error exists;
- at least 30% of `hint` rows use an explicit student attempt;
- at least 20% of all rows require two or more linked reasoning steps.
- at least 25% of all rows contain three or more independently checked instruction constraints; `hint`, `diagnose`, and `explain` skew higher than `solve`;
- at least 40% of each Physics/Chemistry `solve`, `hint`, and `diagnose` cell is quantitative and multi-step, including sign, unit, ratio, or limiting-reactant reasoning where applicable;
- at least 5% of `explain` rows request an African-context analogy selected from a separately verified structured context bank.

These are starting hypotheses. Ablations should change one factor at a time.

### 4.4 Short-cell policy

The preferred target is 500 rows per cell. After licensing and quality gates, a production v8 build requires at least 475 valid rows in every cell. If any cell has fewer than 475, the build may not claim the 10,000-row matrix.

The team must either add valid sources/generators or freeze a smaller balanced target `N` equal to the smallest valid cell, provided `N >= 400`. Every cell is then sampled to `N`, and the release is named and reported using that target. If any cell remains below 400, the full build is blocked and remains a pilot. This replaces the contradictory policy of allowing a 430-row cell while also requiring at most 5% deviation from 500.

### 4.5 Candidate selection and routing

A source record is first treated as a **canonical content family**, not as a finished `solve`, `explain`, `hint`, or `diagnose` row. Selection has three separate decisions:

1. Is the source record trustworthy and useful enough to keep?
2. What is its primary subject?
3. Which tutoring behaviors can be derived from it without inventing unsupported material?

The pipeline must not automatically turn every source question into all four behaviors. Doing so would create the appearance of balance while repeating the same knowledge four times and forcing weak hints or artificial misconceptions from unsuitable questions.

#### Step A: hard eligibility filters

A candidate is rejected before scoring if any of the following is true:

- its license, revision, original ID, or source split is unknown;
- it comes from a validation/test split reserved for evaluation;
- the question is broken, ambiguous, trivial, or missing information;
- its required target cannot be independently verified: `solve`, `hint`, and `diagnose` require a canonical answer/solution, while `explain` may set `canonical_answer` to null but requires verified supporting claims and a reference explanation;
- required figures, tables, or experiment context are unavailable;
- it falls outside the target secondary-school level without a documented reason;
- it is an exact or near duplicate of frozen evaluation or an already selected family;
- its scientific explanation cannot be grounded in the source or another approved reference.

#### Step B: metadata-first subject assignment

Subject assignment is normally deterministic because many row-structured datasets already provide a subject, topic, category, skill, book, or configuration. The importer maps those source fields into the five TebebAI subjects using a versioned lookup table. No model call is needed when the source label is present and unambiguous.

Use [Jev](https://docs.typesafe.ai/models) only when subject metadata is missing, too broad (for example, `natural_science`), ambiguous, or conflicting. Subject routing is one TypeSafe Choice question over the five subjects plus `no_match`. Store the selected option, full probability distribution, and Choice confidence, which measures concentration among competing options rather than factual correctness. Low-confidence, `no_match`, or conflicting cases go to the adjudicator; they are not silently accepted.

Priority order:

1. explicit source subject/topic metadata;
2. deterministic dataset/configuration mapping;
3. keyword or ontology rules for narrow, unambiguous categories;
4. Jev subject classification for unresolved rows;
5. frontier-model adjudication for low-confidence or conflicting cases.

Jev is used for unresolved subject categorization and multi-label behavior eligibility. It does not make the final quota-aware bucket assignment, write tutoring responses, prove an answer correct, or validate scientific reasoning.

The **central knowledge required to answer** determines the primary subject, not incidental words in the story:

| Primary subject | Routing rule |
|---|---|
| `math` | The central work is arithmetic, algebra, geometry, probability, statistics, or proof and does not require a scientific law. |
| `physics` | The answer depends mainly on motion, forces, energy, waves, electricity, magnetism, heat, or a physical law/model. |
| `chemistry` | The answer depends mainly on substances, atomic/molecular structure, bonding, reactions, stoichiometry, acids/bases, or chemical equilibrium. |
| `biology` | The answer depends mainly on living systems, cells, genetics, evolution, physiology, ecology, or biological mechanisms. |
| `earth_science` | The answer depends mainly on geology, weather, climate, oceans, Earth systems, natural hazards, or school-level Earth/space science. |

Cross-disciplinary items keep one primary subject for quota accounting and one or more secondary tags. For example, calculating kinetic energy is physics even though it uses algebra; population-growth modeling is biology when biological interpretation is essential, and math when it is a context-free function exercise.

#### Step C: hard behavior checks, Jev eligibility, and bucket assignment

The build pipeline first applies hard requirements from the structured fields already present in the canonical row. It then asks Jev four independent TypeSafe Noul questions in one request—one for each behavior—and records `p_yes` plus the thresholded eligibility decision for every bucket. Noul returns the probability of yes; it does not return a separate confidence value. This is multi-label classification: a row may qualify for several behaviors or none.

| Behavior | A candidate is eligible only when… | Typical source material |
|---|---|---|
| `solve` | It has an unambiguous task, canonical answer, and reproducible reasoning or calculation path. | Exercises, word problems, answerable MCQs after removing answer-position artifacts. |
| `explain` | It contains a concept, mechanism, distinction, or causal chain supported by approved grounding. A final numerical answer is not required. | Structured support fields, lesson rows, worked examples, and science QA with evidence. |
| `hint` | There is a meaningful intermediate idea or next step that can be revealed without leaking the answer. One-step recall questions usually do not qualify. | Multi-step problems, conceptual comparisons, partially completed solutions. |
| `diagnose` | An explicit attempt is available. For erroneous attempts, the first error can be located uniquely; for the controlled correct-attempt slice, every step can be verified and the target is to avoid inventing an error. A distractor alone is insufficient unless its misconception is verified. | Worked solutions, tutoring dialogues, structured distractors, common-misconception material. |

Jev receives the canonical question, answer, solution/support fields, available student attempt or distractors, and the behavior definitions above. Each Noul question states its full behavior definition and hard exclusions because questions in one request run independently and cannot see one another's answers. A behavior is ineligible whenever its hard requirement fails, regardless of Jev's probability. For behaviors that pass the hard gate, accept eligibility only when `p_yes` meets the calibrated per-behavior threshold. Values in the calibrated uncertainty band around the threshold go to adjudication.

Deterministic-generator rows with behavior eligibility proven by their template contract and MathDial/Bridge turns with an explicit source teacher move may use `subject_method` or `behavior_method: "deterministic_source"` and skip the corresponding Jev question. The manifest reports both Jev-routed and deterministically routed counts; deterministic routing is not represented as a fabricated Jev result.

A family may be eligible for zero, one, or several behaviors. By default, select no more than **two behavior variants from one family** for training. This preserves knowledge diversity: the 10,000 rows should come from at least 5,000 canonical families. Exceptions are allowed for unusually rich, real tutoring dialogues, but must be reported.

The quota controller selects among the eligible buckets using the remaining subject × behavior targets. For example, a row with a verified canonical answer and multi-step solution is eligible for `solve` and `hint`; if the Math/Solve cell is already full but Math/Hint is short, it is assigned to `hint`. The selected bucket is stored before response generation.

#### Step D: per-behavior quality score

Eligible candidates receive a 0–2 score on each dimension below, for a maximum of 12:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Grounding | Missing/unsupported | Partial | Direct, sufficient evidence |
| Answer verifiability | Unverifiable | Model-only agreement | Deterministic or strong independent verification |
| Curriculum fit | Wrong level/topic | Borderline | Clear secondary-school objective |
| Behavior affordance | Forced/artificial | Usable with repair | Natural fit for the behavior |
| Clarity | Ambiguous/broken | Minor cleanup | Clear and self-contained |
| Diversity value | Near-duplicate/common | Adds some variety | Fills an underrepresented topic/error/difficulty |

Accept a candidate-behavior pair only when:

- total score is at least 9/12;
- Grounding, Answer verifiability, Curriculum fit, and Behavior affordance are all non-zero;
- the hard bucket rules pass, Jev's Noul `p_yes` meets the calibrated behavior threshold, and the separately prompted verifier accepts the generated result, or adjudication resolves an uncertainty-band case or verifier disagreement.

The score is used for ranking within a cell, not as a substitute for the hard gates. The preferred implementation asks six independent TypeSafe Score questions over the same state in one Jev request, with concrete 0/1/2 criteria matching the table. Store each level distribution and derived score, then let code compute the total and apply the non-compensating hard gates. This follows TypeSafe's composite-scoring pattern and avoids an extra generative-model scoring pass. Validate the scores against the calibration set before using them for selection.

#### Step E: quota-aware selection

For each of the 20 subject × behavior cells:

1. Build a candidate pool at least three times larger than the required 500 rows when source availability permits.
2. Select rare topics, error types, and hard examples first.
3. Fill the remainder using weighted stratified sampling over source, topic, difficulty, and transformation recipe.
4. Enforce the 15% topic cap and 20% transformation-template cap.
5. Target no more than 40% of a cell from one source dataset; document any exception caused by limited licensed coverage.
6. Select the 50 highest-value, most diverse accepted rows for high-assurance adjudication rather than simply taking the 50 highest quality scores from one topic.
7. Save every accepted and rejected candidate with its routing decision, scores, model versions, and rejection reason.

Selection is therefore quota-aware but never quota-forced. If Earth-science diagnosis has only 430 valid examples, the 500-row target is blocked and the short-cell policy either requires new valid supply or resizes every cell to the same declared target.

#### Routing examples

| Source item | Primary subject | Eligible behaviors | Why |
|---|---|---|---|
| Solver-backed inclined-plane problem | Physics | Solve, Hint, Diagnose | The numerical solution, units, sign convention, and planted first error are independently checkable. |
| Filtered STEM grade 9–12 text-only question | Determined from grade and skill metadata, with Jev only if ambiguous | Solve; possibly Explain or Hint | The source supplies an authentic curriculum skill and answer choice, but explanation-based behaviors require separately verified grounding. |
| MATH/NuminaMath algebra problem | Math | Solve, Hint, Diagnose | Retain only target-level rows with symbolically or numerically checkable answers; exclude benchmark-family matches. |
| EXAMS high-school science item | Physics, Chemistry, Biology, or Earth science | Solve; possibly Explain | Translation, if required, and the generated reasoning must be separately verified; MCQ distractors are not automatically misconceptions. |
| MathDial turn containing a student's wrong step | Math | Diagnose, Hint | The authentic student state makes diagnosis natural; converting it into a separate solve row may duplicate its underlying problem. |

## 5. Dataset decision matrix

License claims must be rechecked against the pinned revision before download or training. “Transform” means the raw dataset is useful, but its existing answer format does not yet meet the tutoring schema.

### 5.0 Row-structured source policy

The default v8 build uses datasets already organized as records in JSON, JSONL, Parquet, or an equivalent table. Preferred source fields include stable ID, split, question, answer, choices or student attempt, supporting facts/explanation, and subject/topic tags. All approved sources are normalized into the canonical row schema and may also be exported as Parquet for querying.

Only structured, row-based sources are eligible. The primary content stack is deterministic STEM generators, filtered STEM train rows, MATH train, a selectively audited NuminaMath slice, and conditional EXAMS train rows. MathDial and Bridge contribute tutoring behavior rather than broad subject coverage. The v8 pipeline does not ingest unstructured books or chapter extracts.

| Dataset | Coverage | Decision | What it can supply | Required modification / restriction |
|---|---|---|---|---|
| **Deterministic STEM generators v1** | Secondary/early-university algebra, precalculus, physics, chemistry | **Primary quantitative source** | Parameterized problems, exact solutions, units, controlled first-error attempts | Generate structured rows for exponential substitution, elementary limits, kinematics/sign conventions, `F=ma`, inclines, energy/power, stoichiometry, limiting reactants, conservation, and related topics. Pin code/seed/template version; verify with symbolic/numeric/unit/equation solvers; plant exactly one controlled error before optional downstream steps. |
| **STEM train (stemdataset/STEM)** | K–12 math, science, technology, and engineering with course/grade and skill labels | **Primary high-school breadth source after audit** | Large structured question pool, explicit grade/course, skill, choices, and answer index | Pin the train split; retain only upper-secondary course/grade rows; require `pic_prob=false` and `pic_choice=false`; map fine-grained skills to the five subjects; freeze validation/test; reject non-self-contained or weakly sourced rows. Because worked explanations are absent, use Explain/Hint/Diagnose only after separately verified grounding is created. Recheck source-level provenance despite the Apache-2.0 dataset declaration. |
| **MATH train (Hendrycks)** | Competition secondary mathematics | **Use selectively with transformation** | Structured problems and worked solutions spanning algebra, geometry, number theory, counting/probability, and precalculus | MIT-licensed official repository; use train only; filter for target level and unambiguous solutions; hold out benchmark/test families; normalize LaTeX and answer formats. |
| **NuminaMath 1.5** | K–12 through olympiad mathematics with reference solutions and domain labels | **Use a small, aggressively filtered slice** | Algebra, geometry, calculus/precalculus, checkable word problems, and multi-step reference solutions | Pin the revision and source field; keep target-level, self-contained rows with checkable answers; reject `answer=notfound`, excessive olympiad/proof difficulty, ambiguous parsing, and all frozen benchmark-family matches. Cap at 15% of Mathematics rows and report contribution by upstream source, not only under the Numina aggregate name. |
| **EXAMS train** | Authentic multilingual high-school examinations across science and other subjects | **Conditional high-school source** | JSONL questions with subject/language labels, choices, answers, and official splits | Preserve CC BY-SA 4.0 attribution; use train only; select relevant STEM subjects; translate non-English rows only through a pinned translation-plus-backcheck process; verify the answer survives translation; create explanations separately; keep parallel-language versions in one family. |
| **MathDial train** | Math tutoring dialogue grounded in GSM8K | **Use with light transformation** | Student confusion, multi-turn context, teacher moves | Map dialogue turns to the validated move/intent taxonomy; retain history; use train only; preserve verified CC BY-SA 4.0 attribution. Hash and exclude MRBench, BEA, and MathTutorBench conversation/prompt families before import. Cap this GSM8K-derived lineage at 15% of Mathematics rows. |
| **Bridge training data** | Math remediation | **Use with light transformation** | Error, strategy, intention, response tuples | Map cognitive-task-analysis decisions to `student_state` and `pedagogy`; preserve MIT license notice; hash and exclude all MRBench/BEA/MathTutorBench families before import. |
| **EduAdapt grades 9–12** | Biology, chemistry, physics, ecology, geography, and geology | **Audit candidate; not approved by default** | Convenient subject-separated JSONL and Earth-science coverage | Its CC BY 4.0 declaration is useful, but original content provenance is insufficiently specific. Admit rows only if source lineage, duplication, and redistribution rights are established at row/source level; otherwise exclude. |
| **ChemistryQA** | Approximately 4,500 chemistry questions across about 200 topics | **Conditional / license review** | Chemistry-and-math reasoning questions with official TSV splits | The source uses a Computational Use of Data Agreement and Socratic-derived content. Use only if the pinned agreement permits TebebAI training and planned model distribution; otherwise exclude. Independently verify level and every retained answer. |
| **SceMQA** | College-entrance Mathematics, Physics, Chemistry, and Biology | **Evaluation only** | Multiple-choice/free-response tasks, knowledge points, and detailed explanations at the target difficulty | Freeze all rows, contexts, images, explanations, normalized hashes, and semantic clusters before training selection. Do not use its content to generate or tune training rows. |
| **MMLU high-school subsets** | High-school mathematics, physics, chemistry, biology, and related subjects | **Evaluation only by default** | Familiar subject-level accuracy slices | Freeze dev/validation/test and auxiliary-train families. Training on any MMLU material invalidates clean MMLU reporting; allow auxiliary-train use only in an explicitly contaminated experimental profile that does not report MMLU as external evidence. |
| **MRBench / BEA shared-task data** | Tutor-response evaluation derived from MathDial/Bridge | **Evaluation only** | Mistake identification, location, guidance, and actionability rubrics | Freeze conversation IDs, normalized text, templates, and semantic hashes before importing MathDial/Bridge. Borrow taxonomy and scoring logic, not benchmark answers. |
| **MathTutorBench** | Math tutoring evaluation with MathDial/Bridge/GSM8K lineage | **Evaluation only** | Separate expertise, student-understanding, and pedagogy checks | Freeze all available prompt/context/reference families before importing their upstream sources. Do not train on benchmark prompts or preferred responses. |
| **GPQA and similar hard science benchmarks** | Graduate-level science | **Exclude from training** | Optional out-of-domain diagnostic | Misaligned grade level and high contamination sensitivity; preserve only for evaluation if needed. |

### 5.1 Replaced legacy content sources

Direct GSM8K, SciQ, ARC, QASC, ScienceQA, TQA, WorldTree, and OpenBookQA rows are removed from the v8 content-training plan. They are retained only in historical manifests and, where useful, frozen evaluation. This avoids spending scarce cell capacity on predominantly grade-school material and removes several explanation, image, license, and ARC-family contamination complications. MathDial remains because it contributes authentic tutoring dialogue, but its GSM8K-derived content is treated as a limited pedagogy source rather than broad Mathematics coverage.

### 5.2 License classes and export profiles

Every source row is assigned one of `permissive`, `share_alike`, `noncommercial`, `custom_restricted`, `unknown`, or `excluded` after checking the pinned revision. `unknown` and `excluded` never enter generation. MathDial and EXAMS are currently documented as CC BY-SA 4.0, STEM and NuminaMath declare Apache 2.0, and MATH declares MIT; the build must still verify the exact pinned files and upstream provenance. ChemistryQA remains `custom_restricted` unless its Computational Use of Data Agreement is approved for the planned use.

The mixed `sft_mix_v8.jsonl` is an **internal ADTC training artifact**, not a public dataset release. The public repository releases reproducible import/transformation code, manifests, checksums, attribution and license reports, aggregate coverage, and only sample rows whose redistribution is confirmed. Internal rows are also segmented into `sft_v8_permissive.jsonl`, `sft_v8_sharealike.jsonl`, and `sft_v8_noncommercial.jsonl`. Do not publish all segments under one blanket license; downstream model distribution and commercial-use implications receive a separate legal/license review rather than being inferred from dataset availability.

### 5.3 General-capability replay

The 10,000-row matrix measures the tutoring curriculum and remains unchanged. During training, mix it at 90% effective sampling weight with 10% separately licensed general reasoning/instruction replay to reduce catastrophic forgetting. The initial replay source is a pinned, permissively licensed slice selected from the original base model's documented post-training source family when available; if no such source is documented and licensable, replay is blocked until `sources.yaml` names an approved substitute. “General replay” is not a sufficient provenance label.

Replay rows pass the same provenance, family, evaluation-decontamination, message-rendering, and Stage 7 source-artifact gates as tutoring rows, live in a separate manifest, and do not count toward any subject × behavior cell. In particular, replay may not reintroduce `####`, `<<...>>`, `<think>` content, dataset-specific role tokens, or unrequested `Final answer:` boilerplate. Validate the ratio against a frozen general-capability set and reduce or remove replay only when evidence shows no regression.

## 6. Source-specific transformation recipes

### 6.1 Deterministic STEM generators

Use code-generated records to close the secondary and early-university gaps that the existing grade-school-heavy sources do not cover. Initial template families must include quadratic equations, exponential substitution, elementary limits, projectile motion and sign conventions, inclined-plane forces, work/energy/power, stoichiometry, limiting reactants, and conservation equations.

Each template family must define:

- a versioned parameter space, domain assumptions, valid numerical ranges, difficulty rule, and unit system;
- a deterministic solver that produces the canonical answer and intermediate steps independently of the language model;
- property-based tests plus fixed golden cases for the solver, including boundary conditions and unit/equation validity;
- a `template_fingerprint`, code revision, seed, and maximum contribution under the 20% per-cell template cap;
- at least several independently written surface/context families per governing relationship, so number substitution alone cannot satisfy the diversity target;
- controlled student attempts created by changing exactly one first step, followed by a second pass that proves no earlier or accidental extra error exists;
- prompt variants that do not reproduce, paraphrase, or tune against known judge examples.

For generated content, `family_id` identifies the template version plus the surface/context scenario, not each random parameter draw. Parameter draws from one template and scenario are siblings in the same family; changing only names, numbers, or units does not create a new family. This prevents the 5,000-family diversity floor from being satisfied by random seeds alone.

The language model may verbalize a solver-verified record, but it may not alter the quantities, governing relationship, canonical answer, or planted error. A deterministic disagreement rejects the row even when every model agrees.

### 6.2 MATH train

Use the official MIT-licensed training split selectively for target-level algebra, geometry, counting/probability, number theory, and precalculus. Normalize LaTeX, extract the final answer, independently verify symbolic or numeric equivalence, and reject problems that require competition tricks far beyond the intended curriculum or whose solution cannot be checked reliably. Freeze MATH test hashes before selection and keep each transformed problem under one family ID.

### 6.3 STEM train

Filter before any answer expansion:

1. require the official training split and a pinned dataset revision;
2. retain only high-school course/grade labels and approved precursor skills needed by that curriculum;
3. require text-only questions and text-only choices (`pic_prob=false`, `pic_choice=false`);
4. map the source `grade` and `skill` to subject/topic through a versioned lookup, using Jev only when science skills remain ambiguous;
5. recompute quantitative answers when possible and independently verify every remaining answer index;
6. remove option letters and position cues before generation;
7. create Explain, Hint, or Diagnose only when new grounding and any proposed misconception pass their corresponding hard gates.

STEM is a candidate-question source, not a source of trusted worked explanations. A correct answer index alone cannot justify a causal explanation or diagnosis.

### 6.4 NuminaMath 1.5

Select by `source`, `problem_type`, `question_type`, answer status, and curriculum fit. Prefer algebra, geometry, elementary calculus/precalculus, and self-contained word problems with deterministic answers. Reject `notfound`, image-dependent, malformed, proof-only, and excessive olympiad rows. Recompute symbolic/numeric equivalence, retain the upstream source name in provenance, freeze common math benchmark families, and cap the final contribution at 15% of Mathematics rows.

### 6.5 EXAMS

Use only official training rows in relevant STEM subjects. Keep the original language, parallel-question IDs, and raw checksum. When an English rendering is needed, generate it with a pinned translator, back-translate with a separate pass, and verify that the scientific meaning, quantities, units, options, and correct answer are unchanged. Put all language variants of one problem in the same family. EXAMS supplies authentic questions and answers, not trusted tutoring explanations; generated reasoning requires normal verification.

### 6.6 MathDial and Bridge

These are the main sources for tutoring organization rather than broad science coverage.

- Preserve multi-turn history when the current response depends on it.
- Create one training record per teacher turn, with only prior turns exposed.
- Map MathDial's broad and fine-grained teacher moves into the v8 vocabulary.
- Map Bridge's error, remediation strategy, and intention into `student_state` and `pedagogy`.
- Do not rewrite a probing turn into a complete worked answer merely to make it self-contained.
- Keep original and transformed identifiers for attribution and split auditing.
- Before either import, freeze exact, normalized, template, and semantic hashes for MRBench, the BEA shared task, and MathTutorBench. Reject an upstream row when its conversation, problem, or prompt family matches any frozen evaluation family.
- Treat MathDial's GSM8K-derived content as one provenance group and cap it at 15% of Mathematics rows. It exists to teach interaction and remediation, not to dominate mathematical knowledge.

### 6.7 Audited conditional sources

EduAdapt grades 9–12 may enter only after its original source lineage and redistribution rights are established per source family. ChemistryQA may enter only after its pinned Computational Use of Data Agreement is approved for both training and the intended model distribution. Passing a factual quality check does not override a failed provenance or license check.

### 6.8 Source sanitization and benchmark-family firewall

Before transformation, normalize source-specific annotations into metadata rather than model-visible text. MathDial and every aggregated math importer must strip or reject GSM8K-style `####`, `<<...>>`, `Final answer:` boilerplate, hidden-calculation markup, and dataset-specific chat/template tokens. Preserve the raw record checksum separately for audit.

The benchmark firewall is built **before** importing related training sources. It contains stable IDs, normalized content hashes, template fingerprints, and semantic clusters for MRBench, BEA, MathTutorBench, SceMQA, all MMLU high-school families, STEM validation/test, EXAMS dev/test, MATH test, common NuminaMath-overlapping math benchmarks, and every internal evaluation family. A record matching any layer is rejected or adjudicated before generation; paraphrasing or translation after import cannot make it eligible.

## 7. Construction pipeline

The construction pipeline deliberately separates content generation from acceptance.

### Stage 1: ingest and provenance lock

- Download only approved sources and configurations.
- Pin repository commit, dataset revision, or archive checksum.
- Save the source license and required attribution.
- Assign a stable original ID before any transformation.
- Fail closed if license, revision, or split is missing.

### Stage 2: canonical problem and solution

- Normalize question, answer, equations, significant figures, and units.
- Produce explicit solution steps.
- Attach supporting facts or source passages.
- Solve quantitative items with deterministic code where possible.
- Reject ambiguous, broken, or unsupported items.

### Stage 3: Jev behavior eligibility and quota assignment

- Apply the hard field requirements for all four behaviors.
- For candidates not deterministically routed from source/template metadata, ask pinned `jev-1.13.0` in one structured request for four logically independent Noul probabilities (`p_yes`) for `solve`, `explain`, `hint`, and `diagnose`; do not make four separate calls.
- Send values in the calibrated per-behavior uncertainty bands to adjudication.
- Let the quota controller choose the final bucket from the eligible set using coverage and diversity targets.
- Store the exact Jev model ID, pinned TypeSafe SDK version, policy version, Noul probabilities, thresholded decisions, routing method, and selected bucket before answer generation.

Calibrate one `p_yes` threshold and uncertainty band per behavior before bulk routing on a stratified 300-row AI-adjudicated reference set covering all 20 cells and clear negative cases. The configured generator, verifier, and adjudicator independently label eligibility; deterministic hard gates resolve factual eligibility. Remaining disagreements must receive a final documented adjudication or be retained as unresolved calibration failures—they may not be silently removed from the denominator. Choose the lowest Jev threshold that achieves at least 95% precision for `eligible` on resolved reference cases, report per-bucket precision/recall plus the unresolved rate, and pin thresholds with the Jev model and policy versions. This is model-curated calibration, not human labeling.

### Stage 4: targeted student-state generation

Generate errors from the controlled taxonomy rather than asking for a generic “wrong solution.” The proposed error must be:

- plausible for the target grade level;
- localized to a first incorrect step;
- inconsistent with the canonical solution for a stated reason;
- repairable using the information available to the student;
- free of accidental second errors before the intended error point.

An independent examination pass verifies that the proposed attempt is actually wrong and that the assigned error label is correct.

For the 10–15% correct-attempt Diagnose slice, the examination pass instead proves that every displayed step is valid and complete enough for the requested level. These rows use `first_error_step: null`, `error_type: "none"`, `error_class: null`, and require the tutor to affirm the valid reasoning without inventing a mistake.

### Stage 5: tutoring decision plan

Following Bridge, create a hidden structured plan:

1. What is the first error or missing idea?
2. Which remediation strategy best addresses it?
3. What should the student's next cognitive action be?
4. Should the answer be revealed, partially revealed, or withheld?

This plan populates metadata and conditions generation. It is not emitted as artificial chain-of-thought in the final answer.

### Stage 6: bucket-conditioned answer expansion

The quota controller assigns exactly one selected behavior bucket to each generation job. The API-preflighted generator, verifier, and adjudicator role IDs from `build_manifest.json` are used; no role may fall back silently to a marketing name or Cursor-internal slug. A cross-provider model may substitute only when its exact callable model ID, prompt version, and terms are pinned. Expand the canonical source row into the output contract for that bucket:

- `solve`: produce the answer and a correct, complete reasoning path;
- `explain`: teach the requested concept, mechanism, or distinction;
- `hint`: expose only the smallest useful next step and withhold the answer;
- `diagnose`: locate the first error or misconception and guide the student's repair.

The generation model receives the canonical grounding, assigned bucket, student state when required, explicit behavior instruction, and tutoring decision plan. It does not select or change the subject or bucket. Generate each behavior directly from canonical grounding—not by rewriting a previously generated variant. Keep the requested number of questions and parts explicit. Multi-part prompts must store each atomic requirement in `constraints.items` so omissions can be tested.

Where a pinned cross-provider verifier is available, prefer it to reduce correlated errors. Otherwise use the exact same-provider role IDs and label the result `same-provider-reduced-independence`; do not describe it as provider-independent evidence. Deterministic checks remain authoritative in either configuration.

### Stage 7: deterministic validation

Every row must pass:

- JSON Schema and controlled-vocabulary validation;
- required-role and message-order validation;
- exact duplicate and normalized duplicate checks;
- number-aware forbidden-answer leakage checks for `hint` and answer-withholding `diagnose`, including equivalent fractions, decimals, scientific notation, unit conversions, and accepted textual variants while avoiding substring false positives;
- requested-part coverage and question-count checks;
- numeric recomputation where applicable;
- unit dimensionality and equation-balance checks where applicable;
- option-position and answer-letter artifact checks;
- unsupported citation and invalid markup checks;
- source-artifact checks rejecting `<think>`, `</think>`, `####`, `<<...>>`, dataset-specific role tokens, and hidden-calculation markup in all source and message content fields;
- a `Final answer:` check: reject it in Hint/Diagnose and allow it in Solve only when the renderer explicitly requests that label.

The `<think>` rule applies to dataset content, not to empty control tokens that the Qwen3 chat template may insert during tokenization. Rendered-content validation must distinguish template-owned control tokens from source-authored or model-authored hidden reasoning.

### Stage 8: independent semantic verification

A verifier evaluates the response without receiving the generator's hidden rationale. It scores:

- factual correctness;
- correctness of the located error;
- consistency with the canonical solution;
- usefulness and actionability of the next step;
- compliance with the reveal/withhold policy;
- appropriateness for the target grade and requested behavior.

Rows are rejected, not merely downweighted, for factual errors, a wrong first-error location, answer leakage, or missing required parts.

After the corpus is frozen, draw a reproducible stratified 10% sample from the 9,000 standard-assurance rows, with representation from every cell, source group, difficulty, and checker type. Re-run verification using the adjudicator prompt/model, blinded to the first verdict. At least 95% must pass; any systemic failure class triggers a full affected-slice audit and rebuild. Report this as **same-provider re-verification** when the default roles use one provider, not human validation or provider independence.

### Stage 9: high-assurance adjudication

Every row receives the independent semantic verification in Stage 8. The 1,000-row high-assurance tier—50 examples from every subject × behavior cell—receives an additional review from a frontier model different from the generator where practical. The adjudicator sees provenance, canonical solution, student attempt, planned strategy, response, and the first verifier's structured findings. Disagreements trigger regeneration or rejection; the adjudicator must not silently average incompatible answers.

This tier is still model-curated, not human gold. All novel quantitative chemistry and physics templates must also pass deterministic domain checks before bulk generation. If two models agree but a calculator, symbolic solver, unit checker, or chemical-equation checker disagrees, the deterministic result blocks acceptance.

### Stage 9.5: call and token budget

Budget the build before bulk generation. The initial ceiling is a 30,000-family candidate pool, with one batched Jev request per non-deterministically routed candidate for behavior Noul questions and, when used, six quality Score questions. Budget every other model-mediated stage explicitly: calibration labels and final disagreement adjudication; source grounding; student-state generation and examination; tutoring decision plans; answer expansion; semantic verification; high-assurance adjudication; EXAMS translation and back-translation; African-context-bank generation and verification; and blinded standard-tier re-verification.

Treat 12,000 generation attempts for a 10,000-row target only as a provisional 20% replacement reserve. The 200-row QA pilot must estimate rejection, regeneration, translation, and token rates; the final ceiling is then recomputed with a stated confidence margin. Cap regeneration at two attempts per logical row, then reject the family/bucket pair.

`build_manifest.json` records per-stage calls, input/output tokens, retries, cached calls, unit price, and total cost using model-specific token usage and pinned prices. It also records deterministically skipped Jev calls so routing savings are auditable. The pipeline stops before exceeding any approved per-stage or total call, token, or monetary budget.

### Stage 10: family-level split and decontamination

Split only after assigning `family_id`. All paraphrases, behavior variants, translations, and student-error variants derived from the same original item stay in one split.

Use an 80/10/10 train/validation/test split by family for TebebAI-created material. Respect source-provided splits first: an official test item never becomes training material merely to satisfy the 80/10/10 target.

#### Stage 10a: stable identity hierarchy

A single hash of the final text is insufficient because paraphrasing, changed numbers, or different prompt rendering can hide repetition. Every row therefore has five related identities:

1. `source_key`: the pinned dataset, revision, split, and original record ID. If the source has no stable ID, use its pinned row index plus raw-record hash.
2. `family_id`: a SHA-256 identifier shared by every derivative of the same canonical problem, including different behaviors, prompts, student attempts, and translations. For deterministic generators it hashes the template version plus surface/context scenario; parameter draws that differ only in names, numbers, units, or random seed remain in that family.
3. `constraint_signature`: a SHA-256 hash of the sorted constraint IDs and their parameters.
4. `row_id`: a SHA-256 hash of `family_id + selected_behavior + constraint_signature + renderer_version`. The same logical training row therefore keeps the same ID even if it is rebuilt.
5. `generation_attempt_id`: a separate attempt identifier stored in the generation log, not used as the training-row identity. Regenerating a rejected answer does not create a new logical row.

All hashes include an explicit normalization/version prefix. Changing the normalization algorithm creates a new version rather than silently changing existing identities.

#### Stage 10b: repetition detection layers

Deduplication runs across the union of all source datasets, pilot rows, newly generated rows, and frozen evaluation:

1. **Source identity:** prevent importing the same pinned source record twice.
2. **Exact normalized hash:** lowercase and normalize Unicode, whitespace, punctuation, equations, and choice ordering before hashing canonical content, user prompt, and assistant response separately.
3. **Variant uniqueness:** allow at most one accepted row for the same `family_id + behavior + constraint_signature + renderer_version`.
4. **Template fingerprint:** replace names and numerical constants with typed placeholders, then hash the remaining problem structure. For imported data, this catches examples such as the same algebra template with only numbers changed. For declared deterministic-generator rows, a shared template fingerprint is expected and is governed by family grouping, the 20% template cap, parameter-space coverage, and surface/context diversity; it does not by itself reject every sibling draw.
5. **Lexical near-duplicate:** use token shingles plus MinHash to find lightly rewritten questions and answers.
6. **Semantic cluster:** use embeddings to identify paraphrases and the same problem imported from different datasets.
7. **Response boilerplate:** cluster normalized assistant responses and repeated openings/closings so a generation template cannot dominate a cell.
8. **Evaluation isolation:** compare every family, content hash, template fingerprint, and semantic cluster against frozen evaluation before acceptance.

Exact duplicates are rejected automatically. High template or semantic similarity triggers high-assurance adjudication because two legitimate science questions can use the same law without being duplicate training examples. Declared generator siblings follow the explicit exception above but remain subject to exact, response-boilerplate, family-count, and evaluation-isolation checks. When two datasets contain the same problem, keep the row with the stronger grounding, clearer license, and higher quality score; store the rejected row's `duplicate_of` pointer to the retained representative.

The existing limits still apply: normally no more than two behavior variants per family, no more than 15% of a cell from one topic, and no more than 20% from one rendering/transformation template.

#### Stage 10c: QA pilot, mini-train, and full transition

The balanced 200-row pilot (10 per cell) is a **pipeline-QA artifact only**. Use it to test imports, routing, generation, validators, costs, and reports; do not fine-tune a model on it or claim cell-level performance. Register every accepted QA-pilot row so later builds cannot accidentally recreate it under a new identity.

If an early training signal is needed, build a separate balanced 1,000-row mini-train (50 per cell). It is large enough only for directional pipeline comparison, not a final accuracy claim. Its rows may become a registered subset of the full corpus, subject to the exposure rules below.

The recommended full-training procedure is:

1. freeze `pilot_manifest.json` for QA-only rows and `mini_train_manifest.json` for any rows actually used in a mini-train, including all `row_id`, `family_id`, hashes, model versions, and output checksums;
2. build new candidates against the same identity registry;
3. exclude QA-only pilot rows from training; define the full corpus as `accepted_mini_train_rows UNION accepted_new_unique_rows` when a mini-train exists, otherwise use only accepted full-build rows;
4. run all deduplication layers again across the complete union;
5. produce a final manifest proving every `row_id` occurs exactly once;
6. restart full training from the original base-model checkpoint, not the mini-trained checkpoint, so every training row has the intended exposure count.

If compute constraints require continuing from the mini-trained checkpoint, exclude every mini-train `row_id` from the full-stage dataloader and record its earlier exposure count. Never continue from that checkpoint while also feeding those rows again.

Pilot **evaluation** rows are different from pilot training rows: evaluation rows and all their families, templates, and close semantic matches remain permanently excluded from training.

#### Stage 10d: identity registry

Maintain `data/manifests/v8/identity_registry.sqlite` as the build-time source of truth. A `source_records` table has a unique index on `source_key`; a `training_rows` table has unique indexes on `row_id` and the logical variant key; a `build_membership` table records whether the row belongs to `qa_pilot`, `mini_train`, `full`, or `evaluation`. The registry also stores hashes, family/cluster IDs, acceptance status, duplicate pointers, rejection reasons, and exposure counts. JSONL and Parquet files are exports; they do not decide uniqueness independently.

## 8. Evaluation organization

### 8.1 Internal frozen evaluation

Create a new frozen evaluation pack before generating training variants:

| Slice | Minimum examples | What it measures |
|---|---:|---|
| Subject × behavior cells | 50 per cell | Balanced 5 × 4 performance |
| Multi-part instruction following | 100 | Completion of every requested part |
| Quantitative units/equations | 100 | Numerical and scientific validity |
| Misconception diagnosis | 200 | Error identification and location |
| Hint answer-withholding | 100 | Useful guidance without leakage |
| Judge replay | 13 known unique prompts | Regression on observed ADTC failures, including sign convention, multi-part completeness, exact diagnosis, scaffolding, and answer leakage |

Use family-level isolation and record source, topic, difficulty, and rubric. The existing custom tutor smoke tests should be corrected and expanded; they cannot be considered a reliable accuracy measure while they rely mainly on length or loose keyword checks.

The judge-replay slice is transcribed from the two official feedback PDFs, deduplicated to 13 unique prompts, hashed, and frozen before source selection. Its prompts, paraphrases, named quantities, templates, and close semantic matches enter the benchmark firewall and may not be used to write generator templates, tune prompts, select checkpoints, or create training rows. Report it as a known-judge regression slice rather than evidence of unseen-finals generalization.

### 8.2 External evaluation

- SceMQA for target-level Mathematics, Physics, Chemistry, and Biology, reporting text-only and multimodal slices separately;
- frozen MMLU high-school mathematics, physics, chemistry, biology, and related subsets, provided no MMLU material entered the reported training profile;
- STEM validation/test and EXAMS dev/test only as held-out source-family checks, clearly distinguished from truly external evaluation;
- MRBench/BEA tutor dimensions where setup and license permit;
- MathTutorBench for math expertise, student understanding, and pedagogical response generation;
- a separate Earth-science set from a source family not used in training;
- blinded pairwise review of v7 versus v8 by a judge model not used to generate the compared examples when available; otherwise label the pinned judge same-provider.

Report macro averages across subject × behavior cells in addition to micro averages. This prevents the largest category from dominating the reported score.

### 8.3 Metrics

**Correctness**

- exact match / multiple-choice accuracy where appropriate;
- numeric tolerance with unit correctness;
- equation or symbolic equivalence;
- factual entailment from approved grounding;
- first-error step accuracy.

**Tutoring quality**

- mistake identification;
- mistake location;
- guidance quality;
- actionability;
- answer-withholding compliance;
- behavior/instruction adherence;
- multi-part completeness.

**Ablations**

1. v7 baseline versus balanced v8.
2. full v8 versus v8 trained without Diagnose rows (same total training tokens, with the removed quota replaced proportionally from the other behaviors).

Defer source-by-source, prompt-style, and high-assurance weighting ablations until these two answer the primary questions without multiplying training cost.

## 9. Acceptance gates

The dataset is releasable for training only when all hard gates pass.

| Gate | Threshold |
|---|---:|
| Rows with schema-valid canonical records | 100% |
| Rows with pinned provenance and approved license | 100% |
| Publicly exported rows whose license profile permits that export | 100% |
| Duplicate `row_id` or logical variant keys in the final manifest | 0 |
| Accepted families with more than two behavior variants without a documented exception | 0 |
| Mini-train rows appearing more than once in the full corpus | 0 |
| Pilot/evaluation families or semantic duplicates entering training | 0 |
| Non-deterministically routed rows with four Jev Noul probabilities, thresholded decisions, and pinned model/SDK/policy versions | 100% |
| Deterministically routed rows with an auditable source/template rule and no fabricated Jev result | 100% |
| Rows whose selected bucket failed its hard gate or Jev `p_yes` threshold | 0 |
| Unadjudicated Jev decisions inside a calibrated uncertainty band used for generation | 0 |
| Exact overlap with frozen evaluation | 0 |
| Known family crossing train/eval boundary | 0 |
| Quantitative rows with recomputed canonical result | 100% |
| Chemistry equations checked for balance when applicable | 100% |
| Hint/diagnosis rows passing forbidden-answer check | 100% |
| Rendered rows containing banned source artifacts (`####`, `<<...>>`, `<think>` tokens) | 0 |
| Rows passing every declared atomic constraint | 100% |
| Rows with at least three independently checked constraints | at least 25% |
| Deterministic-generator rows passing solver, property, and template tests | 100% |
| Frozen benchmark-family matches entering training | 0 |
| Rows accepted by the separately prompted semantic verifier | 100% |
| High-assurance rows accepted by the pinned adjudicator | 100% |
| Subject × behavior cell count | exactly the declared balanced target `N`; `N=500` preferred and `N>=400` required for a full build |
| Exact duplicate assistant responses after normalization | 0 |
| Blinded, stratified 10% standard-tier re-verification acceptance | at least 95% |

A row that fails a hard correctness gate is excluded. It is not retained because the dataset is below its target size.

## 10. Required artifacts and directory layout

```text
data/
  manifests/v8/
    sources.yaml
    licenses/
    frozen_eval_hashes.json
    identity_registry.sqlite
    pilot_manifest.json
    mini_train_manifest.json
    replay_manifest.json
    build_manifest.json
  canonical/v8/
    train.jsonl
    validation.jsonl
    test.jsonl
  train/
    sft_mix_v8.jsonl                 # internal combined training artifact
    sft_v8_permissive.jsonl
    sft_v8_sharealike.jsonl
    sft_v8_noncommercial.jsonl
    general_replay_v8.jsonl
  eval/v8/
    balanced_tutoring_v8.jsonl
    rubrics_v8.json
docs/artifacts/v8/
  coverage_report.json
  quality_report.json
  decontamination_report.json
  license_report.md
  model_curation_report.md
```

All paths in this document are relative to the `adtc/` project directory inside the git repository, so `data/` means the existing project's data directory—not another `adtc/data/` nested inside it.

`sources.yaml` must contain source name, canonical URL, pinned revision/checksum, configuration, imported split, exact license, license class, attribution text, allowed-use/export decision, and review model/date. `pilot_manifest.json` freezes the QA-pilot membership and hashes; `mini_train_manifest.json` records any actually trained early subset and its exposures. `identity_registry.sqlite` enforces cross-build identity and uniqueness. `build_manifest.json` records all scripts, versions, random seeds, input hashes, output hashes, row counts, rejection reasons, model-independence label, calls/tokens/cost, exposure counts, and final sampling weights.

## 11. Recommended implementation sequence

0. Run a no-generation supply census over pinned source metadata and deterministic hard gates. Report unique eligible families by source, subject, behavior, grade, grounding availability, and license/export class; estimate the post-dedup lower bound for every cell. Do not spend on bulk Jev or generative-model calls until every cell has a credible path to at least 400 rows. For Biology and Earth science, record but do not preselect the remediation options: approve a licensed grounding-only reference corpus, add solver-backed generators for suitable topics, or revise the curriculum matrix through a separate go/no-go decision.
1. Freeze the v8 internal evaluation pack—including the 13-prompt judge-replay slice—and source hashes.
2. Implement the canonical schema, validators, normalization functions, and SQLite identity registry.
3. Implement and test the deterministic STEM generators; import filtered STEM train, MATH train, the audited NuminaMath slice, MathDial, and Bridge; import EXAMS train only after the translation and license path is pinned.
4. Freeze external benchmark-family hashes before importing their related upstream sources.
5. Preflight every configured model role against the provider API, then pin callable IDs, TypeSafe `jev-1.13.0`, `typesafe-sdk==0.7.2` (or an explicitly reviewed successor), prompts, thresholds, and prices in the manifest.
6. Build a balanced 200-row QA pilot through the complete generation, verification, cost, and deduplication pipeline; do not train on it.
7. Freeze `pilot_manifest.json`, inspect failures and costs, then correct the pipeline with explicitly versioned rules.
8. Optionally build and train a separate balanced 1,000-row mini-train for a directional signal; register every exposure.
9. Generate only new candidates absent from the identity registry until the 20 cells reach their valid balanced target.
10. Create and adjudicate the 1,000-row high-assurance tier, then run the blinded stratified re-verification sample.
11. Union eligible mini-train and new rows, rerun cross-source/evaluation deduplication, and render a manifest with each `row_id` exactly once.
12. Restart from the original base-model checkpoint and train the full internal `sft_mix_v8.jsonl` with the declared 90/10 tutoring-to-replay sampling ratio.

## 12. Go/no-go decisions

- **Go:** equalize final subject × behavior coverage.
- **Go:** replace the previous grade-school-heavy content mix with deterministic generators, filtered high-school STEM train rows, selected MATH train rows, a small audited NuminaMath slice, and conditional EXAMS train rows.
- **Go:** use MathDial and Bridge structures to encode teacher decisions.
- **Go:** prefer row-structured JSON/JSONL/Parquet datasets with stable IDs, splits, answers, and metadata.
- **Go:** use source metadata and deterministic mappings first; invoke Jev when subject categorization is missing or ambiguous.
- **Go:** use one TypeSafe Choice question for unresolved subject routing, four independent Noul questions in one request for unresolved behavior eligibility, and calibrated `p_yes` thresholds; use deterministic source/template routing when it already proves eligibility.
- **Go:** use source-grounded synthetic transformations with independent verification.
- **Go:** assign the behavior bucket before generation, then use the exact API-preflighted model stack for bucket-conditioned answer expansion, separate verification, and 10% high-assurance adjudication; label same-provider verification honestly.
- **Go:** use solver-backed deterministic generators and selected MATH-train rows to cover secondary/early-university quantitative gaps.
- **Go:** use a 90/10 effective sampling mix of tutoring rows and separately governed general-capability replay, subject to held-out regression results.
- **Go:** use the SQLite identity registry plus exact, template, lexical, and semantic fingerprints across every build stage.
- **Go:** use the 200-row pilot for pipeline QA only; register any 1,000-row mini-train subset and restart full training from the original base checkpoint.
- **Go:** keep mixed-license training rows internal and publish only clearly permitted artifacts/segments with attribution.
- **No-go:** treating distractors as correct misconception labels without validation.
- **No-go:** training on MRBench, MathTutorBench, BEA evaluation, SceMQA, MMLU high-school evaluation families, STEM validation/test, EXAMS dev/test, or MATH test.
- **No-go:** importing direct GSM8K, SciQ, ARC, QASC, ScienceQA, TQA, WorldTree, or OpenBookQA rows into the v8 content mix.
- **No-go:** importing a dataset whose exact license or pinned revision is unknown.
- **No-go:** starting bulk model calls before the supply census shows a credible path to the declared balanced minimum in every cell.
- **No-go:** pinning a generator, verifier, or adjudicator ID that has not passed an API preflight with the project credentials.
- **No-go:** ingesting unstructured books or chapter extracts.
- **No-go:** measuring tutoring quality using response length alone.
- **No-go:** fine-tuning on the 200-row QA pilot or treating its cell results as meaningful.
- **No-go:** continuing from a mini-trained checkpoint while feeding the mini-train rows again.

## References

- Macina et al. [MathDial: A Dialogue Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning Problems](https://aclanthology.org/2023.findings-emnlp.372/), 2023. [Official dataset repository and CC BY-SA 4.0 notice](https://github.com/eth-nlped/mathdial).
- Wang et al. [Bridge: A Cognitive Task Analysis Framework for Modeling the Behavior of Expert Tutors](https://aclanthology.org/2024.naacl-long.120/), 2024. [Official repository](https://github.com/rosewang2008/bridge).
- Google LearnLM Team. [LearnLM: Improving Gemini for Learning](https://arxiv.org/abs/2412.16429), 2024.
- Sonkar et al. [FEAT: An Evaluation Framework for Effective AI Tutoring](https://aclanthology.org/2025.acl-short.45/), 2025.
- Maurya et al. [Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors](https://aclanthology.org/2025.naacl-long.57/), 2025. [Official MRBench repository](https://github.com/kaushal0494/UnifyingAITutorEvaluation).
- Maurya et al. [BEA 2025 Shared Task on Pedagogical Ability Assessment of AI-powered Tutors](https://aclanthology.org/2025.bea-1.77/), 2025.
- Zhao et al. [MathTutorBench: A Benchmark for Measuring Open-ended Pedagogical Capabilities of LLM Tutors](https://aclanthology.org/2025.emnlp-main.11/), 2025. [Official repository](https://github.com/eth-lre/mathtutorbench).
- Hendrycks et al. [Measuring Mathematical Problem Solving With the MATH Dataset](https://arxiv.org/abs/2103.03874), 2021. [Official MIT-licensed repository](https://github.com/hendrycks/math).
- Shen et al. [Measuring Vision-Language STEM Skills of Neural Models](https://openreview.net/forum?id=spvaV5LELF), ICLR 2024. [Official STEM dataset](https://huggingface.co/datasets/stemdataset/STEM).
- Project Numina. [NuminaMath 1.5 dataset card and source breakdown](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5).
- Hardalov et al. [EXAMS: A Multi-subject High School Examinations Dataset for Cross-lingual and Multilingual Question Answering](https://aclanthology.org/2020.emnlp-main.438/), 2020. [Official repository](https://github.com/mhardalov/exams-qa).
- Liang et al. [SceMQA: A Scientific College Entrance Level Multimodal Question Answering Benchmark](https://arxiv.org/abs/2402.05138), 2024. [Official repository](https://github.com/SceMQA/SceMQA).
- Hendrycks et al. [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300), 2020. [MMLU dataset](https://huggingface.co/datasets/cais/mmlu).
- Conditional-source records: [EduAdapt](https://huggingface.co/datasets/notefill/eduadapt) and [Microsoft ChemistryQA](https://github.com/microsoft/chemistry-qa).
- TypeSafe documentation: [Jev models](https://docs.typesafe.ai/models), [Noul](https://docs.typesafe.ai/primitives/noul), [Choice](https://docs.typesafe.ai/primitives/choice), [Score](https://docs.typesafe.ai/primitives/score), and [confidence](https://docs.typesafe.ai/confidence).
