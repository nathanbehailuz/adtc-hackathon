"""Pydantic schema for canonical TebebAI v8 records."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .vocab import ALLOWED_MESSAGE_ORDERS, BEHAVIORS, MOVE_INTENTS, SCHEMA_VERSION


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Source(StrictModel):
    dataset: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    config: str = Field(min_length=1)
    source_split: str = Field(min_length=1)
    original_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    license: str = Field(min_length=1)
    license_class: Literal[
        "permissive",
        "share_alike",
        "noncommercial",
        "custom_restricted",
    ]
    export_profile: Literal["permissive", "sharealike", "noncommercial", "internal_only"]
    transformation: str = Field(min_length=1)


class Curriculum(StrictModel):
    subject: Literal["math", "physics", "chemistry", "biology", "earth_science"]
    secondary_subjects: list[
        Literal["math", "physics", "chemistry", "biology", "earth_science"]
    ] = Field(default_factory=list)
    topic: str = Field(min_length=1)
    subtopic: str | None = None
    grade_band: Literal["secondary_9_12"]
    difficulty: Literal["easy", "medium", "hard"]
    learning_objective: str = Field(min_length=1)

    @model_validator(mode="after")
    def primary_is_not_secondary(self) -> "Curriculum":
        if self.subject in self.secondary_subjects:
            raise ValueError("primary subject cannot also be a secondary subject")
        if len(set(self.secondary_subjects)) != len(self.secondary_subjects):
            raise ValueError("secondary_subjects must be unique")
        return self


class Pedagogy(StrictModel):
    behavior: Literal["solve", "explain", "hint", "diagnose"]
    teacher_move: Literal["focus", "probe", "tell", "generic"]
    intent: Literal[
        "guide_focus",
        "seek_strategy",
        "seek_explanation",
        "seek_self_correction",
        "recall_relevant_fact",
        "perturb_problem",
        "reveal_strategy",
        "reveal_step",
        "reveal_answer",
        "encourage",
        "check_understanding",
        "greet_or_close",
        "general_inquiry",
    ]
    principles: list[
        Literal[
            "active_learning",
            "manage_cognitive_load",
            "adapt_to_learner",
            "stimulate_curiosity",
            "deepen_metacognition",
        ]
    ] = Field(default_factory=list)
    answer_policy: Literal["reveal_final", "partial_reveal", "withhold_final"]
    question_count: int = Field(ge=0)
    interaction_mode: Literal["single_turn", "multi_turn"]

    @model_validator(mode="after")
    def valid_move_intent(self) -> "Pedagogy":
        if self.intent not in MOVE_INTENTS[self.teacher_move]:
            allowed = ", ".join(MOVE_INTENTS[self.teacher_move])
            raise ValueError(
                f"intent {self.intent!r} is invalid for move {self.teacher_move!r}; "
                f"allowed: {allowed}"
            )
        return self


class Grounding(StrictModel):
    problem: str = Field(min_length=1)
    canonical_answer: str | None
    solution_steps: list[str] = Field(default_factory=list)
    supporting_facts: list[str] = Field(default_factory=list)
    reference_explanation: str | None = None
    equations: list[str] = Field(default_factory=list)
    units: list[str] = Field(default_factory=list)
    accepted_variants: list[str] = Field(default_factory=list)


class StudentState(StrictModel):
    attempt: str = Field(min_length=1)
    first_error_step: int | None = Field(default=None, ge=1)
    error_type: Literal[
        "none",
        "concept_confusion",
        "wrong_assumption",
        "unsupported_claim",
        "instruction_miss",
        "premature_conclusion",
        "arithmetic_error",
        "algebra_error",
        "ratio_error",
        "unit_error",
        "sign_error",
        "sign_convention_error",
        "coordinate_choice_error",
        "formula_selection_error",
        "transcription_error",
        "causal_reversal",
        "category_error",
        "scale_error",
        "conservation_error",
        "equilibrium_error",
        "representation_error",
    ]
    error_class: Literal["conceptual", "procedural"] | None
    error_class_rationale: str = Field(min_length=1)
    misconception: str = Field(min_length=1)

    @model_validator(mode="after")
    def correct_attempt_semantics(self) -> "StudentState":
        if self.error_type == "none":
            if self.first_error_step is not None or self.error_class is not None:
                raise ValueError(
                    "correct attempts require first_error_step=null and error_class=null"
                )
        elif self.first_error_step is None or self.error_class is None:
            raise ValueError(
                "erroneous attempts require first_error_step and error_class"
            )
        return self


class ConstraintItem(StrictModel):
    id: str = Field(min_length=1)
    check: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ConstraintResult(StrictModel):
    passed: bool
    observed: Any | None = None
    matched_variants: list[str] = Field(default_factory=list)
    notes: str | None = None


class Constraints(StrictModel):
    items: list[ConstraintItem] = Field(min_length=1)
    forbidden_answers: list[str] = Field(default_factory=list)
    constraint_results: dict[str, ConstraintResult]
    all_constraints_pass: bool

    @model_validator(mode="after")
    def results_match_items(self) -> "Constraints":
        item_ids = [item.id for item in self.items]
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("constraint item IDs must be unique")
        if set(item_ids) != set(self.constraint_results):
            raise ValueError("constraint_results keys must exactly match constraint item IDs")
        actual = all(result.passed for result in self.constraint_results.values())
        if self.all_constraints_pass != actual:
            raise ValueError("all_constraints_pass disagrees with constraint results")
        return self


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class BehaviorEligibility(StrictModel):
    eligible: bool
    p_yes: float | None = Field(default=None, ge=0.0, le=1.0)
    method: Literal["jev_noul", "deterministic_source"]

    @model_validator(mode="after")
    def probability_matches_method(self) -> "BehaviorEligibility":
        if self.method == "jev_noul" and self.p_yes is None:
            raise ValueError("jev_noul decisions require p_yes")
        if self.method == "deterministic_source" and self.p_yes is not None:
            raise ValueError("deterministic decisions must not fabricate p_yes")
        return self


class QualityScores(StrictModel):
    grounding: int = Field(ge=0, le=2)
    answer_verifiability: int = Field(ge=0, le=2)
    curriculum_fit: int = Field(ge=0, le=2)
    behavior_affordance: int = Field(ge=0, le=2)
    clarity: int = Field(ge=0, le=2)
    diversity_value: int = Field(ge=0, le=2)

    @property
    def total(self) -> int:
        return sum(
            (
                self.grounding,
                self.answer_verifiability,
                self.curriculum_fit,
                self.behavior_affordance,
                self.clarity,
                self.diversity_value,
            )
        )


class Selection(StrictModel):
    decision: Literal["accept", "reject", "pending"]
    subject_method: Literal[
        "source_metadata",
        "deterministic_mapping",
        "ontology_rule",
        "jev_choice",
        "adjudicator",
    ]
    subject_distribution: dict[str, float] | None = None
    subject_choice_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    behavior_eligibility: dict[str, BehaviorEligibility]
    behavior_classifier: str
    typesafe_sdk: str | None = None
    selected_behavior: Literal["solve", "explain", "hint", "diagnose"]
    scores: QualityScores
    total: int = Field(ge=0, le=12)
    rejection_reasons: list[str] = Field(default_factory=list)

    @field_validator("behavior_eligibility")
    @classmethod
    def all_behaviors_present(
        cls, value: dict[str, BehaviorEligibility]
    ) -> dict[str, BehaviorEligibility]:
        if set(value) != set(BEHAVIORS):
            raise ValueError(f"behavior_eligibility must contain exactly {BEHAVIORS}")
        return value

    @model_validator(mode="after")
    def selection_consistency(self) -> "Selection":
        if self.total != self.scores.total:
            raise ValueError("selection total must equal the six quality scores")
        if not self.behavior_eligibility[self.selected_behavior].eligible:
            raise ValueError("selected_behavior must be eligible")
        if self.subject_method == "jev_choice":
            expected = {
                "math",
                "physics",
                "chemistry",
                "biology",
                "earth_science",
                "no_match",
            }
            if self.subject_distribution is None or set(self.subject_distribution) != expected:
                raise ValueError("jev_choice requires all six subject probabilities")
            total = sum(self.subject_distribution.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError("subject probabilities must sum to 1")
            if self.subject_choice_confidence is None:
                raise ValueError("jev_choice requires subject_choice_confidence")
        return self


class Quality(StrictModel):
    generator: str
    verifiers: list[str] = Field(min_length=1)
    factual_score: float = Field(ge=0.0, le=1.0)
    pedagogy_score: float = Field(ge=0.0, le=1.0)
    constraint_pass: bool
    curator_review: Literal["accepted", "rejected", "pending"]
    curator_model: str | None = None
    model_independence: Literal[
        "same-provider-reduced-independence",
        "cross-provider",
        "not-applicable",
    ]
    review_passes: int = Field(ge=0)
    assurance_tier: Literal["standard", "high"]
    selection: Selection


class Identity(StrictModel):
    source_key: str = Field(min_length=1)
    family_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    row_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    record_version: int = Field(ge=1)
    parent_row_ids: list[str] = Field(default_factory=list)


class Dedup(StrictModel):
    normalization_version: str = Field(min_length=1)
    canonical_content_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prompt_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    assistant_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    constraint_signature: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    template_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    semantic_cluster_id: str | None = None
    nearest_eval_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    duplicate_of: str | None = None
    decision: Literal["keep", "reject", "adjudicate"]


class CanonicalRecord(StrictModel):
    schema_version: Literal["tebeb-v8.1"]
    id: str = Field(min_length=1)
    split: Literal["train", "validation", "test", "external_eval"]
    source: Source
    curriculum: Curriculum
    pedagogy: Pedagogy
    grounding: Grounding
    student_state: StudentState | None = None
    constraints: Constraints
    messages: list[Message] = Field(min_length=2, max_length=3)
    quality: Quality
    identity: Identity
    dedup: Dedup

    @model_validator(mode="after")
    def cross_field_contract(self) -> "CanonicalRecord":
        roles = tuple(message.role for message in self.messages)
        if roles not in ALLOWED_MESSAGE_ORDERS:
            raise ValueError(f"invalid message order: {roles}")
        if self.pedagogy.behavior == "diagnose" and self.student_state is None:
            raise ValueError("diagnose rows require student_state")
        if self.pedagogy.behavior not in {"diagnose", "hint"} and self.student_state:
            raise ValueError("student_state is only allowed for diagnose or hint")
        if self.grounding.canonical_answer is None:
            if self.pedagogy.behavior != "explain":
                raise ValueError("only explain rows may omit canonical_answer")
            if not self.grounding.supporting_facts or not self.grounding.reference_explanation:
                raise ValueError(
                    "explain rows without canonical_answer require supporting facts "
                    "and a reference explanation"
                )
        if self.pedagogy.behavior != self.quality.selection.selected_behavior:
            raise ValueError("pedagogy behavior must match selected_behavior")
        if self.quality.constraint_pass != self.constraints.all_constraints_pass:
            raise ValueError("quality.constraint_pass disagrees with constraints")
        return self


def canonical_json_schema() -> dict[str, Any]:
    """Return the machine-readable JSON Schema for tebeb-v8.1."""

    schema = CanonicalRecord.model_json_schema()
    schema["$id"] = "https://tebeb.ai/schemas/tebeb-v8.1.schema.json"
    schema["title"] = f"TebebAI canonical record {SCHEMA_VERSION}"
    return schema
