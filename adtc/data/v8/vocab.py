"""Controlled vocabularies and cross-field rules for tebeb-v8.1."""

from __future__ import annotations

SCHEMA_VERSION = "tebeb-v8.1"

SUBJECTS = ("math", "physics", "chemistry", "biology", "earth_science")
BEHAVIORS = ("solve", "explain", "hint", "diagnose")
TEACHER_MOVES = ("focus", "probe", "tell", "generic")
GRADE_BANDS = ("secondary_9_12",)
DIFFICULTIES = ("easy", "medium", "hard")
ANSWER_POLICIES = ("reveal_final", "partial_reveal", "withhold_final")
INTERACTION_MODES = ("single_turn", "multi_turn")
SPLITS = ("train", "validation", "test", "external_eval")
ERROR_CLASSES = ("conceptual", "procedural")
ASSURANCE_TIERS = ("standard", "high")
SELECTION_DECISIONS = ("accept", "reject", "pending")
DEDUP_DECISIONS = ("keep", "reject", "adjudicate")
LICENSE_CLASSES = (
    "permissive",
    "share_alike",
    "noncommercial",
    "custom_restricted",
    "unknown",
    "excluded",
)
EXPORT_PROFILES = ("permissive", "sharealike", "noncommercial", "internal_only", "excluded")

MOVE_INTENTS: dict[str, tuple[str, ...]] = {
    "focus": ("seek_strategy", "guide_focus"),
    "probe": (
        "recall_relevant_fact",
        "seek_explanation",
        "seek_self_correction",
        "perturb_problem",
        "check_understanding",
    ),
    "tell": ("reveal_strategy", "reveal_step", "reveal_answer"),
    "generic": ("encourage", "greet_or_close", "general_inquiry"),
}
INTENTS = tuple(intent for intents in MOVE_INTENTS.values() for intent in intents)

PRINCIPLES = (
    "active_learning",
    "manage_cognitive_load",
    "adapt_to_learner",
    "stimulate_curiosity",
    "deepen_metacognition",
)

ERROR_TYPES = (
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
)

ROLES = ("system", "user", "assistant")
ALLOWED_MESSAGE_ORDERS = (
    ("user", "assistant"),
    ("system", "user", "assistant"),
)

CELL_TARGET = 500
CELL_MINIMUM = 400
