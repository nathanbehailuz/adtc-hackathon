"""Small deterministic fixtures used by Phase A validation tests."""

from __future__ import annotations

from typing import Any

from .identity import make_family_id, make_row_id, make_source_key
from .normalize import (
    constraint_signature,
    content_hash,
    message_hash,
    template_fingerprint,
)


def valid_diagnose_record() -> dict[str, Any]:
    source_key = make_source_key(
        "tebeb/deterministic-stem-v1",
        "fixture-revision",
        "train",
        "stoich_fixture_001",
    )
    family_id = make_family_id(
        generator_template_version="stoich-limiting-v1",
        surface_context="carbon-monoxide-oxidation",
    )
    items = [
        {"id": "identify_first_error", "check": "semantic_verifier", "parameters": {}},
        {"id": "no_final_numeric_answer", "check": "numeric_leak_v2", "parameters": {}},
    ]
    signature = constraint_signature(items)
    row_id = make_row_id(family_id, "diagnose", signature, "renderer-v1")
    problem = "Use 2 CO + O2 -> 2 CO2 to identify the limiting reactant."
    user = (
        "A student uses a 2:2 ratio for O2 to CO2. Identify the first error and "
        "give one hint without revealing the final mass."
    )
    assistant = (
        "The first error is the oxygen-to-carbon-dioxide mole ratio: the balanced "
        "equation shows one mole of O2 for two moles of CO2. Which coefficients "
        "should you compare before converting moles to mass?"
    )
    scores = {
        "grounding": 2,
        "answer_verifiability": 2,
        "curriculum_fit": 2,
        "behavior_affordance": 2,
        "clarity": 2,
        "diversity_value": 1,
    }
    eligibility = {
        behavior: {
            "eligible": behavior in {"solve", "hint", "diagnose"},
            "p_yes": None,
            "method": "deterministic_source",
        }
        for behavior in ("solve", "explain", "hint", "diagnose")
    }
    return {
        "schema_version": "tebeb-v8.1",
        "id": "detstem_stoich_fixture_001__diagnose__v1",
        "split": "train",
        "source": {
            "dataset": "tebeb/deterministic-stem-v1",
            "revision": "fixture-revision",
            "config": "default",
            "source_split": "train",
            "original_id": "stoich_fixture_001",
            "url": "repo:data/generators/stem_v1",
            "license": "project-generated",
            "license_class": "permissive",
            "export_profile": "permissive",
            "transformation": "detstem_controlled_error_diagnose_v1",
        },
        "curriculum": {
            "subject": "chemistry",
            "secondary_subjects": ["math"],
            "topic": "stoichiometry",
            "subtopic": "limiting_reactant",
            "grade_band": "secondary_9_12",
            "difficulty": "medium",
            "learning_objective": "Use mole ratios to identify a limiting reactant.",
        },
        "pedagogy": {
            "behavior": "diagnose",
            "teacher_move": "probe",
            "intent": "seek_self_correction",
            "principles": ["active_learning", "manage_cognitive_load"],
            "answer_policy": "withhold_final",
            "question_count": 1,
            "interaction_mode": "single_turn",
        },
        "grounding": {
            "problem": problem,
            "canonical_answer": "33 g CO2",
            "solution_steps": [
                "Read the balanced coefficients.",
                "Identify the limiting reactant.",
                "Convert product moles to mass.",
            ],
            "supporting_facts": ["The O2:CO2 coefficient ratio is 1:2."],
            "reference_explanation": None,
            "equations": ["2 CO + O2 -> 2 CO2"],
            "units": ["mol", "g"],
            "accepted_variants": ["33 grams of carbon dioxide"],
        },
        "student_state": {
            "attempt": "The ratio of O2 to CO2 is 2:2.",
            "first_error_step": 1,
            "error_type": "ratio_error",
            "error_class": "conceptual",
            "error_class_rationale": "The balanced coefficients are misunderstood.",
            "misconception": "The student assigns oxygen the coefficient 2.",
        },
        "constraints": {
            "items": items,
            "forbidden_answers": ["33 g CO2"],
            "constraint_results": {
                "identify_first_error": {"passed": True},
                "no_final_numeric_answer": {
                    "passed": True,
                    "matched_variants": [],
                },
            },
            "all_constraints_pass": True,
        },
        "messages": [
            {
                "role": "system",
                "content": "Act as a secondary-school chemistry tutor.",
            },
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "quality": {
            "generator": "fixture/deterministic",
            "verifiers": ["fixture/deterministic"],
            "factual_score": 1.0,
            "pedagogy_score": 1.0,
            "constraint_pass": True,
            "curator_review": "accepted",
            "curator_model": None,
            "model_independence": "not-applicable",
            "review_passes": 1,
            "assurance_tier": "standard",
            "selection": {
                "decision": "accept",
                "subject_method": "source_metadata",
                "subject_distribution": None,
                "subject_choice_confidence": None,
                "behavior_eligibility": eligibility,
                "behavior_classifier": "deterministic/template-v1",
                "typesafe_sdk": None,
                "selected_behavior": "diagnose",
                "scores": scores,
                "total": sum(scores.values()),
                "rejection_reasons": [],
            },
        },
        "identity": {
            "source_key": source_key,
            "family_id": family_id,
            "row_id": row_id,
            "record_version": 1,
            "parent_row_ids": [],
        },
        "dedup": {
            "normalization_version": "norm-v1",
            "canonical_content_hash": content_hash(problem),
            "prompt_hash": message_hash(user, role="user"),
            "assistant_hash": message_hash(assistant, role="assistant"),
            "constraint_signature": signature,
            "template_fingerprint": template_fingerprint(problem),
            "semantic_cluster_id": None,
            "nearest_eval_similarity": None,
            "duplicate_of": None,
            "decision": "keep",
        },
    }
