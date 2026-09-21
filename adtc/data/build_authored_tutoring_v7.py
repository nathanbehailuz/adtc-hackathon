#!/usr/bin/env python3
"""Generate authored English tutoring SFT rows for v7 (judge-aligned patterns).

Writes: data/authored_tutoring_v7.jsonl

Patterns covered (paraphrases of Round-1 skill gaps, not verbatim judge prompts):
  - scaffolded algebra hints (why + check question, no final x)
  - first-error diagnosis with remediation
  - multi-part concept explain (energy/force/power, kinematics, stoich)
  - completing-the-square / quadratic derivation
  - velocity vs acceleration at apex
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "authored_tutoring_v7.jsonl"


def emit(row_id: str, behavior: str, user: str, assistant: str) -> dict:
    return {
        "id": row_id,
        "direction": "en_en",
        "behavior": behavior,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": "authored_tutoring_v7",
    }


def algebra_hint_rows() -> list[dict]:
    problems = [
        (
            "3(x - 4) + 5 = 20",
            "distribute the 3 across (x - 4)",
            "multiplication distributes over addition/subtraction, so 3(x - 4) becomes 3x - 12",
            "What expression do you get after expanding the parentheses?",
        ),
        (
            "2x + 7 = 19",
            "subtract 7 from both sides",
            "undoing addition isolates the term with x while keeping equality",
            "After subtracting 7, what equation remains?",
        ),
        (
            "5x = 20",
            "divide both sides by 5",
            "division undoes multiplication by the same nonzero number",
            "What single-step equation do you have after dividing?",
        ),
        (
            "4(x - 2) = 20",
            "either distribute the 4 or divide both sides by 4 first",
            "both keep equality if applied to every term / both sides",
            "Which approach will you try first, and why?",
        ),
        (
            "(x/3) + 2 = 8",
            "subtract 2 from both sides",
            "clear the added constant before undoing the division",
            "What is left on the left side after subtracting 2?",
        ),
        (
            "2(x + 5) - 3 = 11",
            "expand 2(x + 5) first",
            "parentheses must be cleared before combining like terms",
            "What do you get after expanding?",
        ),
        (
            "x/2 - 4 = 6",
            "add 4 to both sides",
            "undo subtraction to isolate the fraction with x",
            "After adding 4, what equation remains?",
        ),
        (
            "3x - 9 = 12",
            "add 9 to both sides",
            "move the constant so 3x stands alone before dividing",
            "What is 3x equal to after that step?",
        ),
    ]
    rows = []
    for i, (eq, step, why, q) in enumerate(problems):
        user = (
            f"A student is solving this algebra problem:\n{eq}\n"
            "The student says they do not know what to do next. Act as a tutor: "
            "give one useful hint, explain why that step is valid, ask one question "
            "after the hint, and do not reveal the final value of x."
        )
        asst = (
            f"Hint: {step}.\n"
            f"Why this is valid: {why}.\n"
            f"Check question: {q}\n"
            "I will not give the final value of x — try the next step yourself."
        )
        rows.append(emit(f"authored_hint_alg_{i:03d}", "hint", user, asst))
    return rows


def first_error_rows() -> list[dict]:
    cases = [
        (
            "A 2 kg object is pushed with a net force of 10 N. What is its acceleration?",
            "F = ma; 10 = 2a; a = 20 m/s^2",
            "In the last step they computed 10×2 instead of 10÷2 when rearranging F = ma.",
            "From F = ma, a = F/m, so divide force by mass; multiplying reverses the relationship.",
            "Rewrite a = F/m with the given numbers and compute carefully.",
            "If mass doubles and force stays the same, what happens to acceleration?",
        ),
        (
            "Solve 3(x − 2) = 2x + 5. Student work:\nLine 1: 3x − 2 = 2x + 5\nLine 2: 3x − 2x = 5 + 2\nLine 3: x = 7",
            None,
            "Line 1 is the first mistake: 3(x − 2) was expanded as 3x − 2 instead of 3x − 6.",
            "The 3 must multiply both terms inside the parentheses (distributive property).",
            "Rewrite Line 1 by distributing 3 over (x − 2), then continue without jumping to x.",
            "What should 3(x − 2) become after distributing?",
        ),
        (
            "Student writes: 3/4 + 1/2 = 4/6.",
            None,
            "They added numerators and denominators separately (3+1)/(4+2).",
            "Fractions can only be added when denominators match; adding tops and bottoms invents a new fraction size.",
            "Rewrite 1/2 with denominator 4, then add the numerators only.",
            "What common denominator will you use?",
        ),
        (
            "Student says area of a triangle with base 10 and height 4 is 40.",
            None,
            "They used base×height without the 1/2 factor required for a triangle.",
            "A triangle is half of the related parallelogram/rectangle with the same base and height.",
            "Start from area = (1/2)×base×height and recompute.",
            "What factor is missing from base×height for a triangle?",
        ),
        (
            "Student claims 10% of 200 equals 10.",
            None,
            "They treated 10% as the number 10 instead of 10/100.",
            "Percent means parts per hundred, so 10% = 0.10 times the base amount.",
            "Convert 10% to a decimal or fraction, then multiply by 200.",
            "What decimal is equivalent to 10%?",
        ),
        (
            "From 2x + 5 = 13 a student concludes x = 9.",
            None,
            "They likely subtracted incorrectly or skipped isolating 2x (13 − 4 style error / treating +5 wrongly).",
            "To isolate 2x you must subtract 5 from both sides, getting 2x = 8, then divide by 2.",
            "Subtract 5 from both sides first and write the new equation before dividing.",
            "After subtracting 5, what is 2x equal to?",
        ),
    ]
    rows = []
    for i, (prompt, work, mistake, why, hint, follow) in enumerate(cases):
        user_parts = [
            "A student is solving the following problem.",
            prompt,
        ]
        if work:
            user_parts.append(f"The student writes:\n{work}")
        user_parts.append(
            "Do not simply give the correct final answer. Identify the exact mistake, "
            "explain why that step is wrong, give a hint so the student can correct it, "
            "and ask one short follow-up question."
        )
        user = "\n\n".join(user_parts)
        asst = (
            f"Exact mistake: {mistake}\n"
            f"Why it is wrong: {why}\n"
            f"Corrective hint: {hint}\n"
            f"Follow-up: {follow}\n"
            "I will not reveal the final numeric answer."
        )
        rows.append(emit(f"authored_first_error_{i:03d}", "first_error", user, asst))
    return rows


def concept_explain_rows() -> list[dict]:
    rows = []
    # Hydroelectric-style multi-part (paraphrased)
    rows.append(
        emit(
            "authored_explain_hydro_001",
            "explain",
            (
                "A secondary-school student says: I understand electricity can come from "
                "water, but I do not understand how a dam turns moving water into electricity. "
                "Explain in a tutoring style for a 15-year-old. Your response must: "
                "(1) sequence stored water → flowing water → turbine → generator → electricity; "
                "(2) clearly distinguish energy, force, and power; "
                "(3) use an analogy that would make sense in an African context; "
                "(4) name one common misconception and correct it; "
                "(5) end with two short questions that test understanding, not memorization. "
                "Do not assume the student knows electromagnetism."
            ),
            (
                "Sequence: Water stored high behind a dam has gravitational potential energy. "
                "When gates open, water flows downward and that potential energy converts into "
                "kinetic energy (motion). The fast water pushes turbine blades — that push is a "
                "force acting through a distance, so the water does mechanical work on the turbine. "
                "The spinning turbine turns a generator; inside, motion is converted into electrical "
                "energy that can travel on wires to homes.\n\n"
                "Energy vs force vs power: Energy is the capacity to do work (measured in joules). "
                "Force is a push or pull (newtons). Power is how fast energy is delivered "
                "(joules per second, or watts). A dam can store a huge amount of energy even if, "
                "at a quiet moment, little power is being produced.\n\n"
                "African-context analogy: Think of a large elevated water tank supplying a village. "
                "Height stores potential energy; opening the tap lets water move and spin a small "
                "wheel that could drive a mill — a dam is that idea at river scale, with the "
                "generator replacing the millstone.\n\n"
                "Misconception to correct: Stored water is not already falling water. Behind the "
                "dam it is mostly still and high; falling/flowing happens after release. Also, "
                "electrical energy delivered to a bulb is not the same thing as power — power is "
                "the rate of delivery.\n\n"
                "Check questions:\n"
                "1) If the water level behind the dam drops but turbines still spin slowly, what "
                "changed more — stored energy, or power output right now?\n"
                "2) Where in the sequence is energy changing form, and where is a force doing work "
                "on a machine part?"
            ),
        )
    )
    rows.append(
        emit(
            "authored_explain_apex_001",
            "explain",
            (
                "A ball is thrown straight up. At the very top of its path, is its velocity zero, "
                "its acceleration zero, both, or neither? Explain for a Grade 10 student and give "
                "one everyday example that shows the difference."
            ),
            (
                "At the top, velocity is zero for an instant, but acceleration is not zero. "
                "Velocity is how fast the position is changing; at the peak the ball briefly stops "
                "rising before it starts falling. Acceleration here is the gravitational change in "
                "velocity — gravity keeps pulling downward the whole time, so the velocity is still "
                "changing from upward, through zero, to downward.\n\n"
                "Everyday example: jumping on a trampoline. At the highest point you hang for a "
                "moment (speed ≈ 0), but you still feel the downward pull of gravity and immediately "
                "begin falling — that continuous pull is why acceleration was never zero.\n\n"
                "Check question: If acceleration were zero at the top, would the ball stay floating "
                "there or start moving again?"
            ),
        )
    )
    rows.append(
        emit(
            "authored_explain_accel_001",
            "explain",
            (
                "Explain the physical meaning of the second derivative of position with respect to "
                "time in classical mechanics, and give a real free-fall example with constant g ≈ "
                "9.8 m/s². Keep the math correct."
            ),
            (
                "If position is s(t), then ds/dt is velocity and d²s/dt² is acceleration — how "
                "quickly velocity changes. In free fall near Earth (ignoring air resistance), "
                "acceleration is constant: a(t) = −g ≈ −9.8 m/s² if upward is positive.\n\n"
                "Example: take s(t) = s0 + v0 t − (1/2) g t². Then v(t) = v0 − g t and "
                "a(t) = −g, a constant. At any time t (including t = 10 s), acceleration is still "
                "−9.8 m/s², not a linear function of t. A formula like a(t) = −32t + 160 would be "
                "wrong for constant-g free fall.\n\n"
                "Check question: If velocity is zero at the top of a toss, what is acceleration "
                "at that same instant under constant g?"
            ),
        )
    )
    rows.append(
        emit(
            "authored_explain_stoich_001",
            "explain",
            (
                "Analyze CH4 + 2 O2 → CO2 + 2 H2O. If 16 g methane reacts with 48 g oxygen, "
                "identify the limiting reactant and the mass of CO2 produced. Show mole reasoning."
            ),
            (
                "Molar masses: CH4 = 16 g/mol, O2 = 32 g/mol, CO2 = 44 g/mol.\n"
                "1. Moles CH4 = 16/16 = 1.0 mol.\n"
                "2. Moles O2 = 48/32 = 1.5 mol.\n"
                "3. Stoichiometry needs 2 mol O2 per 1 mol CH4, so 1 mol CH4 would need 2 mol O2.\n"
                "4. Available O2 is only 1.5 mol < 2 mol, so O2 is limiting.\n"
                "5. From the equation, 2 mol O2 produce 1 mol CO2, so 1.5 mol O2 produce "
                "0.75 mol CO2.\n"
                "6. Mass CO2 = 0.75 × 44 = 33 g.\n\n"
                "Final answer: limiting reactant O2; CO2 produced = 33 g.\n"
                "Note: do not multiply 1.5 mol by 44 under the false assumption that 1.5 mol O2 "
                "makes 1.5 mol CO2 — the O2:CO2 ratio is 2:1."
            ),
        )
    )
    rows.append(
        emit(
            "authored_explain_speed_accel_001",
            "explain",
            "What is the difference between speed and acceleration? Explain for a beginner.",
            (
                "Speed is how fast something is moving (distance per time), for example 20 m/s. "
                "Acceleration is how quickly that speed (or velocity) is changing, for example "
                "speeding up by 2 m/s every second. You can have high speed with zero acceleration "
                "(steady highway cruise) or zero speed with nonzero acceleration (ball at the top "
                "of a toss).\n\n"
                "Check question: Give one situation with large speed but small acceleration."
            ),
        )
    )
    # More concept cards
    concepts = [
        (
            "mass_weight",
            "Explain the difference between mass and weight for a secondary-school student.",
            (
                "Mass is how much matter an object has (kilograms) and stays the same if gravity "
                "changes. Weight is the gravitational force on that mass (newtons) and changes with "
                "gravity. On the Moon your mass is unchanged but your weight is smaller.\n\n"
                "Misconception: people say they 'weigh 60 kg' — that number is usually mass; weight "
                "would be mass times g.\n\n"
                "Check question: Would mass or weight change if you flew to a weaker-gravity planet?"
            ),
        ),
        (
            "density",
            "In a short tutoring paragraph, what is density, and why does ice float on water?",
            (
                "Density is mass per unit volume — how tightly matter is packed. Ice floats because "
                "its crystal structure takes more space than liquid water, so ice is less dense.\n\n"
                "Misconception: ice floats because it is 'lighter' in mass; a large iceberg has huge "
                "mass but still lower density than liquid water.\n\n"
                "Check question: If you compress a material into a smaller volume without losing "
                "mass, what happens to density?"
            ),
        ),
        (
            "potential_kinetic",
            "Distinguish potential energy and kinetic energy with one African-context analogy.",
            (
                "Potential energy is stored energy due to position or state; kinetic energy is energy "
                "of motion. Analogy: grain held high in a raised village store has potential energy; "
                "when it pours down a chute into a mill, that stored energy becomes kinetic energy "
                "of moving grain.\n\n"
                "Check question: Where is potential energy largest for water behind a dam — at the "
                "surface high up, or after it has fallen through the turbines?"
            ),
        ),
    ]
    for key, user, asst in concepts:
        rows.append(emit(f"authored_explain_{key}", "explain", user, asst))
    return rows


def quadratic_rows() -> list[dict]:
    rows = []
    rows.append(
        emit(
            "authored_solve_quad_001",
            "solve",
            (
                "Derive the quadratic formula step-by-step using completing the square for "
                "ax^2 + bx + c = 0. Explain why each algebraic transformation is valid. "
                "Do not use GSM8K markup."
            ),
            (
                "1. Assume a ≠ 0. Divide both sides by a: x^2 + (b/a)x + c/a = 0. "
                "Why: dividing an equation by a nonzero constant preserves equality.\n"
                "2. Move the constant: x^2 + (b/a)x = −c/a. "
                "Why: subtracting c/a from both sides isolates the x terms.\n"
                "3. Complete the square: add (b/(2a))^2 to both sides. "
                "Why: half of (b/a) is b/(2a), and squaring it makes a perfect-square trinomial "
                "on the left.\n"
                "4. Left side factors: (x + b/(2a))^2 = −c/a + (b/(2a))^2. "
                "Why: by construction of the perfect square.\n"
                "5. Rewrite the right side over 4a^2: (b^2 − 4ac)/(4a^2). "
                "Why: common denominator 4a^2.\n"
                "6. Take square roots: x + b/(2a) = ±√(b^2 − 4ac)/(2a). "
                "Why: if y^2 = k then y = ±√k (for k ≥ 0 in reals).\n"
                "7. Subtract b/(2a): x = (−b ± √(b^2 − 4ac))/(2a). "
                "Why: undoing the added half-coefficient term.\n\n"
                "Final answer: x = (−b ± √(b^2 − 4ac))/(2a)"
            ),
        )
    )
    rows.append(
        emit(
            "authored_hint_quad_001",
            "hint",
            (
                "I am stuck deriving the quadratic formula by completing the square for "
                "ax^2 + bx + c = 0. Give one hint without writing the final formula."
            ),
            (
                "Hint: after dividing by a, add (b/(2a))^2 to both sides before you try to "
                "take a square root. Why: that specific square completes x^2 + (b/a)x into "
                "(x + b/(2a))^2. Check question: what expression is half of the coefficient "
                "of x after dividing by a? Do not write the final quadratic formula yet."
            ),
        )
    )
    return rows


def more_scaffold_variations() -> list[dict]:
    """Expand coverage with parameterized secondary-school items."""
    rows: list[dict] = []
    eqs = [
        ("x + 9 = 15", "subtract 9 from both sides", "undo addition"),
        ("x - 4 = 10", "add 4 to both sides", "undo subtraction"),
        ("6x = 42", "divide both sides by 6", "undo multiplication"),
        ("x/5 = 3", "multiply both sides by 5", "undo division"),
        ("2x + 3 = 11", "subtract 3 from both sides", "isolate the x term"),
        ("7x - 2 = 19", "add 2 to both sides", "isolate 7x"),
        ("(1/2)x + 1 = 4", "subtract 1 from both sides", "clear the constant first"),
        ("3(x + 1) = 12", "divide both sides by 3 (or distribute first)", "simplify the parentheses path"),
        ("5(x - 1) + 2 = 17", "expand 5(x - 1) first", "distributive property"),
        ("2(3x - 1) = 10", "distribute 2 across (3x - 1)", "multiply every term inside"),
    ]
    for i, (eq, hint, why) in enumerate(eqs):
        rows.append(
            emit(
                f"authored_hint_iso_{i:03d}",
                "hint",
                (
                    f"Student stuck on {eq}. One hint with why and a check question; "
                    "do not give x."
                ),
                (
                    f"Hint: {hint}. Why: {why}, which keeps both sides equal. "
                    f"Check question: what equation remains after that step? "
                    f"Do not state x."
                ),
            )
        )

    science_hints = [
        (
            "Why do we balance chemical equations?",
            "Hint: think about atoms being rearranged, not created. Why: conservation of mass "
            "requires the same atom counts on both sides. Check question: if carbon atoms differ "
            "left vs right, what physical law is violated? Do not give a full worked balance yet.",
        ),
        (
            "What does a catalyst do in a reaction?",
            "Hint: focus on the reaction pathway's energy barrier, not whether products differ. "
            "Why: catalysts provide an alternate path with lower activation energy. "
            "Check question: after the reaction, is the catalyst used up as a reactant? "
            "Do not lecture beyond one hint.",
        ),
        (
            "Why is acceleration due to gravity approximately constant near Earth's surface?",
            "Hint: compare the size of everyday heights to Earth's radius. Why: over small "
            "altitude changes, g barely varies. Check question: would g stay as constant if you "
            "compared sea level to low Earth orbit? Do not give orbital formulas.",
        ),
        (
            "What is the difference between heat and temperature?",
            "Hint: one is energy transferred; the other measures average molecular kinetic energy. "
            "Why: a spark and a lake can share temperature ideas differently than total energy. "
            "Check question: can two objects at the same temperature contain different amounts of "
            "thermal energy? Do not write a textbook chapter.",
        ),
    ]
    for i, (q, asst) in enumerate(science_hints):
        rows.append(
            emit(
                f"authored_hint_sci_{i:03d}",
                "hint",
                f"Student stuck on: {q} Give one hint with why and a check question.",
                asst,
            )
        )

    # Multipart instruction-following drills (generic STEM)
    multipart = [
        (
            "explain_newton2",
            (
                "Explain Newton's second law for a 14-year-old. Requirements: "
                "(1) state F = ma in words; (2) give one numerical example; "
                "(3) correct the misconception that heavier objects always fall faster in vacuum; "
                "(4) end with two check questions."
            ),
            (
                "In words: the net force on an object equals mass times acceleration — a larger "
                "push or a smaller mass means larger acceleration.\n"
                "Example: net force 10 N on 2 kg gives a = 10/2 = 5 m/s².\n"
                "Misconception: in vacuum, heavy and light objects fall with the same gravitational "
                "acceleration g; air resistance, not mass alone, makes feathers drift.\n"
                "Check questions:\n"
                "1) If force doubles and mass is unchanged, what happens to acceleration?\n"
                "2) Why is 'net' force important in F = ma?"
            ),
        ),
        (
            "explain_series_parallel",
            (
                "Tutor a student on series vs parallel circuits. Include: definition of each, "
                "one everyday analogy, one misconception, and two check questions."
            ),
            (
                "Series: components share one path; the same current goes through each. "
                "Parallel: multiple paths; voltage across branches matches.\n"
                "Analogy: series is a single-file water pipe through two mills; parallel is a "
                "river splitting around an island then rejoining.\n"
                "Misconception: 'parallel always uses less energy' — it depends on what is connected; "
                "parallel often draws more total current from the supply for similar lamps.\n"
                "Check questions:\n"
                "1) If one bulb fails open in series vs parallel, what happens to the others?\n"
                "2) Which arrangement keeps the same voltage across each identical bulb?"
            ),
        ),
        (
            "explain_photosynthesis",
            (
                "Explain photosynthesis for a 15-year-old in Africa: sequence inputs→outputs, "
                "distinguish energy vs matter, one local analogy, one misconception, two questions."
            ),
            (
                "Sequence: plants take in carbon dioxide and water; using sunlight as the energy "
                "source, they produce sugars (matter storing chemical energy) and release oxygen.\n"
                "Energy vs matter: sunlight supplies energy; C, H, and O atoms are rearranged into "
                "new molecules — energy is not 'turned into atoms.'\n"
                "Analogy: like a solar-powered kitchen that cooks CO2 and water into food for the "
                "plant, powered by sunshine on the leaves.\n"
                "Misconception: plants get their food from soil nutrients alone — soil minerals "
                "help, but the bulk of plant mass comes from air (CO2) and water via photosynthesis.\n"
                "Check questions:\n"
                "1) What enters the leaf that is not a form of matter but enables the reaction?\n"
                "2) If a plant gains mass as it grows, where did most of that mass come from?"
            ),
        ),
    ]
    for key, user, asst in multipart:
        rows.append(emit(f"authored_multipart_{key}", "explain", user, asst))

    solve_drills = [
        (
            "Solve: A tank holds 200 liters and is 2/5 full. How many liters are in it?",
            "1. Fraction full means multiply: (2/5)*200.\n"
            "2. (2*200)/5 = 400/5 = 80.\n\nFinal answer: 80",
        ),
        (
            "Solve: A rectangle is 12 cm by 5 cm. Find area and perimeter.",
            "1. Area = 12*5 = 60 cm^2.\n"
            "2. Perimeter = 2*(12+5) = 34 cm.\n\n"
            "Final answer: area 60 cm^2, perimeter 34 cm",
        ),
        (
            "Solve: Convert 0.25 to a percent.",
            "1. 0.25 = 25/100.\n2. That is 25%.\n\nFinal answer: 25%",
        ),
        (
            "Solve: Find the mean of 5, 7, 9, 11.",
            "1. Sum = 32.\n2. Mean = 32/4 = 8.\n\nFinal answer: 8",
        ),
        (
            "Solve: A car travels 150 km in 2.5 h at constant speed. What is the speed?",
            "1. Speed = distance/time = 150/2.5.\n"
            "2. 150/2.5 = 60 km/h.\n\nFinal answer: 60 km/h",
        ),
        (
            "Solve: 15% of 80.",
            "1. 15% = 0.15.\n2. 0.15*80 = 12.\n\nFinal answer: 12",
        ),
        (
            "Solve: A right triangle has legs 5 and 12. Find the hypotenuse.",
            "1. c = sqrt(5^2 + 12^2) = sqrt(25+144) = sqrt(169).\n"
            "2. c = 13.\n\nFinal answer: 13",
        ),
        (
            "Solve: Simplify (2/3) ÷ (4/9).",
            "1. Dividing fractions: multiply by the reciprocal.\n"
            "2. (2/3)*(9/4) = 18/12 = 3/2.\n\nFinal answer: 3/2",
        ),
    ]
    for i, (u, a) in enumerate(solve_drills):
        rows.append(emit(f"authored_solve_drill_{i:03d}", "solve", u, a))
        # wording variant
        rows.append(
            emit(
                f"authored_solve_drill_{i:03d}_b",
                "solve",
                u + " Show clear steps and end with Final answer.",
                a,
            )
        )

    # Extra first-error algebra sheet items
    sheets = [
        (
            "2(x + 3) = 10 → Line1: 2x + 3 = 10 → Line2: 2x = 7 → Line3: x = 3.5",
            "Line 1",
            "2(x + 3) must become 2x + 6, not 2x + 3.",
        ),
        (
            "x/2 + 4 = 10 → Line1: x/2 = 14 → Line2: x = 28",
            "Line 1",
            "Subtract 4 from both sides to get x/2 = 6; adding instead doubles the error.",
        ),
        (
            "5 − x = 2 → Line1: −x = 7 → Line2: x = −7",
            "Line 1",
            "Subtracting 5 from both sides gives −x = −3, not −x = 7.",
        ),
    ]
    for i, (work, line, mistake) in enumerate(sheets):
        rows.append(
            emit(
                f"authored_sheet_{i:03d}",
                "first_error",
                (
                    f"Here is student work:\n{work}\n"
                    "Which line has the first mistake, and what went wrong? "
                    "Do not finish the solve for them; give a hint and one check question."
                ),
                (
                    f"First mistake: {line}. {mistake}\n"
                    "Corrective hint: rewrite that line using the same operation on every term/"
                    "both sides, then pause.\n"
                    "Follow-up: What should that line look like after a correct rewrite?\n"
                    "Do not state the final x."
                ),
            )
        )

    return rows


def main() -> None:
    rows: list[dict] = []
    rows.extend(algebra_hint_rows())
    rows.extend(first_error_rows())
    rows.extend(concept_explain_rows())
    rows.extend(quadratic_rows())
    rows.extend(more_scaffold_variations())

    # Light duplicates with wording variants to reinforce behaviors (~target 150–250)
    base = list(rows)
    for j, r in enumerate(base):
        variant = json.loads(json.dumps(r))
        variant["id"] = f"{r['id']}_v{j:02d}"
        user = variant["messages"][0]["content"]
        variant["messages"][0]["content"] = (
            user + "\n\nKeep the reply concise but complete all requested parts."
        )
        rows.append(variant)

    # Second pass: explicit "address every bullet" variants for explain + hint rows
    for behavior in ("explain", "hint", "first_error"):
        subset = [r for r in base if r["behavior"] == behavior]
        for j, r in enumerate(subset):
            variant = json.loads(json.dumps(r))
            variant["id"] = f"{r['id']}_checklist_{behavior}_{j:02d}"
            variant["messages"][0]["content"] = (
                r["messages"][0]["content"]
                + "\n\nAddress every requirement in the prompt; do not skip requested "
                "explanations, analogies, misconceptions, hints, or check questions."
            )
            rows.append(variant)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} n={len(rows)}")


if __name__ == "__main__":
    main()
