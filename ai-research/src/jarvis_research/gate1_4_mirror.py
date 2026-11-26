"""HZN-IR-12 — mirror experiment construction and grading.

Read-only, deterministic machinery for testing whether the model can
re-examine its own prior answers, classify the visible failure mechanics,
and revise toward evidence rather than merely revise. No outcome of this
module authorizes any intervention.
"""

import random
import re

from jarvis_research.gate1_4_benchmark import (
    LABELS, RULES, grade_task, magnitude_answer, render_task_body)


MIRROR_CAUSES = (
    "no_failure",
    "none_qualify",
    "tie",
    "missing_observation",
    "threshold_error",
    "tiebreak_error",
    "missed_qualifier",
    "unknown",
)

CAUSE_DESCRIPTIONS = {
    "no_failure": "the previous answer is correct",
    "none_qualify": "no candidate qualifies, but a candidate was selected",
    "tie": "qualifying candidates tie exactly, but one candidate was selected",
    "missing_observation": "the selected candidate relies on unavailable observations",
    "threshold_error": "the selected candidate fails an ar, contrast, or sign requirement",
    "tiebreak_error": "the selected candidate qualifies but is not the tie-break winner",
    "missed_qualifier": "the previous answer was none even though a candidate qualifies",
    "unknown": "the evidence shown does not determine the failure type",
}


def normalize_answer(answer):
    a = str(answer).strip()
    if a.lower() == "none":
        return "none"
    return a.upper()


def permuted_relabel_task(task, seed):
    """Return an independently permuted, relabeled copy of a task.

    Greedy decoding makes a verbatim retry bit-identical, so M0 permutes
    candidate order. Labels are reassigned by position, matching the
    benchmark's original no-position-leak construction. The returned
    label_map maps original labels to M0 labels.
    """
    rng = random.Random(seed)
    old = [dict(c) for c in task["candidates"]]
    n = len(old)
    perm = list(range(n))
    while True:
        rng.shuffle(perm)
        if perm != list(range(n)):
            break
    cands = []
    label_map = {}
    for new_i, old_i in enumerate(perm):
        c = dict(old[old_i])
        label_map[c["label"]] = LABELS[new_i]
        c["label"] = LABELS[new_i]
        cands.append(c)
    out = {"family": task.get("family", "unknown"),
           "rule_key": task["rule_key"], "candidates": cands}
    out["truth"] = grade_task(task["rule_key"], cands)
    out["magnitude_answer"] = magnitude_answer(cands)
    out["label_map"] = label_map
    return out


def candidate_evaluation(rule_key, candidate):
    """Deterministic per-candidate rule evidence. This exposes intermediate
    computation without revealing the final answer."""
    r = RULES[rule_key]
    if not candidate["obs_available"]:
        return {"obs_available": False, "qualifies": False,
                "ar_pass": None, "contrast_pass": None,
                "tiebreak_value": None}
    ar_pass = candidate["ar"] >= r["ar_thr"]
    if rule_key == "avoid_negative":
        contrast_pass = candidate["lc"] >= 0
        tiebreak_value = abs(candidate["lc"])
    else:
        contrast_pass = abs(candidate["lc"]) >= r["lc_thr"]
        tiebreak_value = (candidate["ar"] if rule_key == "conj_active"
                          else abs(candidate["lc"]))
    return {"obs_available": True,
            "ar_pass": ar_pass, "contrast_pass": contrast_pass,
            "qualifies": bool(ar_pass and contrast_pass),
            "tiebreak_value": tiebreak_value}


def _pf(flag):
    return "pass" if flag else "fail"


def render_evidence_expansion(rule_key, candidates):
    """Render deterministic intermediate rule evidence for M3."""
    r = RULES[rule_key]
    lines = []
    for c in candidates:
        ev = candidate_evaluation(rule_key, c)
        if not ev["obs_available"]:
            lines.append(f"candidate {c['label']}: observations unavailable; "
                         f"cannot qualify")
            continue
        if rule_key == "avoid_negative":
            contrast = (f"lc={c['lc']:.2f} {_pf(ev['contrast_pass'])} "
                        f"(requires lc>=0)")
            tie_name = "|lc|"
        else:
            contrast = (f"|lc|={abs(c['lc']):.2f} "
                        f"{_pf(ev['contrast_pass'])} "
                        f"(requires |lc|>={r['lc_thr']:.2f})")
            tie_name = "ar" if rule_key == "conj_active" else "|lc|"
        lines.append(
            f"candidate {c['label']}: ar={c['ar']:.2f} "
            f"{_pf(ev['ar_pass'])} (requires ar>={r['ar_thr']:.2f}); "
            f"{contrast}; qualifies={'yes' if ev['qualifies'] else 'no'}; "
            f"tie-break {tie_name}={ev['tiebreak_value']:.2f}")
    return "\n".join(lines)


def render_cause_instructions():
    lines = ["Cause labels:"]
    for cause in MIRROR_CAUSES:
        lines.append(f"- {cause}: {CAUSE_DESCRIPTIONS[cause]}")
    return "\n".join(lines)


def render_mirror_prompt(task, previous_answer, verdict=None,
                         expanded_evidence=False):
    """Render an M1/M2/M3 mirror prompt.

    previous_answer is the model's own prior T2 answer. verdict, when
    present, is only 'correct' or 'incorrect' — never the true answer.
    Expanded evidence exposes deterministic intermediate rule tests but
    still does not state the correct answer.
    """
    prev = normalize_answer(previous_answer)
    parts = [render_task_body(task),
             f"Previous answer: {prev}"]
    if verdict is not None:
        parts.append(f"Independent correctness verdict: {verdict}")
    if expanded_evidence:
        parts.append("Deterministic evidence expansion:\n" +
                     render_evidence_expansion(task["rule_key"],
                                               task["candidates"]))
    parts.append(
        "Re-examine the task and the previous answer. Classify the visible "
        "failure mechanics of the previous answer, then give the answer you "
        "now judge correct.")
    parts.append(render_cause_instructions())
    parts.append(
        "Reply with exactly two lines. The first line must start 'cause: ' "
        "followed by one cause label. The second line must start 'answer: ' "
        "followed by one candidate letter or none. No other text.")
    return "\n\n".join(parts)


def parse_mirror_response(response, valid_labels):
    """Strictly parse the two-line mirror output contract."""
    nonempty = [ln.strip() for ln in str(response).splitlines() if ln.strip()]
    if len(nonempty) != 2:
        return {"parse_status": "invalid_format"}
    m1 = re.fullmatch(r"cause:\s*([A-Za-z_]+)", nonempty[0],
                      flags=re.IGNORECASE)
    m2 = re.fullmatch(r"answer:\s*([A-Za-z]+)", nonempty[1],
                      flags=re.IGNORECASE)
    if not m1 or not m2:
        return {"parse_status": "invalid_format"}
    cause = m1.group(1).lower()
    answer = normalize_answer(m2.group(1))
    if cause not in MIRROR_CAUSES:
        return {"parse_status": "invalid_cause", "cause": cause,
                "answer": answer}
    if answer not in valid_labels:
        return {"parse_status": "invalid_answer", "cause": cause,
                "answer": answer}
    return {"parse_status": "valid", "cause": cause, "answer": answer}


def answer_verdict(answer, expected, valid_labels):
    """Mirror-compatible answer grading using the IR-11 verdict vocabulary."""
    if expected == "none":
        return "correct" if answer == "none" else "invalid_pick"
    if answer == "none":
        return "abstain_when_answer"
    if answer == expected:
        return "correct"
    if answer in valid_labels:
        return "valid_but_incorrect"
    return "invalid_reference"


def reference_failure_cause(rule_key, candidates, previous_answer):
    """Best mechanical reference class for the visible failure outcome.

    These classes describe outcome mechanics, not certain knowledge of the
    model's internal cognitive cause. Global none outcomes dominate because
    the decisive error is failing to respect the rule's none branch.
    """
    truth = grade_task(rule_key, candidates)
    prev = normalize_answer(previous_answer)
    if prev == truth["answer"]:
        return "no_failure"
    if truth["answer"] == "none":
        return "tie" if truth["reason"] == "tie" else "none_qualify"
    if prev == "none":
        return "missed_qualifier"
    prev_c = next((c for c in candidates if c["label"] == prev), None)
    if prev_c is None:
        return "unknown"
    if not prev_c["obs_available"]:
        return "missing_observation"
    if not candidate_evaluation(rule_key, prev_c)["qualifies"]:
        return "threshold_error"
    return "tiebreak_error"


def grade_mirror_response(response, expected, previous_answer, valid_labels,
                          reference_cause):
    """Grade answer revision and diagnosis separately. Parse failures never
    count as behavioral evidence in either direction."""
    parsed = parse_mirror_response(response, valid_labels)
    out = dict(parsed)
    out["reference_cause"] = reference_cause
    if parsed["parse_status"] != "valid":
        out.update({"answer_verdict": parsed["parse_status"],
                    "answer_correct": False,
                    "changed_from_previous": False,
                    "cause_correct": False})
        return out
    prev = normalize_answer(previous_answer)
    verdict = answer_verdict(parsed["answer"], expected, valid_labels)
    out.update({
        "answer_verdict": verdict,
        "answer_correct": verdict == "correct",
        "changed_from_previous": parsed["answer"] != prev,
        "cause_correct": parsed["cause"] == reference_cause,
    })
    return out
