"""HZN-ORIENT-01 — orientation scaffold and salience-capture probe.

Tests whether a compact center / who-what-when-where-how / why orientation
frame changes behavior on salience-sensitive candidate reasoning tasks.
This is a prompt-level behavioral probe, not evidence of an implemented
orientation architecture.
"""

import hashlib
import random

from jarvis_research.gate1_4_benchmark import (
    FAMILIES, make_task, render_task, render_task_body)


ORIENT_VERSION = 119
N_TASKS = 18
CONDITIONS = ("BASE", "CAREFUL", "ORIENT")

ORIENT_PREAMBLE = """Orient internally before answering:
center: the decision required by the stated rule.
who/which: which candidate each row describes.
what: the rule conditions and evidence values.
where: the binding between rule variables and candidate fields.
when: the conditions under which none is required.
how: how each candidate passes or fails the rule.
why: choose only what the evidence justifies under the stated rule.
Do not write this analysis."""

CAREFUL_PREAMBLE = ("Before answering, carefully review the rule, compare all "
                    "candidate evidence, check whether any candidate "
                    "qualifies, and check whether none is required. Do not "
                    "write your analysis.")


def task_seed(version, index):
    return int.from_bytes(hashlib.sha256(
        f"orient-v{version}-task:{index}".encode()).digest()[:8], "little")


def build_orient_tasks(n_tasks=N_TASKS, version=ORIENT_VERSION):
    """Build the fixed paired task battery. Seed selection occurred before
    execution using construction metadata only: balanced rules/families,
    low magnitude-truth agreement, substantial abstention fraction."""
    tasks = []
    for i in range(n_tasks):
        fam = FAMILIES[i % len(FAMILIES)]
        rng = random.Random(task_seed(version, i))
        task = make_task(rng, fam)
        task["task_index"] = i
        tasks.append(task)
    return tasks


def render_careful_task(task):
    """Generic carefulness control: semantically close reminders without the
    center/interrogative/why topology."""
    return (CAREFUL_PREAMBLE + "\n\n" + render_task_body(task) +
            "\n\nReply with ONLY the candidate letter, or the word none.")


def render_orient_task(task):
    """Topology-inspired orientation scaffold followed by the identical task
    and output contract."""
    return (ORIENT_PREAMBLE + "\n\n" + render_task_body(task) +
            "\n\nReply with ONLY the candidate letter, or the word none.")


def render_condition(task, condition):
    if condition == "BASE":
        return render_task(task)
    if condition == "CAREFUL":
        return render_careful_task(task)
    if condition == "ORIENT":
        return render_orient_task(task)
    raise ValueError(condition)


def _rate(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def summarize_orient_cells(cells, conditions=CONDITIONS):
    """Compute preregistered orientation/salience metrics.

    cells: iterable of dicts with truth, magnitude_answer, and conditions
    mapping condition -> record containing answer, answer_correct and
    format_valid. Invalid output counts as incorrect but is reported.
    """
    cells = list(cells)
    out = {}
    for cond in conditions:
        rows = []
        for c in cells:
            r = c.get("conditions", {}).get(cond)
            if r and "answer_correct" in r:
                rows.append((c, r))
        conflict = [(c, r) for c, r in rows
                    if c["magnitude_answer"] != c["truth"]]
        answer_present = [(c, r) for c, r in rows if c["truth"] != "none"]
        none_truth = [(c, r) for c, r in rows if c["truth"] == "none"]
        s = {
            "n": len(rows),
            "format_valid_rate": _rate(bool(r.get("format_valid"))
                                       for c, r in rows),
            "accuracy": _rate(bool(r["answer_correct"]) for c, r in rows),
            "selection_accuracy_answer_present": _rate(
                bool(r["answer_correct"]) for c, r in answer_present),
            "abstention_accuracy_none": _rate(
                bool(r["answer_correct"]) for c, r in none_truth),
            "none_response_rate_answer_present": _rate(
                r.get("answer") == "none" for c, r in answer_present),
            "none_response_rate_none_truth": _rate(
                r.get("answer") == "none" for c, r in none_truth),
            "magnitude_capture_rate_conflict": _rate(
                r.get("answer") == c["magnitude_answer"] for c, r in conflict),
            "n_answer_present": len(answer_present),
            "n_none_truth": len(none_truth),
            "n_magnitude_conflict": len(conflict),
        }
        out[cond] = s

    comparisons = {}
    for a in conditions:
        for b in conditions:
            if a >= b:
                continue
            both = []
            for c in cells:
                ra = c.get("conditions", {}).get(a)
                rb = c.get("conditions", {}).get(b)
                if ra and rb and "answer_correct" in ra and "answer_correct" in rb:
                    both.append((bool(ra["answer_correct"]),
                                 bool(rb["answer_correct"])))
            a_only = sum(1 for x, y in both if x and not y)
            b_only = sum(1 for x, y in both if y and not x)
            comparisons[f"{a}_vs_{b}"] = {
                "n": len(both), f"{a}_only_correct": a_only,
                f"{b}_only_correct": b_only,
                "accuracy_delta": ((out[a]["accuracy"] or 0) -
                                   (out[b]["accuracy"] or 0)),
            }
    out["paired_comparisons"] = comparisons
    return out
