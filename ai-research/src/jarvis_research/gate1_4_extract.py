"""HZN-OOI-03 — extraction-mediated delegation probe.

Simplifies the operational artifact to the part a neural system is actually
needed for in this closed benchmark: binding candidate labels to raw evidence.
Deterministic computation then performs qualification, tie-break comparison,
and exact-tie inhibition. This module is inference-only construction/grading.
"""

import re

from jarvis_research.gate1_4_benchmark import render_task, render_task_body
from jarvis_research.gate1_4_mirror import answer_verdict, normalize_answer
from jarvis_research.gate1_4_operational import (
    OOI_FAMILIES, apply_exact_tie_guard, build_operational_tasks, rule_summary)


EXTRACT_VERSION = 134
CONDITIONS = ("BASE", "XTRACT")


def build_extract_tasks():
    """Fresh operational battery selected by construction metadata only."""
    return build_operational_tasks(version=EXTRACT_VERSION)


def render_extract_task(task):
    """Render the raw evidence-orientation artifact plus final neural answer."""
    n_cand = len(task["candidates"])
    instructions = f"""Produce the compact evidence-orientation artifact before the final answer.
The output must contain exactly {n_cand + 1} lines:
- lines 1-{n_cand}: one row for each candidate, in the order shown above, formatted as label|ar|lc
- line {n_cand + 1}: answer=<candidate letter or none>

Copy ar and lc exactly as shown. If observations are missing, use na for both values. Do not add a header, explanation, pass/fail fields, or summary."""
    return render_task_body(task) + "\n\n" + instructions


def render_extract_condition(task, condition):
    if condition == "BASE":
        return render_task(task)
    if condition == "XTRACT":
        return render_extract_task(task)
    raise ValueError(condition)


def _parse_number(text):
    if text == "na":
        return None
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        raise ValueError(text)
    return float(text)


def parse_extract_artifact(response, candidate_labels):
    """Strictly parse compact rows plus final answer."""
    candidate_labels = list(candidate_labels)
    lines = [ln.strip() for ln in str(response).splitlines() if ln.strip()]
    if len(lines) != len(candidate_labels) + 1:
        return {"parse_status": "invalid_format"}
    rows = []
    try:
        for expected_label, line in zip(candidate_labels, lines[:-1]):
            parts = line.split("|")
            if len(parts) != 3 or parts[0] != expected_label:
                return {"parse_status": "invalid_format"}
            ar = _parse_number(parts[1])
            lc = _parse_number(parts[2])
            if (ar is None) != (lc is None):
                return {"parse_status": "invalid_format"}
            rows.append({"label": parts[0], "ar": ar, "lc": lc,
                         "obs": ar is not None})
    except (ValueError, TypeError):
        return {"parse_status": "invalid_format"}

    m = re.fullmatch(r"answer=([A-Za-z]+)", lines[-1])
    if not m:
        return {"parse_status": "invalid_format"}
    answer = normalize_answer(m.group(1))
    if answer not in candidate_labels + ["none"]:
        return {"parse_status": "invalid_answer", "rows": rows,
                "answer": answer}
    return {"parse_status": "valid", "rows": rows, "answer": answer}


def _num_eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(round(a, 2) - round(b, 2)) < 1e-9


def extracted_candidates(parsed):
    return [{"label": r["label"], "ar": r["ar"], "lc": r["lc"],
             "obs_available": r["obs"]} for r in parsed.get("rows", [])]


def grade_extract_artifact(response, task):
    """Grade neural final choice, raw extraction, and deterministic delegation."""
    labels = [c["label"] for c in task["candidates"]]
    parsed = parse_extract_artifact(response, labels)
    truth = task["truth"]["answer"]
    valid = labels + ["none"]
    out = dict(parsed)
    out["expected"] = truth
    if parsed["parse_status"] != "valid":
        out.update({
            "answer_verdict": parsed["parse_status"],
            "answer_correct": False,
            "format_valid": False,
            "raw_extraction_exact": False,
            "raw_row_accuracy": 0.0,
            "delegated_answer": None,
            "delegated_correct": False,
        })
        return out

    raw_flags = []
    rows_by_label = {r["label"]: r for r in parsed["rows"]}
    for source in task["candidates"]:
        row = rows_by_label[source["label"]]
        raw_flags.append(row["obs"] == source["obs_available"] and
                         (not source["obs_available"] or
                          (_num_eq(row["ar"], source["ar"]) and
                           _num_eq(row["lc"], source["lc"]))))
    delegated = rule_summary(task["rule_key"], extracted_candidates(parsed))
    verdict = answer_verdict(parsed["answer"], truth, valid)
    out.update({
        "answer_verdict": verdict,
        "answer_correct": verdict == "correct",
        "format_valid": True,
        "raw_extraction_exact": all(raw_flags),
        "raw_row_accuracy": sum(raw_flags) / len(raw_flags),
        "raw_flags": raw_flags,
        "delegated_answer": delegated["answer"],
        "delegated_correct": delegated["answer"] == truth,
        "delegated_summary": delegated,
    })
    return out


def render_reference_extract(task):
    lines = []
    for c in task["candidates"]:
        if c["obs_available"]:
            lines.append(f"{c['label']}|{c['ar']:.2f}|{c['lc']:.2f}")
        else:
            lines.append(f"{c['label']}|na|na")
    lines.append(f"answer={task['truth']['answer']}")
    return "\n".join(lines)


def _rate(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def _family_rows(rows, family):
    return [(c, r) for c, r in rows if c["family"] == family]


def summarize_extract_cells(cells, conditions=CONDITIONS):
    cells = list(cells)
    out = {}
    for cond in conditions:
        rows = []
        for c in cells:
            r = c.get("conditions", {}).get(cond)
            if r and "answer_correct" in r:
                rows.append((c, r))
        answer_present = [(c, r) for c, r in rows if c["truth"] != "none"]
        none_truth = [(c, r) for c, r in rows if c["truth"] == "none"]
        exact_rows = _family_rows(rows, "exact_tie")
        critical_selections = []
        first_critical = []
        for c, r in exact_rows:
            ans = r.get("answer")
            if ans in c.get("critical_labels", []):
                critical_selections.append((c, r))
                item = c["label_to_item"].get(ans)
                first_critical.append(item == c["first_critical_item"])
        out[cond] = {
            "n": len(rows),
            "format_valid_rate": _rate(bool(r.get("format_valid"))
                                       for c, r in rows),
            "accuracy": _rate(bool(r["answer_correct"]) for c, r in rows),
            "selection_accuracy_answer_present": _rate(
                bool(r["answer_correct"]) for c, r in answer_present),
            "abstention_accuracy_none": _rate(
                bool(r["answer_correct"]) for c, r in none_truth),
            "overinhibition_rate_answer_present": _rate(
                r.get("answer") == "none" for c, r in answer_present),
            "family_accuracy": {
                fam: _rate(bool(r["answer_correct"])
                           for c, r in _family_rows(rows, fam))
                for fam in OOI_FAMILIES
            },
            "exact_tie_critical_selection_rate": _rate(
                r.get("answer") in c.get("critical_labels", [])
                for c, r in exact_rows),
            "exact_tie_first_critical_capture_rate": _rate(first_critical),
            "n_answer_present": len(answer_present),
            "n_none_truth": len(none_truth),
        }

    xtract_rows = []
    for c in cells:
        r = c.get("conditions", {}).get("XTRACT")
        if r and "answer_correct" in r:
            xtract_rows.append((c, r))
    if xtract_rows:
        answer_present = [(c, r) for c, r in xtract_rows if c["truth"] != "none"]
        none_truth = [(c, r) for c, r in xtract_rows if c["truth"] == "none"]
        out["XTRACT_DELEGATED"] = {
            "n": len(xtract_rows),
            "accuracy": _rate(bool(r["delegated_correct"]) for c, r in xtract_rows),
            "selection_accuracy_answer_present": _rate(
                bool(r["delegated_correct"]) for c, r in answer_present),
            "abstention_accuracy_none": _rate(
                bool(r["delegated_correct"]) for c, r in none_truth),
            "overinhibition_rate_answer_present": _rate(
                r.get("delegated_answer") == "none" for c, r in answer_present),
            "family_accuracy": {
                fam: _rate(bool(r["delegated_correct"])
                           for c, r in _family_rows(xtract_rows, fam))
                for fam in OOI_FAMILIES
            },
        }
        valid_rows = [(c, r) for c, r in xtract_rows if r.get("format_valid")]
        out["XTRACT_ARTIFACT"] = {
            "n_valid": len(valid_rows),
            "raw_extraction_exact_rate": _rate(
                bool(r["raw_extraction_exact"]) for c, r in valid_rows),
            "raw_row_accuracy": _rate(r["raw_row_accuracy"] for c, r in valid_rows),
        }

        def pair_metrics(rows):
            wrong = [(c, r) for c, r in rows if not r["answer_correct"]]
            right = [(c, r) for c, r in rows if r["answer_correct"]]
            return {
                "n_neural_wrong": len(wrong),
                "n_neural_correct": len(right),
                "repair_rate_neural_wrong": _rate(
                    bool(r["delegated_correct"]) for c, r in wrong),
                "damage_rate_neural_correct": _rate(
                    not r["delegated_correct"] for c, r in right),
            }

        out["XTRACT_DELEGATION_PAIRING"] = pair_metrics(xtract_rows)
        out["XTRACT_DELEGATION_PAIRING"]["exact_tie"] = pair_metrics(
            _family_rows(xtract_rows, "exact_tie"))
        neural_acc = out["XTRACT"]["accuracy"]
        del_acc = out["XTRACT_DELEGATED"]["accuracy"]
        if neural_acc is not None and del_acc is not None:
            out["delegation_delta_accuracy"] = del_acc - neural_acc

    base_rows = []
    for c in cells:
        r = c.get("conditions", {}).get("BASE")
        if r and "answer" in r:
            guarded = apply_exact_tie_guard(
                r["answer"], c["rule_key"], c["candidates"])
            base_rows.append((c, guarded))
    if base_rows:
        out["BASE_TIE_GUARD"] = {
            "n": len(base_rows),
            "accuracy": _rate(guarded == c["truth"] for c, guarded in base_rows),
            "exact_tie_accuracy": _rate(
                guarded == c["truth"] for c, guarded in base_rows
                if c["family"] == "exact_tie"),
        }

    both = []
    for c in cells:
        ra = c.get("conditions", {}).get("BASE")
        rb = c.get("conditions", {}).get("XTRACT")
        if ra and rb and "answer_correct" in ra and "answer_correct" in rb:
            both.append((bool(ra["answer_correct"]), bool(rb["answer_correct"])))
    out["paired_comparisons"] = {"BASE_vs_XTRACT": {
        "n": len(both),
        "BASE_only_correct": sum(1 for x, y in both if x and not y),
        "XTRACT_only_correct": sum(1 for x, y in both if y and not x),
        "accuracy_delta": ((out["BASE"]["accuracy"] or 0) -
                           (out["XTRACT"]["accuracy"] or 0)),
    }}
    return out
