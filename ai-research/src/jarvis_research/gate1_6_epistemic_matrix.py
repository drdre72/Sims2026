"""HZN-EPI-02 — hypothesis-observation matrix decomposition.

Separates predictive grounding from comparative query selection. A compact neural
artifact estimates hypothesis consistency and per-query predictions; deterministic
computation can then select a discriminating query. Inference-only grading.
"""
import re
from jarvis_research.gate1_6_epistemic import (
    build_epistemic_tasks, eval_rule, parse_choice, render_epistemic_task)

MATRIX_CONDITIONS = ("BASE", "XTRACT", "ORACLEPRED")


def reference_matrix(task):
    fit = []
    for r in task["rules"]:
        fit.append(all(eval_rule(r, o["x"], o["y"]) == o["outcome"]
                       for o in task["observations"]))
    rows = []
    for q in task["queries"]:
        rows.append({"label": q["label"],
                     "pred": [eval_rule(r, q["x"], q["y"]) for r in task["rules"]]})
    return {"fit": fit, "rows": rows}


def yn(v):
    return "yes" if v else "no"


def render_reference_matrix(task, choice=None):
    ref = reference_matrix(task)
    lines = [f"fit|H1={yn(ref['fit'][0])}|H2={yn(ref['fit'][1])}"]
    for row in ref["rows"]:
        lines.append(f"{row['label']}|H1={yn(row['pred'][0])}|H2={yn(row['pred'][1])}")
    lines.append("choice=" + (choice if choice is not None else task["truth"]))
    return "\n".join(lines)


def deterministic_choice(fit, rows):
    valid = [i for i, x in enumerate(fit) if x]
    if len(valid) != 2:
        return "no_query"
    for row in rows:
        if row["pred"][0] != row["pred"][1]:
            return row["label"]
    return "no_query"


def parse_matrix_artifact(response, task):
    labels = [q["label"] for q in task["queries"]]
    lines = [ln.strip() for ln in str(response).splitlines() if ln.strip()]
    if len(lines) != len(labels) + 2:
        return {"parse_status": "invalid_format"}
    def bits(line, prefix):
        m = re.fullmatch(re.escape(prefix) + r"\|H1=(yes|no)\|H2=(yes|no)", line)
        return None if not m else [m.group(1) == "yes", m.group(2) == "yes"]
    fit = bits(lines[0], "fit")
    if fit is None:
        return {"parse_status": "invalid_format"}
    rows = []
    for label, line in zip(labels, lines[1:-1]):
        pred = bits(line, label)
        if pred is None:
            return {"parse_status": "invalid_format"}
        rows.append({"label": label, "pred": pred})
    m = re.fullmatch(r"choice=(Q[1-5]|no_query)", lines[-1])
    if not m:
        return {"parse_status": "invalid_answer", "fit": fit, "rows": rows}
    return {"parse_status": "valid", "fit": fit, "rows": rows,
            "choice": m.group(1)}


def grade_matrix_artifact(response, task):
    parsed = parse_matrix_artifact(response, task)
    truth = task["truth"]
    out = dict(parsed)
    out["expected"] = truth
    if parsed["parse_status"] != "valid":
        out.update({"format_valid": False, "answer_correct": False,
                    "fit_accuracy": 0.0, "prediction_row_accuracy": 0.0,
                    "matrix_exact": False, "delegated_answer": None,
                    "delegated_correct": False})
        return out
    ref = reference_matrix(task)
    fit_flags = [a == b for a, b in zip(parsed["fit"], ref["fit"])]
    row_flags = [a["pred"] == b["pred"] for a, b in zip(parsed["rows"], ref["rows"])]
    delegated = deterministic_choice(parsed["fit"], parsed["rows"])
    out.update({
        "format_valid": True,
        "answer_correct": parsed["choice"] == truth,
        "fit_accuracy": sum(fit_flags) / 2,
        "fit_exact": all(fit_flags),
        "prediction_row_accuracy": sum(row_flags) / len(row_flags),
        "prediction_rows_exact": all(row_flags),
        "matrix_exact": all(fit_flags) and all(row_flags),
        "delegated_answer": delegated,
        "delegated_correct": delegated == truth,
        "reference_matrix_exact": deterministic_choice(ref["fit"], ref["rows"]) == truth,
    })
    return out


def render_matrix_task(task):
    n = len(task["queries"])
    base = render_epistemic_task(task, "NOCOST")
    # Replace the final direct-answer instruction with the artifact contract.
    prefix = base.rsplit("Reply with ONLY", 1)[0]
    return prefix + f"""First produce the hypothesis-observation matrix, then choose.
Output exactly {n + 2} lines:
line 1: fit|H1=<yes/no>|H2=<yes/no> — whether each rule is consistent with every observed outcome
lines 2-{n + 1}: <query label>|H1=<yes/no>|H2=<yes/no> — whether each rule predicts pass for that query
line {n + 2}: choice=<Q1-Q5 or no_query>

Use yes when the rule predicts pass. Do not add a header or explanation."""


def render_oraclepred_task(task):
    ref = reference_matrix(task)
    matrix = "\n".join(render_reference_matrix(task, choice=task["truth"]).splitlines()[:-1])
    return render_epistemic_task(task, "NOCOST").rsplit("Reply with ONLY", 1)[0] + f"""The correct hypothesis-observation matrix is:
{matrix}

Using only this matrix and the observed outcomes, choose the query that distinguishes the rules. If no query is needed or none distinguishes them, choose no_query.
Reply with ONLY choice=<Q1-Q5 or no_query>."""


def render_condition(task, condition):
    if condition == "BASE":
        return render_epistemic_task(task, "NOCOST")
    if condition == "XTRACT":
        return render_matrix_task(task)
    if condition == "ORACLEPRED":
        return render_oraclepred_task(task)
    raise ValueError(condition)


def grade_direct(response, task):
    choice = parse_choice(response, task)
    return {"choice": choice, "answer_correct": choice == task["truth"],
            "format_valid": choice is not None, "expected": task["truth"]}


def _rate(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def summarize_matrix_cells(cells):
    cells = list(cells)
    out = {}
    for cond in MATRIX_CONDITIONS:
        rows = [c["conditions"][cond] for c in cells if cond in c.get("conditions", {})]
        if not rows:
            continue
        fams = {}
        for c in cells:
            if cond not in c.get("conditions", {}): continue
            fams.setdefault(c["family"], []).append(c["conditions"][cond]["answer_correct"])
        out[cond] = {
            "n": len(rows), "accuracy": _rate(r["answer_correct"] for r in rows),
            "format_valid_rate": _rate(r["format_valid"] for r in rows),
            "family_accuracy": {k: _rate(v) for k, v in fams.items()},
        }
    xrows = [(c, c["conditions"]["XTRACT"]) for c in cells if "XTRACT" in c.get("conditions", {})]
    if xrows:
        valid = [r for c, r in xrows if r["format_valid"]]
        out["XTRACT_ARTIFACT"] = {
            "n_valid": len(valid),
            "fit_accuracy": _rate(r["fit_accuracy"] for r in valid),
            "fit_exact_rate": _rate(r["fit_exact"] for r in valid),
            "prediction_row_accuracy": _rate(r["prediction_row_accuracy"] for r in valid),
            "prediction_rows_exact_rate": _rate(r["prediction_rows_exact"] for r in valid),
            "matrix_exact_rate": _rate(r["matrix_exact"] for r in valid),
        }
        out["XTRACT_DELEGATED"] = {
            "n": len(xrows),
            "accuracy": _rate(r["delegated_correct"] for c, r in xrows),
            "family_accuracy": {},
        }
        for fam in sorted({c["family"] for c, r in xrows}):
            z = [r for c, r in xrows if c["family"] == fam]
            out["XTRACT_DELEGATED"]["family_accuracy"][fam] = _rate(r["delegated_correct"] for r in z)
        wrong = [(c, r) for c, r in xrows if not r["answer_correct"]]
        correct = [(c, r) for c, r in xrows if r["answer_correct"]]
        out["XTRACT_DELEGATION_PAIRING"] = {
            "n_neural_wrong": len(wrong),
            "n_neural_correct": len(correct),
            "repair_rate_neural_wrong": _rate(r["delegated_correct"] for c, r in wrong),
            "damage_rate_neural_correct": _rate(not r["delegated_correct"] for c, r in correct),
        }
        out["delegation_delta_accuracy"] = (out["XTRACT_DELEGATED"]["accuracy"] -
                                            out["XTRACT"]["accuracy"])
    out["ORACLE_DELEGATED"] = {
        "accuracy": _rate(c["conditions"]["XTRACT"].get("reference_matrix_exact", False)
                          for c in cells if "XTRACT" in c.get("conditions", {}))
    }
    return out


def build_matrix_tasks():
    return build_epistemic_tasks()
