"""HZN-EPI-03 — minimal comparative primitive probe.

Tests whether the model can identify the unique row where two binary prediction
columns differ, and whether that primitive survives translation into epistemic
query language. Deterministic construction; inference-only grading.
"""
import hashlib, random

COMPARE_VERSION = 1
COMPARE_CONDITIONS = ("RAW", "EPI", "EPIEXPLICIT")
LABELS = "ABCDE"


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(
        ("cmp-" + ":".join(map(str, parts))).encode()).digest()[:8], "little")


def _bits_for(family, target, rng):
    rows = []
    for i in range(5):
        if family == "no_diff":
            diff = False
        elif family == "multiple_diff":
            diff = i in (target, (target + 2) % 5)
        else:
            diff = i == target
        a = rng.randrange(2)
        b = 1 - a if diff else a
        rows.append({"label": LABELS[i], "h1": bool(a), "h2": bool(b),
                     "diff": bool(diff)})
    return rows


def build_comparison_tasks(version=COMPARE_VERSION):
    tasks = []
    # Twelve unique cases: positions 0,1 repeated three times; 2,3,4 twice.
    positions = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
    for rep, pos in enumerate(positions):
        rng = random.Random(_seed(version, "unique_diff", rep))
        rows = _bits_for("unique_diff", pos, rng)
        tasks.append({"task_id": f"c{len(tasks):02d}", "family": "unique_diff",
                      "rows": rows, "truth": LABELS[pos], "target_index": pos})
    for family in ("no_diff", "multiple_diff"):
        for rep in range(6):
            rng = random.Random(_seed(version, family, rep))
            rows = _bits_for(family, rep % 5, rng)
            tasks.append({"task_id": f"c{len(tasks):02d}", "family": family,
                          "rows": rows, "truth": "none", "target_index": None})
    return tasks


def pf(v):
    return "pass" if v else "fail"


def render_compare_task(task, condition):
    if condition == "RAW":
        lines = ["Each row contains two binary values."]
        for r in task["rows"]:
            lines.append(f"row {r['label']}: H1={int(r['h1'])}, H2={int(r['h2'])}")
        lines += ["", "Select the unique row where H1 and H2 are different. If there is no unique differing row, answer none.",
                  "Reply with ONLY answer=<A-E or none>."]
        return "\n".join(lines)
    if condition not in ("EPI", "EPIEXPLICIT"):
        raise ValueError(condition)
    lines = ["Two rules make binary predictions about five possible queries.",
             "Rule 1 and Rule 2 predictions:"]
    for r in task["rows"]:
        lines.append(f"query {r['label']}: Rule 1={pf(r['h1'])}, Rule 2={pf(r['h2'])}")
    if condition == "EPIEXPLICIT":
        lines.append("A query distinguishes the rules exactly when one rule predicts pass and the other predicts fail.")
    lines += ["", "Select the unique query that would distinguish the two rules. If there is no unique distinguishing query, choose no_query.",
              "Reply with ONLY choice=<A-E or no_query>."]
    return "\n".join(lines)


def parse_compare_response(response, condition):
    text = str(response).strip()
    prefix = "answer=" if condition == "RAW" else "choice="
    none = "none" if condition == "RAW" else "no_query"
    if not text.startswith(prefix):
        return None
    value = text.split("=", 1)[1].strip().upper()
    if value in list(LABELS):
        return value
    if value == none.upper():
        return "none"
    return None


def grade_compare_response(response, task, condition):
    ans = parse_compare_response(response, condition)
    truth = task["truth"]
    return {"answer": ans, "expected": truth, "answer_correct": ans == truth,
            "format_valid": ans is not None}


def _rate(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def summarize_compare_cells(cells):
    cells = list(cells)
    out = {}
    for cond in COMPARE_CONDITIONS:
        rows = [(c, c["conditions"][cond]) for c in cells if cond in c.get("conditions", {})]
        if not rows: continue
        fams = {}
        for c, r in rows:
            fams.setdefault(c["family"], []).append(r["answer_correct"])
        unique = [(c, r) for c, r in rows if c["family"] == "unique_diff"]
        out[cond] = {
            "n": len(rows), "accuracy": _rate(r["answer_correct"] for c, r in rows),
            "format_valid_rate": _rate(r["format_valid"] for c, r in rows),
            "family_accuracy": {k: _rate(v) for k, v in fams.items()},
            "unique_truth_capture": _rate(r["answer"] == c["truth"] for c, r in unique),
            "none_rate_unique": _rate(r["answer"] == "none" for c, r in unique),
            "wrong_row_rate_unique": _rate(bool(r["answer"]) and r["answer"] not in ("none", c["truth"]) for c, r in unique),
        }
    return out
