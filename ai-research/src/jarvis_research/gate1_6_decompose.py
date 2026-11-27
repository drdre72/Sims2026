"""HZN-EPI-04 — comparative decomposition and comparator invocation.

The probe separates isolated comparison, vector enumeration, cardinality,
selection, routing, typed invocation, result consumption, and epistemic utility.
All tasks and oracle references are deterministic; model evaluation is inference-only.
"""
import hashlib, random, re

DECOMPOSE_VERSION = 1
LETTERS = "ABCDE"
ALT_LABELS = "KLMNO"
MATRIX_CONDITIONS = ("VECTOR", "COUNT", "SELECT")
EPI_CONDITION = "EPIUTIL"


def seedi(*parts):
    return int.from_bytes(hashlib.sha256(("epi4-" + ":".join(map(str, parts))).encode()).digest()[:8], "little")


def _variant(i):
    # Stratified, not factorial: row permutation, column reversal, label identity,
    # output polarity and surface format rotate independently across cases.
    return {
        "labelset": ALT_LABELS if i % 2 else LETTERS,
        "colrev": bool((i // 2) % 2),
        "permute": bool((i // 4) % 2),
        "yesno": i % 3 == 0,
        "csv": i % 4 == 0,
    }


def _matrix_rows(family, target, rng, labels):
    rows = []
    for i, label in enumerate(labels):
        if family == "no_diff": diff = False
        elif family == "multiple_diff": diff = i in (target, (target + 2) % 5)
        else: diff = i == target
        a = rng.randrange(2)
        b = 1 - a if diff else a
        rows.append({"label": label, "h1": bool(a), "h2": bool(b), "diff": bool(diff), "canonical_index": i})
    return rows


def oracle_flags(case):
    return {r["label"]: bool(r["diff"]) for r in case["rows"]}


def oracle_count(case):
    return sum(r["diff"] for r in case["rows"])


def oracle_unique(case):
    ds = [r["label"] for r in case["rows"] if r["diff"]]
    return ds[0] if len(ds) == 1 else "none"


def build_matrix_cases(version=DECOMPOSE_VERSION):
    cases = []
    positions = [0, 1, 2, 3, 4, 0, 1, 2]
    for rep, pos in enumerate(positions):
        i = len(cases); rng = random.Random(seedi(version, "matrix", "unique_diff", rep))
        v = _variant(i); rows = _matrix_rows("unique_diff", pos, rng, v["labelset"])
        truth = rows[pos]["label"]
        if v["permute"]:
            # Preserve a balanced target-display-position distribution while
            # randomizing non-target row order. Without this constraint, row
            # permutation would confound position effects with family counts.
            for _ in range(100):
                rng.shuffle(rows)
                if next(irow for irow, r in enumerate(rows) if r["label"] == truth) == pos:
                    break
        display_pos = next(irow for irow, r in enumerate(rows) if r["label"] == truth)
        cases.append({"case_id": f"m{i:02d}", "family": "unique_diff", "rows": rows,
                      "truth": truth, "variant": v, "target_label": truth,
                      "target_display_index": display_pos})
    for family, n in (("no_diff", 5), ("multiple_diff", 5)):
        for rep in range(n):
            i = len(cases); rng = random.Random(seedi(version, "matrix", family, rep))
            v = _variant(i); rows = _matrix_rows(family, rep % 5, rng, v["labelset"])
            if v["permute"]: rng.shuffle(rows)
            cases.append({"case_id": f"m{i:02d}", "family": family, "rows": rows,
                          "truth": "none", "variant": v, "target_label": None,
                          "target_display_index": None})
    return cases


def build_single_tasks(version=DECOMPOSE_VERSION):
    tasks = []
    patterns = [(0, 0), (0, 1), (1, 1), (1, 0)]
    for i in range(12):
        a, b = patterns[i % 4]
        tasks.append({"task_id": f"s{i:02d}", "h1": bool(a), "h2": bool(b),
                      "truth": a != b, "variant": _variant(i)})
    return tasks


def yn(v):
    return "yes" if v else "no"


def de(v):
    return "diff" if v else "equal"


def render_matrix(case, oracle=False):
    rows = case["rows"]; v = case["variant"]
    if v["csv"]:
        cols = ("label,H2,H1" if v["colrev"] else "label,H1,H2")
        lines = [cols]
        for r in rows:
            vals = (r["label"], int(r["h2"]), int(r["h1"])) if v["colrev"] else (r["label"], int(r["h1"]), int(r["h2"]))
            lines.append(",".join(map(str, vals)))
    else:
        lines = []
        for r in rows:
            pair = f"H2={int(r['h2'])}, H1={int(r['h1'])}" if v["colrev"] else f"H1={int(r['h1'])}, H2={int(r['h2'])}"
            lines.append(f"row {r['label']}: {pair}")
    if oracle:
        fl = ",".join(f"{r['label']}={de(r['diff'])}" for r in rows)
        lines.append("oracle flags: " + fl)
    return "\n".join(lines)


def render_single(task):
    v = task["variant"]
    pair = f"H2={int(task['h2'])}, H1={int(task['h1'])}" if v["colrev"] else f"H1={int(task['h1'])}, H2={int(task['h2'])}"
    if v["yesno"]:
        return f"Binary pair: {pair}\nAre the two values different? Reply with ONLY answer=<yes or no>."
    return f"Binary pair: {pair}\nCompare the two values. Reply with ONLY relation=<same or different>."


def parse_single(response, task):
    text = str(response).strip().lower()
    yesno = task["variant"]["yesno"]
    prefix = "answer=" if yesno else "relation="
    if not text.startswith(prefix): return None
    val = text.split("=", 1)[1].strip()
    if yesno:
        if val == "yes": return True
        if val == "no": return False
    else:
        if val == "different": return True
        if val == "same": return False
    return None


def grade_single(response, task):
    ans = parse_single(response, task)
    return {"answer": ans, "expected": task["truth"], "answer_correct": ans == task["truth"], "format_valid": ans is not None}


def render_vector(case):
    labels = [r["label"] for r in case["rows"]]
    word = "yes/no" if case["variant"]["yesno"] else "diff/equal"
    return (render_matrix(case) + "\n\nFor each row independently, decide whether H1 and H2 differ. "
            f"Use {word} flags. Reply with ONLY flags=" + ",".join(f"{x}=<{word.split('/')[0]} or {word.split('/')[1]}>" for x in labels) + ".")


def parse_vector(response, case):
    text = str(response).strip()
    if not text.startswith("flags="): return None
    body = text[len("flags="):].strip()
    labels = [r["label"] for r in case["rows"]]
    yesno = case["variant"]["yesno"]
    true_words = {"yes", "diff"}; false_words = {"no", "equal"}
    parts = [p.strip() for p in body.split(",") if p.strip()]
    vals = {}
    for p in parts:
        if "=" not in p: return None
        k, v = [x.strip() for x in p.split("=", 1)]
        if k not in labels or v.lower() not in true_words | false_words: return None
        vals[k] = v.lower() in true_words
    if set(vals) != set(labels): return None
    return vals


def grade_vector(response, case):
    vals = parse_vector(response, case)
    truth = oracle_flags(case)
    if vals is None:
        return {"flags": None, "answer_correct": False, "format_valid": False,
                "row_accuracy": 0.0, "detected_diff": [], "false_positive": [], "false_negative": list(k for k, v in truth.items() if v)}
    fp = [k for k in truth if vals[k] and not truth[k]]
    fn = [k for k in truth if not vals[k] and truth[k]]
    return {"flags": vals, "answer_correct": vals == truth, "format_valid": True,
            "row_accuracy": sum(vals[k] == truth[k] for k in truth) / len(truth),
            "detected_diff": [k for k in truth if vals[k]], "false_positive": fp, "false_negative": fn}


def render_count(case):
    return render_matrix(case, oracle=True) + "\n\nUsing only the oracle flags, count how many rows are marked diff. Reply with ONLY count=<0-5>."


def parse_count(response):
    m = re.fullmatch(r"count=([0-5])", str(response).strip())
    return int(m.group(1)) if m else None


def grade_count(response, case):
    ans = parse_count(response); truth = oracle_count(case)
    return {"answer": ans, "expected": truth, "answer_correct": ans == truth, "format_valid": ans is not None}


def render_select(case):
    labels = "/".join(r["label"] for r in case["rows"])
    return render_matrix(case, oracle=True) + ("\n\nUsing only the oracle flags, select the unique row marked diff. "
        f"If there is not exactly one diff row, answer none. Reply with ONLY select=<{labels} or none>.")


def parse_select(response, case):
    text = str(response).strip()
    if not text.startswith("select="): return None
    val = text.split("=", 1)[1].strip()
    labels = [r["label"] for r in case["rows"]]
    if val in labels or val == "none": return val
    return None


def grade_select(response, case):
    ans = parse_select(response, case); truth = oracle_unique(case)
    return {"answer": ans, "expected": truth, "answer_correct": ans == truth, "format_valid": ans is not None}

# ---------- comparator routing ----------

def build_route_tasks(matrix_cases):
    # Six applicable binary-matrix cases spanning all three matrix families.
    idxs = [0, 3, 5, 8, 10, 15]
    tasks = []
    for i, ix in enumerate(idxs):
        tasks.append({"task_id": f"r{i:02d}", "kind": "applicable", "case": matrix_cases[ix], "truth": "comparator"})
    scenarios = [
        "Decide whether two short stories have a similar emotional tone.",
        "Judge which of two job candidates is likely to be a better long-term colleague.",
        "Determine whether a blurred photograph shows a familiar person.",
        "Choose the more elegant proof of a mathematical theorem.",
        "Assess whether a piece of music sounds more energetic than another.",
        "Decide whether a natural-language rumor is socially credible.",
    ]
    for j, text in enumerate(scenarios):
        tasks.append({"task_id": f"r{len(tasks):02d}", "kind": "nonapplicable", "scenario": text, "truth": "manual"})
    return tasks


def render_route_task(task):
    header = ("A binary row comparator is available. It applies only when the problem consists of explicit finite rows, "
              "each with exactly two binary values. It does not judge vague, analog, visual, social or aesthetic material.\n")
    if task["kind"] == "applicable":
        body = render_matrix(task["case"])
    else:
        body = "Problem: " + task["scenario"]
    return header + body + "\n\nChoose how the problem should be handled. Reply with ONLY route=<comparator or manual>."


def parse_route(response):
    text = str(response).strip().lower()
    if text in ("route=comparator", "route=manual"):
        return text.split("=", 1)[1]
    return None


def grade_route(response, task):
    ans = parse_route(response)
    return {"answer": ans, "expected": task["truth"], "answer_correct": ans == task["truth"], "format_valid": ans is not None}


# ---------- typed invocation and result consumption ----------

def render_delegate(case):
    labels = "/".join(r["label"] for r in case["rows"])
    example = ";".join(f"{r['label']}:0/0" for r in case["rows"])
    return (render_matrix(case) + "\n\nYou may answer directly or invoke the typed binary row comparator.\n"
            "To answer directly, reply EXACTLY as:\n"
            f"action=answer\nanswer=<{labels} or none>\n"
            "To invoke the comparator, reply EXACTLY as two lines:\n"
            "action=compare\n"
            f"rows={example}\n"
            "Replace every 0/0 with that row's H1/H2 values. Copy every row exactly once.")


def parse_delegate_action(response, case):
    lines = [x.strip() for x in str(response).strip().splitlines() if x.strip()]
    labels = [r["label"] for r in case["rows"]]
    if len(lines) == 2 and lines[0] == "action=answer" and lines[1].startswith("answer="):
        val = lines[1].split("=", 1)[1].strip()
        if val in labels or val == "none":
            return {"action": "answer", "answer": val, "format_valid": True}
    if len(lines) == 2 and lines[0] == "action=compare" and lines[1].startswith("rows="):
        body = lines[1][len("rows="):]
        vals = {}
        ok = True
        for part in body.split(";"):
            part = part.strip()
            m = re.fullmatch(r"([A-Z]):([01])/([01])", part)
            if not m:
                ok = False; break
            lab, a, b = m.groups()
            if lab in vals or lab not in labels: ok = False; break
            vals[lab] = (a == "1", b == "1")
        if ok and set(vals) == set(labels):
            return {"action": "compare", "rows": vals, "format_valid": True}
    return {"action": None, "format_valid": False}


def operand_exact(call_rows, case):
    if not call_rows: return False
    truth = {r["label"]: (r["h1"], r["h2"]) for r in case["rows"]}
    return call_rows == truth


def comparator_result(call_rows):
    diffs = sorted(k for k, (a, b) in call_rows.items() if a != b)
    unique = diffs[0] if len(diffs) == 1 else "none"
    return {"differences": diffs, "count": len(diffs), "unique": unique}


def render_delegate_result(result):
    ds = ",".join(result["differences"]) if result["differences"] else "none"
    return ("Comparator result:\n"
            f"differences={ds}\ncount={result['count']}\nunique={result['unique']}\n\n"
            "Use the comparator result. Reply with ONLY final=<row label or none>.")


def parse_final(response, case):
    text = str(response).strip()
    if not text.startswith("final="): return None
    val = text.split("=", 1)[1].strip()
    labels = [r["label"] for r in case["rows"]]
    return val if val in labels or val == "none" else None


def grade_final(response, case):
    ans = parse_final(response, case); truth = oracle_unique(case)
    return {"final": ans, "expected": truth, "answer_correct": ans == truth, "format_valid": ans is not None}


def automatic_reference(case):
    return {"answer": oracle_unique(case), "answer_correct": True, "mechanism": "typed automatic primitive dispatch"}

# ---------- epistemic consistency, discrimination, and utility ----------

def _epi_rows(diff_positions, costs, rng):
    rows = []
    for i, label in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        a = rng.randrange(2)
        b = 1 - a if i in diff_positions else a
        rows.append({"label": label, "cost": costs[i], "h1": bool(a), "h2": bool(b), "diff": a != b})
    return rows


def build_epi_tasks(version=DECOMPOSE_VERSION):
    specs = [
        ("need_unique", (True, True), [1], [1, 1, 1, 1]),
        ("need_multiple_equal", (True, True), [0, 2], [1, 1, 1, 1]),
        ("need_multiple_utility", (True, True), [0, 2], [1, 4, 3, 1]),
        ("resolved_still_diff", (True, False), [1], [1, 1, 1, 1]),
        ("consistent_nodisc", (True, True), [], [1, 1, 1, 1]),
        ("resolved_nodisc", (True, False), [], [1, 1, 1, 1]),
    ]
    tasks = []
    for fam, consistent, diffs, base_costs in specs:
        for rep in range(2):
            rng = random.Random(seedi(version, "epi", fam, rep))
            # Rotate differing positions across replicas while preserving intended multiplicity.
            shifted = sorted((x + rep) % 4 for x in diffs)
            costs = list(base_costs)
            rows = _epi_rows(shifted, costs, rng)
            task = {"task_id": f"e{len(tasks):02d}", "family": fam, "consistent": consistent,
                    "rows": rows}
            task["acceptable"] = epistemic_acceptable(task)
            tasks.append(task)
    return tasks


def epistemic_acceptable(task):
    # If observations have already eliminated one hypothesis, further prediction
    # differences do not create useful epistemic uncertainty in this closed task.
    if sum(task["consistent"]) == 1:
        return ["no_query"]
    diffs = [r for r in task["rows"] if r["h1"] != r["h2"]]
    if not diffs: return ["no_query"]
    cheapest = min(r["cost"] for r in diffs)
    return sorted(r["label"] for r in diffs if r["cost"] == cheapest)


def render_epi_task(task):
    c1, c2 = task["consistent"]
    lines = ["Two hypotheses are candidate explanations.",
             f"observation consistency: H1={yn(c1)}, H2={yn(c2)}",
             "Candidate observations:"]
    for r in task["rows"]:
        lines.append(f"{r['label']}: cost={r['cost']}, H1 predicts {pf(r['h1'])}, H2 predicts {pf(r['h2'])}")
    lines += ["", "Choose the lowest-cost observation that can distinguish hypotheses that remain consistent.",
              "If uncertainty is already resolved or no observation can distinguish them, choose no_query.",
              "Reply with ONLY choice=<Q1-Q4 or no_query>."]
    return "\n".join(lines)


def pf(v):
    return "pass" if v else "fail"


def parse_choice(response):
    text = str(response).strip()
    if not text.startswith("choice="): return None
    val = text.split("=", 1)[1].strip()
    return val if val in ("Q1", "Q2", "Q3", "Q4", "no_query") else None


def grade_epi(response, task):
    ans = parse_choice(response)
    ok = ans in task["acceptable"] if ans is not None else False
    return {"answer": ans, "acceptable": list(task["acceptable"]), "answer_correct": ok,
            "format_valid": ans is not None,
            "singleton_truth": task["acceptable"][0] if len(task["acceptable"]) == 1 else None}


# ---------- compact summary ----------

def rate(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def summarize_epi4(payload):
    out = {}
    singles = payload.get("singles", [])
    if singles:
        out["SINGLE"] = {"n": len(singles), "accuracy": rate(r["answer_correct"] for r in singles),
                         "format_valid_rate": rate(r["format_valid"] for r in singles)}
    for cond in MATRIX_CONDITIONS:
        rows = payload.get(cond.lower(), [])
        if not rows: continue
        fams = {}
        for r in rows: fams.setdefault(r["family"], []).append(r["answer_correct"])
        out[cond] = {"n": len(rows), "accuracy": rate(r["answer_correct"] for r in rows),
                     "format_valid_rate": rate(r["format_valid"] for r in rows),
                     "family_accuracy": {k: rate(v) for k, v in fams.items()}}
        if cond == "VECTOR":
            valid = [r for r in rows if r["format_valid"]]
            out[cond]["row_accuracy_valid"] = rate(r["row_accuracy"] for r in valid)
            out[cond]["false_positive_count"] = sum(len(r["false_positive"]) for r in valid)
            out[cond]["false_negative_count"] = sum(len(r["false_negative"]) for r in valid)
    routes = payload.get("routes", [])
    if routes:
        kinds = {}
        for r in routes: kinds.setdefault(r["kind"], []).append(r["answer_correct"])
        out["ROUTE"] = {"n": len(routes), "accuracy": rate(r["answer_correct"] for r in routes),
                        "format_valid_rate": rate(r["format_valid"] for r in routes),
                        "kind_accuracy": {k: rate(v) for k, v in kinds.items()}}
    dels = payload.get("delegates", [])
    if dels:
        invoked = [r for r in dels if r.get("action") == "compare"]
        out["DELEGATE"] = {"n": len(dels),
            "first_action_valid_rate": rate(r["first_format_valid"] for r in dels),
            "invoke_rate": rate(r.get("action") == "compare" for r in dels),
            "operand_exact_rate_invoked": rate(r.get("operand_exact") for r in invoked),
            "final_accuracy": rate(r.get("final_correct", False) for r in dels),
            "family_final_accuracy": {fam: rate(r.get("final_correct", False) for r in dels if r["family"] == fam)
                                      for fam in sorted(set(r["family"] for r in dels))},
            "direct_answer_accuracy": rate(r.get("final_correct", False) for r in dels if r.get("action") == "answer"),
            "invoked_final_accuracy": rate(r.get("final_correct", False) for r in invoked),
        }
    epis = payload.get("epis", [])
    if epis:
        fams = {}
        for r in epis: fams.setdefault(r["family"], []).append(r["answer_correct"])
        out["EPIUTIL"] = {"n": len(epis), "accuracy": rate(r["answer_correct"] for r in epis),
                          "format_valid_rate": rate(r["format_valid"] for r in epis),
                          "family_accuracy": {k: rate(v) for k, v in fams.items()}}
    delib = payload.get("vector_delib", [])
    if delib:
        out["VECTOR_DELIB"] = {"n": len(delib), "accuracy": rate(r["answer_correct"] for r in delib),
                               "format_valid_rate": rate(r["format_valid"] for r in delib),
                               "row_accuracy_valid": rate(r["row_accuracy"] for r in delib if r["format_valid"])}
    out["AUTO_REFERENCE"] = {"accuracy": 1.0, "mechanism": "deterministic typed dispatch; architectural reference, not neural competence"}
    return out
