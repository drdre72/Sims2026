"""HZN-OOI-01 — operational orientation and exact-tie inhibition probe.

Tests whether requiring a structured intermediate orientation artifact changes
end-to-end candidate decisions, and whether deterministic rule evaluation over
the model's extracted evidence can separate observation/extraction failures
from mechanical inhibition failures. Inference-only construction and grading;
no intervention authority is implied.
"""

import hashlib
import random
import re

from jarvis_research.gate1_4_benchmark import (
    LABELS, RULES, grade_task, magnitude_answer, render_task, render_task_body)
from jarvis_research.gate1_4_mirror import (
    answer_verdict, candidate_evaluation, normalize_answer)


OOI_VERSION = 43
N_PER_FAMILY = 6
OOI_FAMILIES = ("exact_tie", "near_tie", "insufficient", "unique_control")
CONDITIONS = ("BASE", "STRUCT")
RULE_SEQUENCE = ("conj_active", "conj_strong", "avoid_negative")
STRUCT_HEADER = "label|ar|lc|ar_pass|contrast_pass|qualifies|tb"


def scenario_seed(version, family, replicate):
    return int.from_bytes(hashlib.sha256(
        f"ooi-v{version}-{family}:{replicate}".encode()).digest()[:8],
        "little")


def tiebreak_name(rule_key):
    return "ar" if rule_key == "conj_active" else "abs_lc"


def rule_summary(rule_key, candidates):
    """Deterministic rule mechanics in a form suited to artifact grading."""
    evaluations = []
    for c in candidates:
        ev = candidate_evaluation(rule_key, c)
        evaluations.append((c, ev))
    qual = [(c, ev) for c, ev in evaluations if ev["qualifies"]]
    if not qual:
        return {"qualifiers": [], "best_tb": None, "winners": [],
                "state": "none_qualify", "answer": "none"}
    best = max(ev["tiebreak_value"] for c, ev in qual)
    winners = [c for c, ev in qual if ev["tiebreak_value"] == best]
    state = "tie" if len(winners) > 1 else "unique"
    return {"qualifiers": [c["label"] for c, ev in qual],
            "best_tb": best,
            "winners": [c["label"] for c in winners],
            "state": state,
            "answer": "none" if state == "tie" else winners[0]["label"]}


def _failing_candidate(item, rng, mode):
    if mode == 0:
        ar = round(rng.uniform(0.05, 0.25), 2)
        lc = rng.choice([-1, 1]) * round(rng.uniform(0.80, 1.50), 2)
    elif mode == 1:
        ar = round(rng.uniform(0.65, 0.90), 2)
        lc = -round(rng.uniform(0.05, 0.20), 2)
    else:
        ar = round(rng.uniform(0.05, 0.25), 2)
        lc = -round(rng.uniform(0.80, 1.50), 2)
    return {"item": item, "ar": ar, "lc": lc, "obs_available": True}


def make_tie_scenario(rng, rule_key, family):
    """Construct an exact- or near-tie scenario for a specific rule.

    The two critical candidates always qualify. In exact_tie their tie-break
    values are identical; in near_tie the winner is exactly 0.01 higher.
    All non-critical candidates fail under every supported rule.
    """
    base_tb = round(rng.uniform(0.68, 0.78), 2)
    high_tb = base_tb if family == "exact_tie" else round(base_tb + 0.01, 2)

    if rule_key == "conj_active":
        low = {"item": "crit_low", "ar": base_tb,
               "lc": round(rng.uniform(0.55, 0.75), 2),
               "obs_available": True}
        high = {"item": "crit_high", "ar": high_tb,
                "lc": round(rng.uniform(0.55, 0.75), 2),
                "obs_available": True}
    else:
        low = {"item": "crit_low",
               "ar": round(rng.uniform(0.65, 0.85), 2),
               "lc": base_tb, "obs_available": True}
        high = {"item": "crit_high",
                "ar": round(rng.uniform(0.65, 0.85), 2),
                "lc": high_tb, "obs_available": True}

    cands = [low, high]
    for i in range(3):
        cands.append(_failing_candidate(f"fill{i}", rng, i))
    rng.shuffle(cands)
    return {"family": family, "rule_key": rule_key, "candidates": cands,
            "critical_items": ["crit_low", "crit_high"],
            "correct_item": None if family == "exact_tie" else "crit_high"}


def make_insufficient_scenario(rng, rule_key):
    cands = [_failing_candidate(f"fill{i}", rng, i % 3) for i in range(5)]
    rng.shuffle(cands)
    return {"family": "insufficient", "rule_key": rule_key,
            "candidates": cands, "critical_items": [], "correct_item": None}


def make_unique_control_scenario(rng, rule_key):
    winner = {"item": "control_winner",
              "ar": round(rng.uniform(0.82, 0.92), 2),
              "lc": round(rng.uniform(0.55, 0.70), 2),
              "obs_available": True}
    king = {"item": "magnitude_king",
            "ar": round(rng.uniform(0.05, 0.20), 2),
            "lc": rng.choice([-1, 1]) * round(rng.uniform(1.20, 1.60), 2),
            "obs_available": True}
    cands = [winner, king]
    for i in range(3):
        cands.append(_failing_candidate(f"fill{i}", rng, i))
    rng.shuffle(cands)
    return {"family": "unique_control", "rule_key": rule_key,
            "candidates": cands, "critical_items": [],
            "correct_item": "control_winner"}


def make_operational_scenario(rng, family, rule_key):
    if family in ("exact_tie", "near_tie"):
        return make_tie_scenario(rng, rule_key, family)
    if family == "insufficient":
        return make_insufficient_scenario(rng, rule_key)
    if family == "unique_control":
        return make_unique_control_scenario(rng, rule_key)
    raise ValueError(family)


def build_operational_scenarios(n_per_family=N_PER_FAMILY, version=OOI_VERSION):
    scenarios = []
    for fi, family in enumerate(OOI_FAMILIES):
        for rep in range(n_per_family):
            rng = random.Random(scenario_seed(version, family, rep))
            rule_key = RULE_SEQUENCE[(fi + rep) % len(RULE_SEQUENCE)]
            sc = make_operational_scenario(rng, family, rule_key)
            sc["scenario_index"] = len(scenarios)
            scenarios.append(sc)
    return scenarios


def _present_scenario(scenario, ordered_candidates, presentation):
    cands = []
    item_to_label = {}
    label_to_item = {}
    for i, src in enumerate(ordered_candidates):
        c = dict(src)
        c["label"] = LABELS[i]
        item_to_label[c["item"]] = c["label"]
        label_to_item[c["label"]] = c["item"]
        cands.append(c)

    truth = grade_task(scenario["rule_key"], cands)
    mechanics = rule_summary(scenario["rule_key"], cands)
    correct_item = scenario["correct_item"]
    if correct_item is not None and truth["answer"] != item_to_label[correct_item]:
        raise AssertionError(
            f"construction drift: {scenario['family']} truth {truth['answer']} "
            f"does not identify {correct_item}")
    if correct_item is None and truth["answer"] != "none":
        raise AssertionError(
            f"construction drift: {scenario['family']} unexpectedly has answer "
            f"{truth['answer']}")

    critical_items = list(scenario["critical_items"])
    critical_positions = [i for i, c in enumerate(cands)
                          if c["item"] in critical_items]
    first_critical_item = None
    if critical_positions:
        first_pos = min(critical_positions)
        first_critical_item = cands[first_pos]["item"]
    return {
        "family": scenario["family"], "rule_key": scenario["rule_key"],
        "scenario_index": scenario["scenario_index"],
        "presentation": presentation,
        "task_id": f"s{scenario['scenario_index']:02d}_{presentation}",
        "candidates": cands, "truth": truth, "mechanics": mechanics,
        "magnitude_answer": magnitude_answer(cands),
        "critical_items": critical_items,
        "critical_labels": [item_to_label[i] for i in critical_items],
        "critical_positions": critical_positions,
        "first_critical_item": first_critical_item,
        "first_critical_label": (item_to_label.get(first_critical_item)
                                 if first_critical_item else None),
        "item_to_label": item_to_label,
        "label_to_item": label_to_item,
    }


def expand_scenario_presentations(scenario):
    original = list(scenario["candidates"])
    variants = [("order_a", original)]
    if scenario["critical_items"]:
        swapped = list(original)
        positions = [i for i, c in enumerate(swapped)
                     if c["item"] in scenario["critical_items"]]
        if len(positions) == 2:
            i, j = positions
            swapped[i], swapped[j] = swapped[j], swapped[i]
            variants.append(("pair_swapped", swapped))
    return [_present_scenario(scenario, order, name)
            for name, order in variants]


def build_operational_tasks(n_per_family=N_PER_FAMILY, version=OOI_VERSION):
    """Build the fixed presentation battery.

    Exact/near-tie scenarios occur in two critical-pair orders; controls occur
    once. With the default six replicates per family this yields 24 scenarios
    and 36 presentations: 12 exact tie, 12 near tie, 6 insufficient, and 6
    unique-control presentations.
    """
    tasks = []
    for scenario in build_operational_scenarios(n_per_family, version):
        tasks.extend(expand_scenario_presentations(scenario))
    return tasks


def render_structured_task(task):
    """Render the operational-artifact arm.

    The topology is operational rather than merely verbal: who/what/where are
    represented by row extraction, when by the none state, how by threshold and
    tie-break fields, and why by the final evidence-justified decision.
    """
    n_cand = len(task["candidates"])
    instructions = f"""Produce the operational orientation artifact before the final answer.
In this artifact, who is the candidate label, what is the observed evidence, where is the binding of evidence to rule fields, when is the none condition, how is the threshold and tie-break computation, and why is the evidence-justified final decision. Do not write this explanation.

The output must contain exactly {n_cand + 4} lines for the {n_cand} candidates:
- line 1: tiebreak=<ar or abs_lc>
- line 2: label|ar|lc|ar_pass|contrast_pass|qualifies|tb
- lines 3-{n_cand + 2}: one row for each candidate, in the order shown above, formatted as label|ar|lc|ar_pass|contrast_pass|qualifies|tb
- line {n_cand + 3}: summary|qualifiers=<comma-separated labels or none>|best_tb=<number or na>|winners=<comma-separated labels or none>|state=<unique, tie, or none_qualify>
- line {n_cand + 4}: answer=<candidate letter or none>

Copy ar and lc exactly as shown. Use yes or no for pass and qualification fields. If observations are missing, use na for ar, lc, both pass fields, and tb, with qualifies=no. Otherwise ar, lc, and tb must be numeric and the pass fields must be yes or no. contrast_pass means the rule's contrast or sign requirement. tb is the numeric tie-break value. Missing observations cannot qualify. No other text."""
    return render_task_body(task) + "\n\n" + instructions


def render_operational_condition(task, condition):
    if condition == "BASE":
        return render_task(task)
    if condition == "STRUCT":
        return render_structured_task(task)
    raise ValueError(condition)


def _parse_number(text):
    if text == "na":
        return None
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        raise ValueError(text)
    return float(text)


def _parse_yes_no_na(text):
    if text == "yes":
        return True
    if text == "no":
        return False
    if text == "na":
        return None
    raise ValueError(text)


def _parse_label_list(text, valid_labels):
    if text == "none":
        return []
    labels = [p.strip() for p in text.split(",") if p.strip()]
    if not labels or any(x not in valid_labels for x in labels):
        raise ValueError(text)
    if len(set(labels)) != len(labels):
        raise ValueError(text)
    return labels


def parse_structured_artifact(response, candidate_labels):
    """Strictly parse the operational artifact.

    candidate_labels are the positional labels expected in row order (normally
    A-E). Invalid rows or summaries are output-contract failures; they are not
    interpreted as orientation evidence.
    """
    candidate_labels = list(candidate_labels)
    lines = [ln.strip() for ln in str(response).splitlines() if ln.strip()]
    if len(lines) != len(candidate_labels) + 4:
        return {"parse_status": "invalid_format"}

    m = re.fullmatch(r"tiebreak=(ar|abs_lc)", lines[0])
    if not m or lines[1] != STRUCT_HEADER:
        return {"parse_status": "invalid_format"}

    rows = []
    try:
        for expected_label, line in zip(candidate_labels,
                                        lines[2:2 + len(candidate_labels)]):
            parts = line.split("|")
            if len(parts) != 7 or parts[0] != expected_label:
                return {"parse_status": "invalid_format"}
            ar = _parse_number(parts[1])
            lc = _parse_number(parts[2])
            ar_pass = _parse_yes_no_na(parts[3])
            contrast_pass = _parse_yes_no_na(parts[4])
            if parts[5] not in ("yes", "no"):
                return {"parse_status": "invalid_format"}
            qualifies = parts[5] == "yes"
            tb = _parse_number(parts[6])
            if (ar is None) != (lc is None):
                return {"parse_status": "invalid_format"}
            obs = ar is not None
            if obs:
                if any(x is None for x in (ar_pass, contrast_pass, tb)):
                    return {"parse_status": "invalid_format"}
            else:
                if any(x is not None for x in (ar_pass, contrast_pass, tb)):
                    return {"parse_status": "invalid_format"}
                if qualifies:
                    return {"parse_status": "invalid_format"}
            rows.append({"label": parts[0], "obs": obs, "ar": ar, "lc": lc,
                         "ar_pass": ar_pass, "contrast_pass": contrast_pass,
                         "qualifies": qualifies, "tb": tb})
    except (ValueError, TypeError):
        return {"parse_status": "invalid_format"}

    sparts = lines[2 + len(candidate_labels)].split("|")
    if len(sparts) != 5 or sparts[0] != "summary":
        return {"parse_status": "invalid_format"}
    try:
        keys = []
        vals = []
        for part in sparts[1:]:
            k, v = part.split("=", 1)
            keys.append(k)
            vals.append(v)
        if keys != ["qualifiers", "best_tb", "winners", "state"]:
            return {"parse_status": "invalid_format"}
        summary = {
            "qualifiers": _parse_label_list(vals[0], candidate_labels),
            "best_tb": _parse_number(vals[1]),
            "winners": _parse_label_list(vals[2], candidate_labels),
            "state": vals[3],
        }
        if summary["state"] not in ("unique", "tie", "none_qualify"):
            return {"parse_status": "invalid_format"}
    except (ValueError, TypeError):
        return {"parse_status": "invalid_format"}

    m = re.fullmatch(r"answer=([A-Za-z]+)", lines[-1])
    if not m:
        return {"parse_status": "invalid_format"}
    answer = normalize_answer(m.group(1))
    if answer not in candidate_labels + ["none"]:
        return {"parse_status": "invalid_answer", "rows": rows,
                "summary": summary, "answer": answer,
                "tiebreak": m and lines[0].split("=", 1)[1]}
    return {"parse_status": "valid", "tiebreak": lines[0].split("=", 1)[1],
            "rows": rows, "summary": summary, "answer": answer}


def _num_eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(round(a, 2) - round(b, 2)) < 1e-9


def _raw_row_correct(row, candidate):
    if row["obs"] != candidate["obs_available"]:
        return False
    if not candidate["obs_available"]:
        return True
    return _num_eq(row["ar"], candidate["ar"]) and \
        _num_eq(row["lc"], candidate["lc"])


def _eval_row_correct(row, evaluation):
    return (row["ar_pass"] == evaluation["ar_pass"] and
            row["contrast_pass"] == evaluation["contrast_pass"] and
            row["qualifies"] == evaluation["qualifies"] and
            _num_eq(row["tb"], evaluation["tiebreak_value"]))


def extracted_candidates(parsed):
    return [{"label": r["label"], "ar": r["ar"], "lc": r["lc"],
             "obs_available": r["obs"]} for r in parsed.get("rows", [])]


def grade_structured_artifact(response, task):
    """Grade final behavior, extraction, evaluation, summary, and delegation."""
    labels = [c["label"] for c in task["candidates"]]
    parsed = parse_structured_artifact(response, labels)
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
            "eval_row_accuracy": 0.0,
            "internal_eval_row_accuracy": 0.0,
            "tiebreak_correct": False,
            "summary_correct": False,
            "delegated_answer": None,
            "delegated_correct": False,
        })
        return out

    raw_flags = []
    eval_flags = []
    internal_flags = []
    extracted = extracted_candidates(parsed)
    extracted_by_label = {c["label"]: c for c in extracted}
    for source in task["candidates"]:
        row = parsed["rows"][labels.index(source["label"])]
        raw_flags.append(_raw_row_correct(row, source))
        expected_ev = candidate_evaluation(task["rule_key"], source)
        eval_flags.append(_eval_row_correct(row, expected_ev))
        internal_ev = candidate_evaluation(
            task["rule_key"], extracted_by_label[source["label"]])
        internal_flags.append(_eval_row_correct(row, internal_ev))

    expected_summary = rule_summary(task["rule_key"], task["candidates"])
    ps = parsed["summary"]
    summary_correct = (
        ps["qualifiers"] == expected_summary["qualifiers"] and
        _num_eq(ps["best_tb"], expected_summary["best_tb"]) and
        ps["winners"] == expected_summary["winners"] and
        ps["state"] == expected_summary["state"])
    delegated = rule_summary(task["rule_key"], extracted)
    verdict = answer_verdict(parsed["answer"], truth, valid)
    out.update({
        "answer_verdict": verdict,
        "answer_correct": verdict == "correct",
        "format_valid": True,
        "raw_extraction_exact": all(raw_flags),
        "raw_row_accuracy": sum(raw_flags) / len(raw_flags),
        "eval_row_accuracy": sum(eval_flags) / len(eval_flags),
        "internal_eval_row_accuracy": sum(internal_flags) / len(internal_flags),
        "tiebreak_correct": parsed["tiebreak"] == tiebreak_name(task["rule_key"]),
        "summary_correct": summary_correct,
        "delegated_answer": delegated["answer"],
        "delegated_correct": delegated["answer"] == truth,
        "delegated_summary": delegated,
        "raw_flags": raw_flags,
        "eval_flags": eval_flags,
        "internal_eval_flags": internal_flags,
    })
    return out


def apply_exact_tie_guard(answer, rule_key, candidates):
    """Narrow deterministic inhibition: exact winning ties must become none."""
    if rule_summary(rule_key, candidates)["state"] == "tie":
        return "none"
    return normalize_answer(answer)


def render_reference_artifact(task):
    """Render the deterministic reference artifact for parser tests."""
    lines = [f"tiebreak={tiebreak_name(task['rule_key'])}", STRUCT_HEADER]
    for c in task["candidates"]:
        ev = candidate_evaluation(task["rule_key"], c)
        if not c["obs_available"]:
            lines.append(f"{c['label']}|na|na|na|na|no|na")
            continue
        yn = lambda x: "yes" if x else "no"
        lines.append(
            f"{c['label']}|{c['ar']:.2f}|{c['lc']:.2f}|"
            f"{yn(ev['ar_pass'])}|{yn(ev['contrast_pass'])}|"
            f"{yn(ev['qualifies'])}|{ev['tiebreak_value']:.2f}")
    mech = task["mechanics"]
    labels_or_none = lambda xs: ",".join(xs) if xs else "none"
    best = "na" if mech["best_tb"] is None else f"{mech['best_tb']:.2f}"
    lines.append(
        f"summary|qualifiers={labels_or_none(mech['qualifiers'])}|"
        f"best_tb={best}|winners={labels_or_none(mech['winners'])}|"
        f"state={mech['state']}")
    lines.append(f"answer={mech['answer']}")
    return "\n".join(lines)


def _rate(values):
    vals = list(values)
    return sum(vals) / len(vals) if vals else None


def _family_rows(rows, family):
    return [(c, r) for c, r in rows if c["family"] == family]


def summarize_operational_cells(cells, conditions=CONDITIONS):
    """Compute preregistered operational-orientation metrics."""
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

        s = {
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
        out[cond] = s

    struct_rows = []
    for c in cells:
        r = c.get("conditions", {}).get("STRUCT")
        if r and "answer_correct" in r:
            struct_rows.append((c, r))
    if struct_rows:
        answer_present = [(c, r) for c, r in struct_rows if c["truth"] != "none"]
        none_truth = [(c, r) for c, r in struct_rows if c["truth"] == "none"]
        out["STRUCT_DELEGATED"] = {
            "n": len(struct_rows),
            "accuracy": _rate(bool(r["delegated_correct"]) for c, r in struct_rows),
            "selection_accuracy_answer_present": _rate(
                bool(r["delegated_correct"]) for c, r in answer_present),
            "abstention_accuracy_none": _rate(
                bool(r["delegated_correct"]) for c, r in none_truth),
            "overinhibition_rate_answer_present": _rate(
                r.get("delegated_answer") == "none" for c, r in answer_present),
            "family_accuracy": {
                fam: _rate(bool(r["delegated_correct"])
                           for c, r in _family_rows(struct_rows, fam))
                for fam in OOI_FAMILIES
            },
        }
        valid_struct_rows = [(c, r) for c, r in struct_rows if r.get("format_valid")]
        out["STRUCT_ARTIFACT"] = {
            "n_valid": len(valid_struct_rows),
            "raw_extraction_exact_rate": _rate(
                bool(r["raw_extraction_exact"]) for c, r in valid_struct_rows),
            "raw_row_accuracy": _rate(r["raw_row_accuracy"] for c, r in valid_struct_rows),
            "eval_row_accuracy": _rate(r["eval_row_accuracy"] for c, r in valid_struct_rows),
            "internal_eval_row_accuracy": _rate(
                r["internal_eval_row_accuracy"] for c, r in valid_struct_rows),
            "tiebreak_accuracy": _rate(bool(r["tiebreak_correct"])
                                       for c, r in valid_struct_rows),
            "summary_accuracy": _rate(bool(r["summary_correct"])
                                      for c, r in valid_struct_rows),
        }

        def delegation_pair_metrics(rows):
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

        exact_struct_rows = _family_rows(struct_rows, "exact_tie")
        out["STRUCT_DELEGATION_PAIRING"] = delegation_pair_metrics(struct_rows)
        out["STRUCT_DELEGATION_PAIRING"]["exact_tie"] = \
            delegation_pair_metrics(exact_struct_rows)
        neural_acc = out["STRUCT"]["accuracy"]
        del_acc = out["STRUCT_DELEGATED"]["accuracy"]
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

    comparisons = {}
    for a, b in (("BASE", "STRUCT"),):
        both = []
        for c in cells:
            ra = c.get("conditions", {}).get(a)
            rb = c.get("conditions", {}).get(b)
            if ra and rb and "answer_correct" in ra and "answer_correct" in rb:
                both.append((c, bool(ra["answer_correct"]),
                             bool(rb["answer_correct"])))
        comparisons[f"{a}_vs_{b}"] = {
            "n": len(both),
            f"{a}_only_correct": sum(1 for c, x, y in both if x and not y),
            f"{b}_only_correct": sum(1 for c, x, y in both if y and not x),
            "accuracy_delta": ((out[a]["accuracy"] or 0) -
                               (out[b]["accuracy"] or 0)),
        }
    out["paired_comparisons"] = comparisons
    return out
