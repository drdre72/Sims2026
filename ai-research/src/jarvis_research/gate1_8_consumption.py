"""HZN-EPI-06 — typed-state consumption: report versus decision.

Separates output-contract repair, state-respect failure, and consumption-mode
asymmetry over the EPI-05 typed-state battery. Arm D executes a mechanical
renderer so mechanical facts never depend on neural transcription.
"""
import copy
import re
from collections import Counter

from jarvis_research.cognitive_bus import (
    AMBIGUOUS, INVALID, RESOLVED, STALE, UNKNOWN,
    auto_dispatch, thaw)
from jarvis_research.gate1_6_decompose import build_matrix_cases, build_epi_tasks, render_matrix
from jarvis_research.gate1_7_substrate import (
    build_stale_hybrid_case, decision_from_matrix_bus, epistemic_decision_from_bus,
    load_epistemic_bus, load_matrix_bus, parse_hybrid_target, render_hybrid_matrix,
    render_raw_rows, render_state_snapshot)

CONSUMPTION_VERSION = 1
CONSUMPTION_ARMS = ("READOUT_BASELINE", "READOUT_REPAIRED", "DECISION_CONSUMPTION", "MECHANICAL_RENDER")
ACTIONS = ("answer", "refresh", "query", "reject_premises", "insufficient_evidence", "abstain")
REPORT_STATUSES = (UNKNOWN, INVALID, AMBIGUOUS, STALE)
ROBUST_BASE_CASE_IDS = ("m02", "m07", "m15")
QUERY_ROBUST_FAMILIES = {"incomplete_missing_bit", "incomplete_missing_row", "unknown_bit"}
REJECT_ROBUST_FAMILIES = {"malformed_bit", "malformed_type", "contradictory_duplicate_label"}
NONINTERVENTION_ACTIONS = {"answer", "insufficient_evidence", "abstain"}


def _mutate_rows(rows, family):
    rows = copy.deepcopy(rows)
    if family == "incomplete_missing_bit":
        del rows[2]["h2"]
    elif family == "malformed_bit":
        rows[2]["h1"] = 2
    elif family == "contradictory_duplicate_label":
        rows[3]["label"] = rows[2]["label"]
    elif family == "incomplete_missing_row":
        rows = rows[:4]
    elif family == "unknown_bit":
        rows[2]["h2"] = None
    elif family == "malformed_type":
        rows[2]["h1"] = "true"
    else:
        raise ValueError(f"unknown robustness family: {family}")
    return rows


def robust_decision_action(family):
    if family in QUERY_ROBUST_FAMILIES:
        return "query"
    if family in REJECT_ROBUST_FAMILIES:
        return "reject_premises"
    raise ValueError(f"no decision policy for robustness family: {family}")


def build_consumption_robustness_cases():
    """Eighteen non-normal cases: the six EPI-05 defect types x three bases."""
    base_by_id = {c["case_id"]: c for c in build_matrix_cases()}
    specs = [
        ("incomplete_missing_bit", UNKNOWN, "one row is missing an H2 value"),
        ("malformed_bit", INVALID, "one bit is outside the binary domain"),
        ("contradictory_duplicate_label", AMBIGUOUS, "two rows carry the same label"),
        ("incomplete_missing_row", UNKNOWN, "only four of five expected rows are present"),
        ("unknown_bit", UNKNOWN, "one bit is explicitly missing"),
        ("malformed_type", INVALID, "one bit has a string type"),
    ]
    cases = []
    for family, expected, description in specs:
        for base_id in ROBUST_BASE_CASE_IDS:
            base = base_by_id[base_id]
            rows = _mutate_rows(base["rows"], family)
            cases.append({
                "case_id": f"n{len(cases):02d}", "family": family, "rows": rows,
                "truth": "status:" + expected, "expected_status": expected,
                "description": description, "base_case_id": base_id,
                "decision_action": robust_decision_action(family),
                "variant": dict(base["variant"]),
            })
    return cases


def consumption_robustness_input_value(case):
    return {"rows": copy.deepcopy(case["rows"]), "expected_rows": 5}


def load_consumption_robustness_bus(case):
    from jarvis_research.cognitive_bus import CognitiveBus
    bus = CognitiveBus()
    input_obj = bus.put(
        "binary_row_set", consumption_robustness_input_value(case),
        provenance={"source": "consumption_robustness_evidence", "case_id": case["case_id"],
                    "base_case_id": case["base_case_id"], "description": case["description"]},
        context_id=case["case_id"])
    events = auto_dispatch(bus)
    return bus, input_obj, events


def naive_matrix_answer(rows):
    """Naive premise recomputation used only as an explanatory failure tag."""
    differences = []
    for row in rows:
        if not isinstance(row, dict) or not {"label", "h1", "h2"} <= set(row):
            continue
        if row.get("h1") is None or row.get("h2") is None or not row.get("label"):
            continue
        if bool(row["h1"]) != bool(row["h2"]):
            differences.append(row["label"])
    return differences[0] if len(differences) == 1 else "none"


def matrix_core_state(bus, input_obj):
    bound = bus.binding("decision.unique")
    if bound is not None:
        value = thaw(bound.value)
        effective = bus.effective_status(bound)
        return {"effective_status": effective, "unique": value.get("unique") if effective == RESOLVED else None,
                "cached_unique": value.get("unique"), "bound_version": bound.version,
                "input_status": bus.effective_status(input_obj)}
    return {"effective_status": bus.effective_status(input_obj), "unique": None, "cached_unique": None,
            "bound_version": None, "input_status": bus.effective_status(input_obj)}


def _matrix_scenario(case, bus, input_obj, events, group, expected_report, decision_action):
    core = matrix_core_state(bus, input_obj)
    return {"scenario_id": case["case_id"], "case": case, "bus": bus, "input_obj": input_obj, "events": events,
            "family": case["family"], "state_group": group, "truth": case.get("truth"),
            "expected_report": expected_report, "expected_action": decision_action,
            "labels": [r["label"] for r in case["rows"] if isinstance(r, dict) and "label" in r],
            "naive_answer": naive_matrix_answer(case["rows"]),
            "n_dispatch_events": len(events), **core}


def build_matrix_state_scenarios():
    scenarios = []
    for case in build_matrix_cases():
        bus, input_obj, events = load_matrix_bus(case)
        scenarios.append(_matrix_scenario(case, bus, input_obj, events, "resolved", case["truth"], "answer"))
    for case in build_consumption_robustness_cases():
        bus, input_obj, events = load_consumption_robustness_bus(case)
        expected = "status:" + case["expected_status"]
        scenarios.append(_matrix_scenario(case, bus, input_obj, events, "nonnormal", expected, case["decision_action"]))
    case, bus, input_obj, events = build_stale_hybrid_case()
    scenarios.append(_matrix_scenario(case, bus, input_obj, events, "stale", "status:stale", "refresh"))
    return scenarios


def parse_readout_relaxed(response, case):
    """Preregistered narrow relaxation: only target: may substitute for target=."""
    text = str(response).strip()
    labels = sorted({r["label"] for r in case["rows"] if isinstance(r, dict) and "label" in r}, key=len, reverse=True)
    value_re = "(?:" + "|".join(map(re.escape, labels + ["none"])) + "|status:(?:unknown|invalid|ambiguous|stale))"
    m = re.fullmatch(r"target[:=](" + value_re + r")", text)
    return m.group(1) if m else None


def grade_readout(response, scenario):
    strict_answer = parse_hybrid_target(response, scenario["case"])
    relaxed_answer = parse_readout_relaxed(response, scenario["case"])
    semantic = strict_answer if strict_answer is not None else relaxed_answer
    expected = scenario["expected_report"]
    strict_correct = strict_answer == expected
    relaxed_correct = relaxed_answer == expected
    format_failure = (not strict_correct) and relaxed_correct
    state_respect = semantic == expected
    premise_recompute = semantic == scenario["naive_answer"] and semantic != expected
    if strict_correct:
        failure_class = None
    elif format_failure:
        failure_class = "format"
    elif semantic is not None and premise_recompute:
        failure_class = "premise-recompute"
    elif semantic is not None:
        failure_class = "state-ignore"
    else:
        failure_class = "hallucinated-content"
    return {"strict_answer": strict_answer, "relaxed_answer": relaxed_answer, "semantic_answer": semantic,
            "expected": expected, "strict_correct": strict_correct, "relaxed_correct": relaxed_correct,
            "format_valid": strict_answer is not None, "format_failure": format_failure,
            "state_respect": state_respect, "premise_recompute": premise_recompute,
            "failure_class": failure_class, "naive_answer": scenario["naive_answer"],
            "stale_value_emitted": scenario["state_group"] == "stale" and semantic == scenario["cached_unique"]}


def _matrix_problem_text(case):
    return render_raw_rows(case["rows"]) if case.get("expected_status") else render_matrix(case)


def render_readout_repaired(scenario):
    """Few-shot contract repair; examples supply format, not current content."""
    case, bus = scenario["case"], scenario["bus"]
    contract = (
        "Output contract examples. These examples show format only; use the current typed state for content.\n"
        "resolved unique row X -> target=X\n"
        "resolved with no unique row -> target=none\n"
        "unknown evidence state -> target=status:unknown\n"
        "invalid evidence state -> target=status:invalid\n"
        "ambiguous evidence state -> target=status:ambiguous\n"
        "stale evidence state -> target=status:stale\n"
        "The separator after target is =, never :. Do not add explanation or extra text.\n\n")
    return (contract +
            "You are the neural interpretation layer above a typed working-state runtime. "
            "The deterministic state below is authoritative for mechanical comparison facts. "
            "Do not recompute or contradict a resolved mechanical result.\n\n"
            "Original matrix:\n" + _matrix_problem_text(case) + "\n\nTyped working state:\n" + render_state_snapshot(bus) +
            "\n\nDownstream control decision: choose the single row to inspect. If exactly one differing row exists, "
            "use its label; otherwise use none. If the evidence state is unknown, invalid, ambiguous or stale, "
            "answer status:<that status>. Reply with ONLY target=<row label, none, or status:STATUS>.")


def build_epistemic_decision_scenarios():
    scenarios = []
    for task in build_epi_tasks():
        bus, input_obj, events = load_epistemic_bus(task)
        result = epistemic_decision_from_bus(bus, input_obj)
        family = task["family"]
        if family.startswith("need_"):
            expected_action = "query"
        elif family == "consistent_nodisc":
            expected_action = "insufficient_evidence"
        else:
            expected_action = "abstain"
        scenarios.append({"scenario_id": task["task_id"], "task": task, "bus": bus, "input_obj": input_obj,
                          "events": events, "family": family, "expected_action": expected_action,
                          "acceptable_choices": list(task["acceptable"]), "mechanical_result": result,
                          "n_dispatch_events": len(events)})
    return scenarios


def _action_definitions():
    return ("Actions:\n"
            "- answer: use the mechanically bound answer; never supply the answer content yourself.\n"
            "- refresh: recompute stale derived state before use.\n"
            "- query: request missing evidence or perform a useful observation.\n"
            "- reject_premises: reject invalid or contradictory evidence.\n"
            "- insufficient_evidence: stop because the evidence cannot support an answer and no useful query is available.\n"
            "- abstain: take no further action because none is needed.")


def render_matrix_decision(scenario):
    return ("You are the decision layer above a typed working-state runtime. The deterministic state is authoritative "
            "for mechanical comparison facts. Choose the next action only; do not report the row, none, or any status.\n\n"
            + _action_definitions() +
            "\n\nTyped working state:\n" + render_state_snapshot(scenario["bus"]) +
            "\n\nReply with ONLY one line: action=<answer|refresh|query|reject_premises|insufficient_evidence|abstain>.")


def render_epistemic_decision(scenario):
    return ("You are the decision layer above a typed working-state runtime. The deterministic state is authoritative "
            "for hypothesis consistency, discrimination and cost comparisons. Choose the next investigation action only.\n\n"
            + _action_definitions() +
            "\n\nFor this epistemic task, a query action must also name the chosen observation. If a useful observation "
            "should be performed, reply exactly: action=query choice=<Q1-Q4>. If uncertainty is already resolved, reply "
            "action=abstain. If uncertainty remains but no available observation can discriminate, reply "
            "action=insufficient_evidence. Do not report hypothesis consistency, costs, candidates or reasons.\n\n"
            "Typed working state:\n" + render_state_snapshot(scenario["bus"]) +
            "\n\nReply with ONLY the required action line.")


def parse_decision(response, parameterized=False):
    text = str(response).strip()
    if parameterized:
        m = re.fullmatch(r"action=query choice=(Q[1-4])", text)
        if m:
            return {"action": "query", "choice": m.group(1), "format_valid": True}
        m = re.fullmatch(r"action=(answer|refresh|reject_premises|insufficient_evidence|abstain)", text)
        if m:
            return {"action": m.group(1), "choice": None, "format_valid": True}
        return {"action": None, "choice": None, "format_valid": False}
    m = re.fullmatch(r"action=(answer|refresh|query|reject_premises|insufficient_evidence|abstain)", text)
    if m:
        return {"action": m.group(1), "choice": None, "format_valid": True}
    return {"action": None, "choice": None, "format_valid": False}


def parse_decision_relaxed(response, parameterized=False):
    """Narrow relaxation: action: may substitute for action=, and choice: for choice=."""
    text = str(response).strip()
    if text.startswith("action:"):
        text = "action=" + text[len("action:"):]
    text = text.replace(" choice:", " choice=")
    return parse_decision(text, parameterized=parameterized)


def _decision_format_failure(strict_result, relaxed_result, strict_correct, relaxed_correct):
    return (not strict_correct) and relaxed_correct and relaxed_result["format_valid"]


def grade_matrix_decision(response, scenario):
    strict = parse_decision(response, parameterized=False)
    relaxed = parse_decision_relaxed(response, parameterized=False)
    semantic = strict if strict["format_valid"] else relaxed
    expected = scenario["expected_action"]
    strict_correct = strict["format_valid"] and strict["action"] == expected
    relaxed_correct = relaxed["format_valid"] and relaxed["action"] == expected
    format_failure = _decision_format_failure(strict, relaxed, strict_correct, relaxed_correct)
    premise_recompute = semantic["action"] == "answer" and expected != "answer"
    if strict_correct:
        failure_class = None
    elif format_failure:
        failure_class = "format"
    elif premise_recompute:
        failure_class = "premise-recompute"
    elif semantic["format_valid"]:
        failure_class = "state-ignore"
    else:
        failure_class = "hallucinated-content"
    return {"action": strict["action"], "relaxed_action": relaxed["action"], "expected_action": expected,
            "action_correct": strict_correct, "relaxed_correct": relaxed_correct,
            "format_valid": strict["format_valid"], "format_failure": format_failure,
            "state_respect": relaxed_correct, "premise_recompute": premise_recompute,
            "failure_class": failure_class, "restraint_applicable": expected in NONINTERVENTION_ACTIONS,
            "restraint_correct": expected in NONINTERVENTION_ACTIONS and strict_correct}


def grade_epistemic_decision(response, scenario):
    strict = parse_decision(response, parameterized=True)
    relaxed = parse_decision_relaxed(response, parameterized=True)
    semantic = strict if strict["format_valid"] else relaxed
    expected = scenario["expected_action"]
    acceptable = set(scenario["acceptable_choices"])

    def correct(parsed):
        if not parsed["format_valid"]:
            return False
        if parsed["action"] == "query":
            return expected == "query" and parsed["choice"] in acceptable
        return parsed["choice"] is None and parsed["action"] == expected

    strict_correct = correct(strict)
    relaxed_correct = correct(relaxed)
    format_failure = _decision_format_failure(strict, relaxed, strict_correct, relaxed_correct)
    premise_recompute = scenario["family"] == "resolved_still_diff" and semantic["action"] == "query"
    unauthorized_choice = semantic["action"] == "query" and semantic["choice"] not in acceptable
    if strict_correct:
        failure_class = None
    elif format_failure:
        failure_class = "format"
    elif premise_recompute:
        failure_class = "premise-recompute"
    elif semantic["format_valid"]:
        failure_class = "state-ignore"
    else:
        failure_class = "hallucinated-content"
    return {"action": strict["action"], "choice": strict["choice"], "relaxed_action": relaxed["action"],
            "relaxed_choice": relaxed["choice"], "expected_action": expected,
            "acceptable_choices": sorted(acceptable), "action_correct": strict_correct,
            "relaxed_correct": relaxed_correct, "format_valid": strict["format_valid"],
            "format_failure": format_failure, "state_respect": relaxed_correct,
            "premise_recompute": premise_recompute, "unauthorized_choice": unauthorized_choice,
            "failure_class": failure_class, "restraint_applicable": expected in NONINTERVENTION_ACTIONS,
            "restraint_correct": expected in NONINTERVENTION_ACTIONS and strict_correct}


def render_mechanical_matrix(scenario):
    status = scenario["effective_status"]
    unique = scenario["unique"]
    if status == RESOLVED and unique == "none":
        key, text = "resolved_none", "There is no unique differing row."
    elif status == RESOLVED:
        key, text = "resolved_unique", f"Row {unique} is the unique differing row."
    elif status == UNKNOWN:
        key, text = "unknown", "The evidence is incomplete, so the comparison cannot be determined."
    elif status == INVALID:
        key, text = "invalid", "The evidence is invalid, so the comparison premises must be rejected."
    elif status == AMBIGUOUS:
        key, text = "ambiguous", "The evidence is ambiguous or contradictory, so the comparison premises must be rejected."
    elif status == STALE:
        key, text = "stale", "The comparison result is stale and must be refreshed before use."
    else:
        raise ValueError(f"unrenderable matrix status: {status}")
    if scenario["state_group"] == "resolved":
        expected_key = "resolved_none" if scenario["expected_report"] == "none" else "resolved_unique"
    else:
        expected_key = scenario["effective_status"]
    fidelity = key == expected_key
    if status == RESOLVED:
        fidelity = fidelity and unique == scenario["expected_report"]
    stale_value_emitted = (scenario["state_group"] == "stale" and scenario["cached_unique"] is not None
                           and f"Row {scenario['cached_unique']}" in text)
    return {"scenario_id": scenario["scenario_id"], "family": scenario["family"], "state_group": scenario["state_group"],
            "render_key": key, "rendered_text": text, "rendered_status": status, "mechanical_answer": unique,
            "fidelity_correct": fidelity, "stale_value_emitted": stale_value_emitted,
            "neural_input_tokens": 0, "neural_output_tokens": 0, "latency_s": 0.0}


def render_mechanical_epistemic(scenario):
    result = scenario["mechanical_result"]
    state = result.get("state") or {}
    status, choice = result.get("status"), result.get("choice")
    candidates = list(result.get("candidates", []))
    reason = state.get("reason")
    requires_choice = False
    if status == RESOLVED and choice == "no_query" and reason == "already_resolved":
        key, text = "already_resolved", "Hypothesis uncertainty is already resolved; no further observation is needed."
    elif status == RESOLVED and choice == "no_query" and reason == "no_discriminating_query":
        key, text = "no_discriminating_query", "No available observation can distinguish the remaining hypotheses; the evidence is insufficient."
    elif status == RESOLVED and choice:
        key, text = "unique_query", f"{choice} is the lowest-cost discriminating observation."
    elif status == AMBIGUOUS and reason == "equal_cost_tie":
        key = "equal_cost_tie"
        text = "The lowest-cost discriminating observations are tied: " + ", ".join(candidates) + "."
        requires_choice = True
    else:
        raise ValueError(f"unrenderable epistemic state: status={status} choice={choice} reason={reason}")
    expected = scenario["expected_action"]
    expected_key = {"insufficient_evidence": "no_discriminating_query", "abstain": "already_resolved"}.get(expected)
    if expected == "query":
        expected_key = "equal_cost_tie" if len(scenario["acceptable_choices"]) > 1 else "unique_query"
    fidelity = key == expected_key
    if key in ("already_resolved", "no_discriminating_query"):
        fidelity = fidelity and choice == "no_query" and not candidates
    else:
        fidelity = fidelity and sorted(candidates) == sorted(scenario["acceptable_choices"])
    if key == "unique_query":
        fidelity = fidelity and choice == scenario["acceptable_choices"][0]
    return {"scenario_id": scenario["scenario_id"], "family": scenario["family"], "render_key": key,
            "rendered_text": text, "rendered_status": status, "mechanical_choice": choice,
            "candidates": candidates, "requires_neural_choice": requires_choice,
            "fidelity_correct": fidelity, "end_to_end_correct": fidelity and not requires_choice,
            "neural_input_tokens": 0, "neural_output_tokens": 0, "latency_s": 0.0}


def render_tie_interpretation_prompt(render):
    candidates = list(render["candidates"])
    options = " or ".join(candidates)
    return ("A mechanical epistemic renderer has already established this fact:\n"
            + render["rendered_text"] +
            "\n\nThe only unresolved element is which tied observation to select. Choose any listed candidate. "
            f"Reply with ONLY choice=<{options}>.")


def parse_tie_choice(response):
    m = re.fullmatch(r"choice=(Q[1-4])", str(response).strip())
    return m.group(1) if m else None


def finalize_tie_render(render, choice):
    return render["rendered_text"] + f"\nSelected tied observation: {choice}."


def rate(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _cost(rows):
    return {"mean_latency_s": mean(r.get("latency_s") for r in rows),
            "mean_output_tokens": mean(r.get("output_tokens") for r in rows)}


def _failure_counts(rows):
    return dict(Counter(r.get("failure_class") for r in rows if r.get("failure_class")))


def _family_accuracy(rows, key):
    return {fam: rate(r[key] for r in rows if r["family"] == fam) for fam in sorted(set(r["family"] for r in rows))}


def _readout_summary(rows):
    return {"n": len(rows), "strict_accuracy": rate(r["strict_correct"] for r in rows),
            "relaxed_accuracy": rate(r["relaxed_correct"] for r in rows),
            "format_valid_rate": rate(r["format_valid"] for r in rows),
            "format_failure_rate": rate(r["format_failure"] for r in rows),
            "state_respect_rate": rate(r["state_respect"] for r in rows),
            "resolved_strict_accuracy": rate(r["strict_correct"] for r in rows if r["state_group"] == "resolved"),
            "nonnormal_strict_accuracy": rate(r["strict_correct"] for r in rows if r["state_group"] == "nonnormal"),
            "stale_strict_accuracy": rate(r["strict_correct"] for r in rows if r["state_group"] == "stale"),
            "family_strict_accuracy": _family_accuracy(rows, "strict_correct"),
            "failure_counts": _failure_counts(rows), **_cost(rows)}


def _decision_summary(rows):
    restraint = [r for r in rows if r.get("restraint_applicable")]
    return {"n": len(rows), "decision_appropriateness": rate(r["action_correct"] for r in rows),
            "relaxed_decision_appropriateness": rate(r["relaxed_correct"] for r in rows),
            "format_valid_rate": rate(r["format_valid"] for r in rows),
            "format_failure_rate": rate(r["format_failure"] for r in rows),
            "state_respect_rate": rate(r["state_respect"] for r in rows),
            "restraint_n": len(restraint), "restraint_accuracy": rate(r["action_correct"] for r in restraint),
            "family_accuracy": _family_accuracy(rows, "action_correct"),
            "failure_counts": _failure_counts(rows), **_cost(rows)}


def summarize_epi6(payload):
    out = {}
    baseline = payload.get("readout_baseline", [])
    if baseline:
        out["READOUT_BASELINE"] = _readout_summary(baseline)
    repaired = payload.get("readout_repaired", [])
    if repaired:
        out["READOUT_REPAIRED"] = _readout_summary(repaired)
    md = payload.get("decision_matrix", [])
    ed = payload.get("decision_epistemic", [])
    if md:
        out["MATRIX_DECISION"] = _decision_summary(md)
    if ed:
        out["EPISTEMIC_DECISION"] = _decision_summary(ed)
    if md or ed:
        out["DECISION_CONSUMPTION"] = _decision_summary(md + ed)
    mm = payload.get("mechanical_matrix_render", [])
    me = payload.get("mechanical_epistemic_render", [])
    if mm or me:
        ties = [r for r in me if r.get("requires_neural_choice")]
        out["MECHANICAL_RENDER"] = {
            "matrix_n": len(mm), "matrix_fidelity_rate": rate(r["fidelity_correct"] for r in mm),
            "matrix_stale_value_emission_count": sum(r.get("stale_value_emitted", False) for r in mm),
            "epistemic_n": len(me), "epistemic_mechanical_fidelity_rate": rate(r["fidelity_correct"] for r in me),
            "end_to_end_accuracy": rate(r["end_to_end_correct"] for r in me),
            "tie_cases": len(ties), "tie_neural_correct_rate": rate(r.get("tie_choice_correct", False) for r in ties),
            "neural_generations": len(ties),
            "mean_tie_latency_s": mean(r.get("latency_s") for r in ties),
            "mean_tie_output_tokens": mean(r.get("output_tokens") for r in ties),
            "zero_neural_token_matrix_cases": sum(r.get("neural_output_tokens") == 0 for r in mm),
        }
    return out
