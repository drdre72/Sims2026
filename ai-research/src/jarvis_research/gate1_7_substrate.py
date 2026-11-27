"""HZN-EPI-05 — typed cognitive substrate and structural binding.

Compares neural-only decisions, manual typed invocation with neural
retranscription, structural result binding, automatic typed dispatch, and hybrid
interpretation over read-only working state. All deterministic references are
executed through the cognitive bus; no automatic ceiling is merely declared.
"""
import copy
import json
import re
import time

from jarvis_research.cognitive_bus import (
    AMBIGUOUS, INVALID, RESOLVED, UNKNOWN, VALID,
    CognitiveBus, auto_dispatch, apply_update_proposal, propose_premise_update,
    thaw, canonical_json)
from jarvis_research.gate1_6_decompose import (
    build_matrix_cases, build_epi_tasks, render_matrix, render_epi_task,
    parse_delegate_action, operand_exact, grade_epi, parse_choice)
from jarvis_research.gate1_6_epistemic import build_epistemic_tasks

SUBSTRATE_VERSION = 1
PRIMARY_ARMS = ("NEURAL_DIRECT", "TOOL_CONTEXT", "BOUND_INVOKE", "AUTO_DISPATCH", "HYBRID")
DIAGNOSTIC_ARMS = ("TOOL_FRESH",)


def matrix_input_value(case):
    rows = [{"label": r["label"], "h1": bool(r["h1"]), "h2": bool(r["h2"])} for r in case["rows"]]
    return {"rows": rows, "expected_rows": len(rows)}


def load_matrix_bus(case):
    bus = CognitiveBus()
    input_obj = bus.put(
        "binary_row_set", matrix_input_value(case),
        provenance={"source": "structured_benchmark_rows", "case_id": case.get("case_id", "unknown")},
        context_id=case.get("case_id", "unknown"))
    events = auto_dispatch(bus)
    return bus, input_obj, events


def decision_from_matrix_bus(bus, input_obj):
    bound = bus.binding("decision.unique")
    if bound is not None:
        status = bus.effective_status(bound)
        value = thaw(bound.value)
        if status == RESOLVED:
            return {"decision": value["unique"], "status": status, "state": value,
                    "output_id": bound.object_id, "output_version": bound.version}
        return {"decision": "status:" + status, "status": status, "state": value,
                "output_id": bound.object_id, "output_version": bound.version}
    status = bus.effective_status(input_obj)
    return {"decision": "status:" + status, "status": status, "state": None,
            "output_id": None, "output_version": None}


def run_auto_matrix_case(case):
    started = time.perf_counter()
    bus, input_obj, events = load_matrix_bus(case)
    runtime_ms = (time.perf_counter() - started) * 1000
    result = decision_from_matrix_bus(bus, input_obj)
    return {"case_id": case["case_id"], "family": case.get("family", "valid"), "rows": case["rows"],
            "truth": case.get("truth"), "decision": result["decision"], "status": result["status"],
            "state": result["state"], "answer_correct": result["decision"] == case.get("truth"),
            "input_status": bus.effective_status(input_obj), "events": events,
            "n_dispatch_events": len(events), "runtime_ms": round(runtime_ms, 3),
            "output_id": result["output_id"], "output_version": result["output_version"],
            "snapshot": bus.snapshot()}


def render_direct_matrix(case):
    labels = "/".join(r["label"] for r in case["rows"])
    return (render_matrix(case) + "\n\nSelect the unique row where H1 and H2 differ. "
            "If there is not exactly one differing row, answer none. "
            f"Reply with ONLY final=<{labels} or none>.")


def parse_matrix_final(response, case):
    text = str(response).strip()
    if not text.startswith("final="):
        return None
    value = text.split("=", 1)[1].strip()
    labels = [r["label"] for r in case["rows"]]
    return value if value in labels or value == "none" else None


def grade_matrix_final(response, case):
    answer = parse_matrix_final(response, case)
    return {"answer": answer, "expected": case["truth"], "answer_correct": answer == case["truth"],
            "format_valid": answer is not None}


def rows_from_call(call_rows, case):
    order = [r["label"] for r in case["rows"]]
    return [{"label": label, "h1": bool(call_rows[label][0]), "h2": bool(call_rows[label][1])}
            for label in order]


def execute_matrix_invocation(call_rows, case):
    bus = CognitiveBus()
    input_obj = bus.put(
        "binary_row_set", {"rows": rows_from_call(call_rows, case), "expected_rows": len(case["rows"])},
        provenance={"source": "neural_typed_invocation", "case_id": case["case_id"]},
        context_id=case["case_id"])
    events = auto_dispatch(bus)
    result = decision_from_matrix_bus(bus, input_obj)
    return bus, input_obj, events, result


def comparator_public_result(result):
    state = result.get("state") or {}
    return {"differences": list(state.get("differences", [])),
            "count": state.get("count"),
            "unique": state.get("unique"),
            "status": result.get("status")}


def render_tool_result(result, case, context_preserving=True):
    public = comparator_public_result(result)
    differences = ",".join(public["differences"]) if public["differences"] else "none"
    if context_preserving:
        intro = "The typed comparator has executed for the original matrix task."
    else:
        intro = "A comparator produced the following result."
    labels = "/".join(r["label"] for r in case["rows"])
    return (intro + "\n"
            f"differences={differences}\ncount={public['count']}\nunique={public['unique']}\n\n"
            "Use the comparator result. "
            f"Reply with ONLY final=<{labels} or none>.")


def compact_value(value):
    return json.dumps(thaw(value), sort_keys=True, separators=(",", ":"))


def render_state_snapshot(bus):
    lines = []
    inverse = {object_id: name for name, object_id in bus.bindings().items()}
    for obj in bus.objects():
        name = inverse.get(obj.object_id, obj.kind)
        lines.append(f"{name}: kind={obj.kind} status={bus.effective_status(obj)} version={obj.version} value={compact_value(obj.value)}")
    return "\n".join(lines)


def render_raw_rows(rows):
    lines = []
    for i, row in enumerate(rows):
        if isinstance(row, dict):
            lines.append(f"row {i}: label={row.get('label', '<missing>')!r}, H1={row.get('h1', '<missing>')!r}, H2={row.get('h2', '<missing>')!r}")
        else:
            lines.append(f"row {i}: {row!r}")
    return "\n".join(lines)


def render_hybrid_matrix(case, bus):
    matrix_text = render_raw_rows(case["rows"]) if case.get("expected_status") else render_matrix(case)
    return ("You are the neural interpretation layer above a typed working-state runtime. "
            "The deterministic state below is authoritative for mechanical comparison facts. "
            "Do not recompute or contradict a resolved mechanical result.\n\n"
            "Original matrix:\n" + matrix_text + "\n\nTyped working state:\n" + render_state_snapshot(bus) +
            "\n\nDownstream control decision: choose the single row to inspect. If exactly one differing row exists, "
            "use its label; otherwise use none. If the evidence state is unknown, invalid, ambiguous or stale, "
            "answer status:<that status>. Reply with ONLY target=<row label, none, or status:STATUS>.")


def parse_hybrid_target(response, case):
    text = str(response).strip()
    if not text.startswith("target="):
        return None
    value = text.split("=", 1)[1].strip()
    labels = [r["label"] for r in case["rows"]]
    if value in labels or value == "none" or re.fullmatch(r"status:(unknown|invalid|ambiguous|stale)", value):
        return value
    return None


def grade_hybrid_matrix(response, case, bus):
    answer = parse_hybrid_target(response, case)
    expected = case.get("expected_status")
    if expected:
        correct_answer = "status:" + expected
    else:
        correct_answer = case["truth"]
    bound = bus.binding("decision.unique")
    bound_value = None
    if bound is not None and bus.effective_status(bound) == RESOLVED:
        bound_value = thaw(bound.value)["unique"]
    return {"answer": answer, "expected": correct_answer, "answer_correct": answer == correct_answer,
            "format_valid": answer is not None, "bound_value": bound_value,
            "state_consistent": (answer == bound_value) if bound_value is not None and answer and not answer.startswith("status:") else None}


def epistemic_input_value(task):
    return {"consistent": list(task["consistent"]),
            "queries": [{"label": r["label"], "cost": r["cost"], "h1": bool(r["h1"]), "h2": bool(r["h2"])}
                         for r in task["rows"]]}


def load_epistemic_bus(task):
    bus = CognitiveBus()
    input_obj = bus.put(
        "epistemic_matrix_problem", epistemic_input_value(task),
        provenance={"source": "structured_epistemic_matrix", "task_id": task["task_id"]},
        context_id=task["task_id"])
    events = auto_dispatch(bus)
    return bus, input_obj, events


def epistemic_decision_from_bus(bus, input_obj):
    bound = bus.binding("epistemic.choice")
    if bound is None:
        status = bus.effective_status(input_obj)
        return {"choice": None, "candidates": [], "status": status, "state": None}
    value = thaw(bound.value)
    return {"choice": value.get("choice"), "candidates": list(value.get("candidates", [])),
            "status": bus.effective_status(bound), "state": value,
            "output_id": bound.object_id, "output_version": bound.version}


def run_auto_epistemic_case(task):
    started = time.perf_counter()
    bus, input_obj, events = load_epistemic_bus(task)
    runtime_ms = (time.perf_counter() - started) * 1000
    result = epistemic_decision_from_bus(bus, input_obj)
    correctness = grade_auto_epistemic_result(result, task)
    return {"task_id": task["task_id"], "family": task["family"], "acceptable": list(task["acceptable"]),
            "result": result, "events": events, "n_dispatch_events": len(events),
            "runtime_ms": round(runtime_ms, 3), "input_status": bus.effective_status(input_obj),
            "snapshot": bus.snapshot(), **correctness}


def grade_auto_epistemic_result(result, task):
    acceptable = list(task["acceptable"])
    status = result.get("status")
    choice = result.get("choice")
    candidates = list(result.get("candidates", []))
    if acceptable == ["no_query"]:
        correct = status == RESOLVED and choice == "no_query"
        mechanical_expected = "no_query"
    elif len(acceptable) > 1:
        correct = status == AMBIGUOUS and sorted(candidates) == sorted(acceptable)
        mechanical_expected = "ambiguous:" + ",".join(sorted(acceptable))
    else:
        correct = status == RESOLVED and choice == acceptable[0]
        mechanical_expected = acceptable[0]
    return {"auto_correct": correct, "mechanical_expected": mechanical_expected,
            "choice": choice, "candidates": candidates, "status": status}


def render_hybrid_epistemic(task, bus):
    return ("You are the neural interpretation layer above a typed working-state runtime. "
            "The deterministic state below is authoritative for hypothesis consistency, discrimination and cost comparisons. "
            "Do not recompute or contradict a resolved mechanical result. If the state is explicitly ambiguous because "
            "multiple lowest-cost queries are tied, choose any listed candidate.\n\n"
            + render_epi_task(task).rsplit("Reply with ONLY", 1)[0]
            + "Typed working state:\n" + render_state_snapshot(bus)
            + "\n\nChoose the investigation action. Reply with ONLY choice=<Q1-Q4 or no_query>.")


def grade_hybrid_epistemic(response, task):
    return grade_epi(response, task)


def build_robustness_cases():
    base_case = next(c for c in build_matrix_cases() if c["case_id"] == "m02")
    base = matrix_input_value(base_case)["rows"]

    def rows():
        return copy.deepcopy(base)

    specs = []
    r = rows(); del r[2]["h2"]
    specs.append(("incomplete_missing_bit", r, UNKNOWN, "one row is missing an H2 value"))
    r = rows(); r[2]["h1"] = 2
    specs.append(("malformed_bit", r, INVALID, "one bit is outside the binary domain"))
    r = rows(); r[3]["label"] = "C"
    specs.append(("contradictory_duplicate_label", r, AMBIGUOUS, "two rows carry the same label"))
    r = rows()[:4]
    specs.append(("incomplete_missing_row", r, UNKNOWN, "only four of five expected rows are present"))
    r = rows(); r[2]["h2"] = None
    specs.append(("unknown_bit", r, UNKNOWN, "one bit is explicitly missing"))
    r = rows(); r[2]["h1"] = "true"
    specs.append(("malformed_type", r, INVALID, "one bit has a string type"))
    cases = []
    for i, (name, row_values, expected, description) in enumerate(specs):
        cases.append({"case_id": f"x{i:02d}", "family": name, "rows": row_values,
                      "truth": "status:" + expected, "expected_status": expected,
                      "description": description,
                      "variant": {"labelset": "ABCDE", "colrev": False, "permute": False, "yesno": False, "csv": False}})
    return cases


def robustness_input_value(case):
    return {"rows": copy.deepcopy(case["rows"]), "expected_rows": 5}


def load_robustness_bus(case):
    bus = CognitiveBus()
    input_obj = bus.put(
        "binary_row_set", robustness_input_value(case),
        provenance={"source": "robustness_evidence", "case_id": case["case_id"], "description": case["description"]},
        context_id=case["case_id"])
    events = auto_dispatch(bus)
    return bus, input_obj, events


def run_auto_robustness_case(case):
    started = time.perf_counter()
    bus, input_obj, events = load_robustness_bus(case)
    runtime_ms = (time.perf_counter() - started) * 1000
    decision = decision_from_matrix_bus(bus, input_obj)
    expected = "status:" + case["expected_status"]
    return {"case_id": case["case_id"], "family": case["family"], "expected_status": case["expected_status"],
            "decision": decision["decision"], "status": decision["status"], "answer_correct": decision["decision"] == expected,
            "input_status": bus.effective_status(input_obj), "events": events,
            "n_dispatch_events": len(events), "runtime_ms": round(runtime_ms, 3), "snapshot": bus.snapshot()}


def run_stale_invalidation_probe():
    case = next(c for c in build_matrix_cases() if c["case_id"] == "m02")
    bus, input_obj, first_events = load_matrix_bus(case)
    initial = decision_from_matrix_bus(bus, input_obj)
    rows = matrix_input_value(case)["rows"]
    for row in rows:
        if row["label"] == "C":
            row["h2"] = row["h1"]
        if row["label"] == "D":
            row["h2"] = not row["h1"]
    proposal = propose_premise_update(bus, input_obj.object_id, {"rows": rows, "expected_rows": 5},
                                      {"proposer": "experiment_owner", "rationale": "corrected evidence arrives after initial dispatch",
                                       "source": "controlled_staleness_probe"})
    updated = apply_update_proposal(bus, proposal)
    stale_before = bus.effective_status(bus.binding("decision.unique"))
    second_events = auto_dispatch(bus)
    refreshed = decision_from_matrix_bus(bus, input_obj)
    return {"case_id": case["case_id"], "initial_decision": initial["decision"],
            "initial_output_version": initial["output_version"],
            "updated_input_version": updated.version,
            "status_after_premise_update": stale_before,
            "invalidated": stale_before == "stale",
            "refreshed_decision": refreshed["decision"], "refreshed_output_version": refreshed["output_version"],
            "refresh_correct": refreshed["decision"] == "D",
            "first_events": first_events, "second_events": second_events,
            "final_snapshot": bus.snapshot()}


def build_stale_hybrid_case():
    case0 = next(c for c in build_matrix_cases() if c["case_id"] == "m02")
    bus, input_obj, events = load_matrix_bus(case0)
    rows = matrix_input_value(case0)["rows"]
    for row in rows:
        if row["label"] == "C":
            row["h2"] = row["h1"]
        if row["label"] == "D":
            row["h2"] = not row["h1"]
    proposal = propose_premise_update(bus, input_obj.object_id, {"rows": rows, "expected_rows": 5},
                                      {"proposer": "experiment_owner", "rationale": "corrected evidence arrives after initial dispatch",
                                       "source": "controlled_staleness_probe"})
    apply_update_proposal(bus, proposal)
    stale_case = {"case_id": "xstale", "family": "stale_dependent_state", "rows": rows,
                  "truth": "status:stale", "expected_status": "stale",
                  "description": "premise updated after dispatch but before recomputation",
                  "variant": dict(case0["variant"])}
    return stale_case, bus, input_obj, events


def build_composition_cases():
    def case(cid, rows):
        diffs = [r["label"] for r in rows if r["h1"] != r["h2"]]
        truth = diffs[0] if len(diffs) == 1 else "none"
        return {"case_id": cid, "family": "composition", "rows": rows, "truth": truth,
                "variant": {"labelset": "".join(r["label"] for r in rows), "colrev": False,
                            "permute": False, "yesno": False, "csv": False}}
    return [
        case("c00", [{"label": "X", "h1": False, "h2": False, "diff": False, "canonical_index": 0},
                     {"label": "Y", "h1": True, "h2": False, "diff": True, "canonical_index": 1},
                     {"label": "Z", "h1": True, "h2": True, "diff": False, "canonical_index": 2}]),
        case("c01", [{"label": chr(80 + i), "h1": i == 3, "h2": False, "diff": i == 3, "canonical_index": i} for i in range(7)]),
        case("c02", [{"label": chr(70 + i), "h1": i % 2 == 0, "h2": i % 2 == 0, "diff": False, "canonical_index": i} for i in range(6)]),
        case("c03", [{"label": "R", "h1": True, "h2": False, "diff": True, "canonical_index": 0},
                     {"label": "S", "h1": False, "h2": False, "diff": False, "canonical_index": 1},
                     {"label": "T", "h1": False, "h2": True, "diff": True, "canonical_index": 2},
                     {"label": "U", "h1": True, "h2": True, "diff": False, "canonical_index": 3}]),
    ]


def rule_problem_input(task):
    return {"rules": copy.deepcopy(task["rules"]),
            "observations": copy.deepcopy(task["observations"]),
            "queries": [{"label": q["label"], "x": q["x"], "y": q["y"]} for q in task["queries"]]}


def build_rule_runtime_tasks():
    tasks = build_epistemic_tasks()
    selected = []
    for family in ("need_query", "already_resolved", "query_irrelevant"):
        selected.extend([t for t in tasks if t["family"] == family][:2])
    return selected


def run_auto_rule_task(task):
    started = time.perf_counter()
    bus = CognitiveBus()
    input_obj = bus.put("rule_epistemic_problem", rule_problem_input(task),
                        provenance={"source": "structured_rule_problem", "task_id": task["task_id"]},
                        context_id=task["task_id"])
    events = auto_dispatch(bus)
    runtime_ms = (time.perf_counter() - started) * 1000
    result = epistemic_decision_from_bus(bus, input_obj)
    correct = result.get("choice") == task["truth"]
    return {"task_id": task["task_id"], "family": task["family"], "truth": task["truth"],
            "choice": result.get("choice"), "status": result.get("status"), "answer_correct": correct,
            "events": events, "n_dispatch_events": len(events), "runtime_ms": round(runtime_ms, 3),
            "input_status": bus.effective_status(input_obj), "snapshot": bus.snapshot()}


def rate(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _cost(rows, latency_key="latency_s", token_key="output_tokens"):
    return {"mean_latency_s": mean(r.get(latency_key) for r in rows),
            "mean_output_tokens": mean(r.get(token_key) for r in rows)}


def summarize_epi5(payload):
    out = {}
    direct = payload.get("matrix_direct", [])
    if direct:
        out["NEURAL_DIRECT"] = {"n": len(direct), "accuracy": rate(r["answer_correct"] for r in direct),
                                "format_valid_rate": rate(r["format_valid"] for r in direct), **_cost(direct)}
    inv = payload.get("matrix_invocations", [])
    if inv:
        invoked = [r for r in inv if r.get("action") == "compare"]
        out["TOOL_CONTEXT"] = {
            "n": len(inv), "accuracy": rate(r.get("context_correct", False) for r in inv),
            "invoke_rate": rate(r.get("action") == "compare" for r in inv),
            "first_format_valid_rate": rate(r.get("first_format_valid") for r in inv),
            "operand_exact_rate_invoked": rate(r.get("operand_exact") for r in invoked),
            "row_copy_accuracy_invoked": mean(r.get("row_copy_accuracy") for r in invoked),
            "retranscription_fidelity_invoked": rate(r.get("context_final") == r.get("comparator_unique") for r in invoked),
            "family_accuracy": {fam: rate(r.get("context_correct", False) for r in inv if r["family"] == fam)
                                 for fam in sorted(set(r["family"] for r in inv))},
            "mean_latency_s": mean(r.get("tool_context_total_latency_s") for r in inv),
            "mean_output_tokens": mean(r.get("tool_context_total_output_tokens") for r in inv),
        }
        fresh = [r for r in invoked if r.get("fresh_final") is not None]
        out["TOOL_FRESH"] = {"n": len(fresh), "readout_accuracy": rate(r.get("fresh_correct", False) for r in fresh),
                             "copy_fidelity": rate(r.get("fresh_final") == r.get("comparator_unique") for r in fresh)}
        out["BOUND_INVOKE"] = {
            "n": len(inv), "accuracy": rate(r.get("bound_correct", False) for r in inv),
            "preservation_rate_invoked": rate(r.get("bound_final") == r.get("comparator_unique") for r in invoked),
            "family_accuracy": {fam: rate(r.get("bound_correct", False) for r in inv if r["family"] == fam)
                                 for fam in sorted(set(r["family"] for r in inv))},
            "mean_latency_s": mean(r.get("first_latency_s") for r in inv),
            "mean_output_tokens": mean(r.get("first_output_tokens") for r in inv),
        }
    auto = payload.get("matrix_auto", [])
    if auto:
        out["AUTO_DISPATCH"] = {"n": len(auto), "accuracy": rate(r["answer_correct"] for r in auto),
                                "mean_runtime_ms": mean(r["runtime_ms"] for r in auto),
                                "mean_dispatch_events": mean(r["n_dispatch_events"] for r in auto),
                                "neural_output_tokens": 0,
                                "unnecessary_dispatch_count": sum(max(0, r["n_dispatch_events"] - 3) for r in auto),
                                "unexpected_dispatch_case_count": sum(r["n_dispatch_events"] != 3 for r in auto),
                                "provenance_complete_rate": rate(all(e.get("inputs") for e in r["events"]) for r in auto)}
    hybrid = payload.get("matrix_hybrid", [])
    if hybrid:
        out["HYBRID"] = {"n": len(hybrid), "accuracy": rate(r["answer_correct"] for r in hybrid),
                         "format_valid_rate": rate(r["format_valid"] for r in hybrid),
                         "state_consistent_rate": rate(r["state_consistent"] for r in hybrid if r.get("state_consistent") is not None),
                         **_cost(hybrid)}
    robauto = payload.get("robustness_auto", [])
    if robauto:
        out["ROBUSTNESS_AUTO"] = {"n": len(robauto), "status_accuracy": rate(r["answer_correct"] for r in robauto),
                                  "mean_runtime_ms": mean(r["runtime_ms"] for r in robauto),
                                  "undispatched_rate": rate(r["n_dispatch_events"] == 0 for r in robauto)}
    robhybrid = payload.get("robustness_hybrid", [])
    if robhybrid:
        out["ROBUSTNESS_HYBRID"] = {"n": len(robhybrid), "status_accuracy": rate(r["answer_correct"] for r in robhybrid),
                                    "format_valid_rate": rate(r["format_valid"] for r in robhybrid), **_cost(robhybrid)}
    stale = payload.get("stale_probe")
    if stale:
        out["STALE_INVALIDATION"] = {"invalidated": stale["invalidated"], "refresh_correct": stale["refresh_correct"],
                                     "initial_decision": stale["initial_decision"], "refreshed_decision": stale["refreshed_decision"]}
    shy = payload.get("stale_hybrid", [])
    if shy:
        out["STALE_HYBRID"] = {"n": len(shy), "status_accuracy": rate(r["answer_correct"] for r in shy),
                               "format_valid_rate": rate(r["format_valid"] for r in shy), **_cost(shy)}
    comp = payload.get("composition_auto", [])
    if comp:
        out["COMPOSITION_AUTO"] = {"n": len(comp), "accuracy": rate(r["answer_correct"] for r in comp),
                                   "row_counts": sorted(set(len(r.get("rows", [])) for r in comp))}
    edirect = payload.get("epistemic_direct", [])
    if edirect:
        out["EPISTEMIC_NEURAL_DIRECT"] = {"n": len(edirect), "accuracy": rate(r["answer_correct"] for r in edirect),
                                          "format_valid_rate": rate(r["format_valid"] for r in edirect), **_cost(edirect)}
    eauto = payload.get("epistemic_auto", [])
    if eauto:
        out["EPISTEMIC_AUTO"] = {"n": len(eauto), "mechanical_state_accuracy": rate(r["auto_correct"] for r in eauto),
                                 "mean_runtime_ms": mean(r["runtime_ms"] for r in eauto),
                                 "family_accuracy": {fam: rate(r["auto_correct"] for r in eauto if r["family"] == fam)
                                                      for fam in sorted(set(r["family"] for r in eauto))},
                                 "neural_output_tokens": 0}
    ehybrid = payload.get("epistemic_hybrid", [])
    if ehybrid:
        out["EPISTEMIC_HYBRID"] = {"n": len(ehybrid), "accuracy": rate(r["answer_correct"] for r in ehybrid),
                                   "format_valid_rate": rate(r["format_valid"] for r in ehybrid),
                                   "family_accuracy": {fam: rate(r["answer_correct"] for r in ehybrid if r["family"] == fam)
                                                        for fam in sorted(set(r["family"] for r in ehybrid))},
                                   **_cost(ehybrid)}
    rules = payload.get("rule_runtime", [])
    if rules:
        out["RULE_CONSISTENCY_AUTO"] = {"n": len(rules), "accuracy": rate(r["answer_correct"] for r in rules),
                                        "mean_runtime_ms": mean(r["runtime_ms"] for r in rules),
                                        "family_accuracy": {fam: rate(r["answer_correct"] for r in rules if r["family"] == fam)
                                                             for fam in sorted(set(r["family"] for r in rules))}}
    return out
