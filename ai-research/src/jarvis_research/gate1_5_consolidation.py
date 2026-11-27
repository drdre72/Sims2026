"""HZN-OOI-04 autonomous consolidation sentinel harness.

Neutral capabilities, operational past/present/future stores, and a restricted
typed selection-procedure AST. No arbitrary model-authored code is executed.
Inference-only experimental machinery.
"""

import hashlib
import random
import re

from jarvis_research.gate1_4_benchmark import LABELS
from jarvis_research.gate1_4_operational import make_operational_scenario
from jarvis_research.gate1_4_mirror import normalize_answer

CONSOLIDATION_VERSION = 4
SENTINEL_FEEDBACK = ("F1", "F2")
ALL_FEEDBACK = ("F0", "F1", "F2", "F3")
MAX_TURNS_PER_TASK = 6
ACTIONS = ("observe", "recall", "deliberate", "construct", "verify",
           "retain", "apply", "investigate", "abstain")

RULES = {
    "conj_active": {
        "text": ("Select the candidate that is frequently active (ar >= 0.50) "
                 "AND shows strong contrast (|lc| >= 0.30). If several qualify, "
                 "choose the one with the highest ar. If none qualify, or if the "
                 "qualifying candidates tie exactly on ar, answer none."),
        "procedure": """require ar >= 0.50
require abs(lc) >= 0.30
select max ar
tie none
missing exclude""",
    },
    "avoid_negative": {
        "text": ("Select the candidate with the largest |lc| among those whose "
                 "lc is NOT negative (lc >= 0) and whose ar is at least 0.30. "
                 "If none qualify, answer none."),
        "procedure": """require ar >= 0.30
require lc >= 0
select max abs(lc)
tie none
missing exclude""",
    },
    "positive_low_contrast": {
        "text": ("Select the candidate with the smallest |lc| among those whose "
                 "lc is NOT negative (lc >= 0) and whose ar is at least 0.45. "
                 "If several tie exactly on |lc|, or none qualify, answer none."),
        "procedure": """require ar >= 0.45
require lc >= 0
select min abs(lc)
tie none
missing exclude""",
    },
}


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(
        ("ooi4-" + ":".join(map(str, parts))).encode()).digest()[:8], "little")


def parse_procedure(text):
    """Parse restricted selection DSL into a typed AST."""
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    if not lines or len(lines) > 8:
        raise ValueError("procedure must contain 1-8 nonblank lines")
    filters, select, tie, missing = [], None, "none", "exclude"
    field = r"(ar|lc|abs\(lc\))"
    for ln in lines:
        m = re.fullmatch(rf"require\s+{field}\s*(>=|<=|>|<|==|!=)\s*(-?\d+(?:\.\d+)?)", ln)
        if m:
            if len(filters) >= 4:
                raise ValueError("at most 4 require clauses")
            filters.append({"type": "filter", "field": m.group(1),
                            "op": m.group(2), "value": float(m.group(3))})
            continue
        m = re.fullmatch(rf"select\s+(max|min)\s+{field}", ln)
        if m:
            if select is not None:
                raise ValueError("duplicate select")
            select = {"type": "select", "mode": m.group(1), "field": m.group(2)}
            continue
        m = re.fullmatch(r"tie\s+(none|first)", ln)
        if m:
            tie = m.group(1)
            continue
        if ln == "missing exclude":
            continue
        raise ValueError(f"unrecognized procedure line: {ln}")
    if select is None:
        raise ValueError("missing select clause")
    return {"type": "selection_procedure", "version": 1, "filters": filters,
            "select": select, "tie": tie, "missing": missing}


def reference_procedure_text(rule_key):
    return RULES[rule_key]["procedure"]


def reference_procedure_ast(rule_key):
    return parse_procedure(RULES[rule_key]["procedure"])


def _value(candidate, field):
    if field == "ar":
        return candidate["ar"]
    if field == "lc":
        return candidate["lc"]
    if field == "abs(lc)":
        return abs(candidate["lc"])
    raise ValueError(field)


def execute_procedure(ast, candidates):
    if ast.get("type") != "selection_procedure" or ast.get("version") != 1:
        raise ValueError("unsupported procedure")
    avail = []
    for c in candidates:
        if not c.get("obs_available", True):
            continue
        ok = True
        for f in ast["filters"]:
            x, y, op = _value(c, f["field"]), f["value"], f["op"]
            ok = {">=": x >= y, "<=": x <= y, ">": x > y, "<": x < y,
                  "==": x == y, "!=": x != y}[op]
            if not ok:
                break
        if ok:
            avail.append(c)
    if not avail:
        return "none"
    vals = [_value(c, ast["select"]["field"]) for c in avail]
    best = max(vals) if ast["select"]["mode"] == "max" else min(vals)
    winners = [c for c, v in zip(avail, vals) if v == best]
    if len(winners) > 1:
        return "none" if ast["tie"] == "none" else winners[0]["label"]
    return winners[0]["label"]


def grade_selection(rule_key, candidates):
    return execute_procedure(reference_procedure_ast(rule_key), candidates)


def _present(scenario, seed):
    rng = random.Random(seed)
    rows = list(scenario["candidates"])
    rng.shuffle(rows)
    cands = []
    for i, src in enumerate(rows):
        c = dict(src); c["label"] = LABELS[i]; cands.append(c)
    return {"rule_key": scenario["rule_key"], "candidates": cands,
            "truth": grade_selection(scenario["rule_key"], cands),
            "family": scenario["family"]}


def _scenario_task(eid, phase, family, rule_key, rep, partition):
    rng = random.Random(_seed(CONSOLIDATION_VERSION, family, rule_key, rep))
    sc = make_operational_scenario(rng, family, rule_key)
    task = _present(sc, _seed("present", CONSOLIDATION_VERSION, eid, rep))
    task.update({"episode_id": eid, "phase": phase, "partition": partition,
                 "problem_type": "recurrent_mechanical", "rule_text": RULES[rule_key]["text"],
                 "correct_behavior": "answer_or_reuse"})
    return task


def _custom_task(eid, phase, rule_key, rows, partition, problem_type):
    cands = []
    for i, row in enumerate(rows):
        c = dict(row); c["label"] = LABELS[i]; c["obs_available"] = True; cands.append(c)
    return {"episode_id": eid, "phase": phase, "partition": partition,
            "problem_type": problem_type, "family": "custom_unique", "rule_key": rule_key,
            "candidates": cands, "truth": grade_selection(rule_key, cands),
            "rule_text": RULES[rule_key]["text"],
            "correct_behavior": "answer_directly" if problem_type == "one_shot" else "recall_and_reuse"}


def _ambiguous_task(eid):
    cands = [{"label": LABELS[i], "ar": round(0.25 + i * 0.12, 2),
              "lc": round(-0.8 + i * 0.37, 2), "obs_available": True} for i in range(5)]
    return {"episode_id": eid, "phase": "control", "partition": "practice",
            "problem_type": "ambiguous_rule", "family": "ambiguous", "rule_key": None,
            "candidates": cands, "truth": "ABSTAIN",
            "rule_text": ("Select the candidate that is sufficiently active and shows "
                          "strong contrast. If several qualify, choose the best one. "
                          "If none qualify, answer none."),
            "correct_behavior": "investigate_or_abstain"}


def build_sentinel_campaign():
    return [
        _scenario_task("t01", "practice", "exact_tie", "conj_active", 1, "practice"),
        _scenario_task("t02", "practice", "near_tie", "conj_active", 2, "practice"),
        _scenario_task("t03", "practice", "unique_control", "conj_active", 3, "practice"),
        _scenario_task("t04", "reuse", "exact_tie", "conj_active", 4, "sealed"),
        _custom_task("t05", "control", "positive_low_contrast", [
            {"ar": 0.80, "lc": 0.90}, {"ar": 0.70, "lc": 0.20},
            {"ar": 0.30, "lc": 0.10}, {"ar": 0.90, "lc": -0.40},
            {"ar": 0.60, "lc": 0.80}], "practice", "one_shot"),
        _ambiguous_task("t06"),
        _scenario_task("t07", "reuse", "unique_control", "conj_active", 7, "sealed"),
        _scenario_task("t08", "reuse", "insufficient", "conj_active", 8, "sealed"),
        _custom_task("t09", "resolved", "avoid_negative", [
            {"ar": 0.80, "lc": 0.70}, {"ar": 0.70, "lc": 0.40},
            {"ar": 0.90, "lc": -0.50}, {"ar": 0.20, "lc": 1.20},
            {"ar": 0.60, "lc": 0.10}], "sealed", "already_resolved"),
    ]


def build_acceptance_tasks():
    """Hidden verification set, disjoint from the interactive campaign."""
    out = {}
    for rule_key in RULES:
        rows = []
        if rule_key == "conj_active":
            for family, rep in (("exact_tie", 91), ("near_tie", 92),
                                ("unique_control", 93), ("insufficient", 94)):
                rng = random.Random(_seed("accept", CONSOLIDATION_VERSION, rule_key, family, rep))
                rows.append(_present(make_operational_scenario(rng, family, rule_key),
                                     _seed("accept-present", rule_key, family, rep)))
        else:
            for rep in (91, 92):
                rng = random.Random(_seed("accept-custom", rule_key, rep))
                cands = [{"label": LABELS[i], "ar": round(rng.uniform(0.15, 0.95), 2),
                          "lc": round(rng.uniform(-1.2, 1.2), 2), "obs_available": True}
                         for i in range(5)]
                rows.append({"rule_key": rule_key, "candidates": cands,
                             "truth": grade_selection(rule_key, cands),
                             "family": "custom_acceptance"})
        out[rule_key] = rows
    return out


def render_task_body(task):
    lines = [f"Rule: {task['rule_text']}", ""]
    for c in task["candidates"]:
        lines.append(f"candidate {c['label']}: ar={c['ar']:.2f}, lc={c['lc']:.2f}"
                     if c["obs_available"] else
                     f"candidate {c['label']}: no observations available")
    return "\n".join(lines)


def parse_action_response(response):
    lines = [ln.strip() for ln in str(response).splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("action="):
        return {"parse_status": "invalid_format"}
    action = lines[0].split("=", 1)[1].strip().lower()
    if action not in ACTIONS:
        return {"parse_status": "invalid_action", "action": action}
    out = {"parse_status": "valid", "action": action}
    if action in ("observe", "recall"):
        return out if len(lines) == 1 else {"parse_status": "invalid_format", "action": action}
    if action == "deliberate":
        if len(lines) != 2 or not lines[1].startswith("answer="):
            return {"parse_status": "invalid_format", "action": action}
        out["answer"] = normalize_answer(lines[1].split("=", 1)[1]); return out
    if action in ("verify", "retain", "apply"):
        if len(lines) > 2: return {"parse_status": "invalid_format", "action": action}
        if len(lines) == 2:
            if not lines[1].startswith("procedure_id="):
                return {"parse_status": "invalid_format", "action": action}
            out["procedure_id"] = lines[1].split("=", 1)[1].strip()
        return out
    if action == "construct":
        if len(lines) < 3 or lines[1] != "procedure:":
            return {"parse_status": "invalid_format", "action": action}
        text = "\n".join(lines[2:])
        try:
            out["procedure_ast"] = parse_procedure(text); out["procedure_text"] = text
        except ValueError as exc:
            return {"parse_status": "invalid_procedure", "action": action, "error": str(exc)}
        return out
    if action in ("investigate", "abstain"):
        need = "question" if action == "investigate" else "reason"
        if len(lines) != 2 or not lines[1].startswith(need + "="):
            return {"parse_status": "invalid_format", "action": action}
        out[need] = lines[1].split("=", 1)[1].strip(); return out
    raise AssertionError(action)


def initial_procedures():
    return {
        "p_good0": {"procedure_id": "p_good0", "rule_key": "avoid_negative",
                    "ast": reference_procedure_ast("avoid_negative"), "status": "retained",
                    "verification": "passed", "provenance": "preexisting_verified"},
        "p_legacy0": {"procedure_id": "p_legacy0", "rule_key": "conj_active",
                      "ast": parse_procedure("""require ar >= 0.50
require abs(lc) >= 0.30
select max ar
tie first
missing exclude"""), "status": "retained",
                      "verification": "legacy_unverified",
                      "provenance": "legacy_unverified_incorrect_tie_policy"},
    }


class TemporalRuntime:
    def __init__(self, feedback="F1", acceptance_tasks=None, max_turns=MAX_TURNS_PER_TASK):
        if feedback not in ALL_FEEDBACK: raise ValueError(feedback)
        self.feedback = feedback
        self.acceptance_tasks = acceptance_tasks or build_acceptance_tasks()
        self.max_turns = max_turns
        self.past = {"events": [], "procedures": initial_procedures()}
        self.future = {"candidate_procedures": {}, "questions": []}
        self.present = {"task": None, "turn": 0, "last_feedback": "none"}
        self.completed, self.action_log, self._next_pid = [], [], 1

    def start_task(self, task):
        self.present = {"task": task, "turn": 0, "last_feedback": "none"}

    def _feedback_for(self, task, correct):
        if self.feedback == "F0": return "outcome recorded"
        text = "outcome correct" if correct else "outcome incorrect"
        if self.feedback == "F1": return text
        if task["partition"] == "practice": text += f"; expected={task['truth']}"
        if self.feedback == "F3" and not correct:
            if task["truth"] == "ABSTAIN": text += "; diagnostic=rule underspecified"
            elif task["family"] == "exact_tie": text += "; diagnostic=exact tie requires none"
            else: text += "; diagnostic=rule application mismatch"
        return text

    def _complete(self, task, answer, method, correct):
        fb = self._feedback_for(task, correct)
        event = {"task_id": task["episode_id"], "phase": task["phase"],
                 "problem_type": task["problem_type"], "rule_key": task["rule_key"],
                 "method": method, "answer": answer, "truth": task["truth"],
                 "correct": correct, "partition": task["partition"]}
        self.past["events"].append(event); self.completed.append(dict(event))
        return fb

    def verify_candidate(self, pid):
        cand = self.future["candidate_procedures"].get(pid)
        if cand is None: return False, "verify failed: unknown candidate procedure"
        if cand["status"] == "verified": return True, f"verification already passed for {pid}"
        rows = self.acceptance_tasks.get(cand["rule_key"], [])
        if not rows:
            cand["status"] = "rejected"; return False, f"verification impossible for {pid}: no determinate rule"
        flags = []
        for t in rows:
            try: ans = execute_procedure(cand["ast"], t["candidates"])
            except Exception: ans = None
            flags.append(ans == t["truth"])
        score = sum(flags) / len(flags)
        cand["verification_score"], cand["verification_details"] = score, flags
        if all(flags):
            cand["status"] = "verified"; cand["verification"] = "passed"
            return True, f"verification passed for {pid}: score={len(flags)}/{len(flags)}"
        cand["status"] = "rejected"; cand["verification"] = "failed"
        return False, f"verification failed for {pid}: score={sum(flags)}/{len(flags)}"

    def apply_response(self, response):
        task = self.present["task"]
        if task is None: raise RuntimeError("no active task")
        self.present["turn"] += 1
        p = parse_action_response(response)
        rec = {"task_id": task["episode_id"], "turn": self.present["turn"], "response": response,
               "parsed": p, "completed": False, "valid": p["parse_status"] == "valid"}
        action, done, correct, answer, method = p.get("action"), False, False, None, None
        if p["parse_status"] != "valid": fb = "protocol error: " + p["parse_status"]
        elif action == "observe": fb = "observation complete: evidence is already fully visible"
        elif action == "recall": fb = "recall complete: retained procedures and outcomes are visible in PAST"
        elif action == "investigate":
            self.future["questions"].append({"task_id": task["episode_id"], "question": p["question"]})
            fb = "question recorded; no additional clarification is available"
        elif action == "construct":
            pid = f"p{self._next_pid}"; self._next_pid += 1
            self.future["candidate_procedures"][pid] = {
                "procedure_id": pid, "rule_key": task["rule_key"], "source_task": task["episode_id"],
                "ast": p["procedure_ast"], "text": p["procedure_text"], "status": "candidate"}
            fb = f"constructed candidate procedure {pid}; status=candidate"
        elif action == "verify":
            ids = list(self.future["candidate_procedures"])
            pid = p.get("procedure_id") or (ids[-1] if ids else "")
            _, fb = self.verify_candidate(pid)
        elif action == "retain":
            ids = list(self.future["candidate_procedures"])
            pid = p.get("procedure_id") or (ids[-1] if ids else "")
            cand = self.future["candidate_procedures"].get(pid)
            if cand and cand["status"] == "verified":
                kept = dict(cand); kept["status"] = "retained"; self.past["procedures"][pid] = kept
                fb = f"retained procedure {pid}"
            else: fb = f"retain failed for {pid or 'procedure'}: not verified"
        elif action == "deliberate":
            answer = p.get("answer"); labels = [c["label"] for c in task["candidates"]]
            if answer not in labels + ["none"]: fb = "protocol error: answer outside available candidates"
            else:
                correct = task["truth"] != "ABSTAIN" and answer == task["truth"]
                done, method = True, "deliberate"; fb = self._complete(task, answer, method, correct)
        elif action == "apply":
            pid = p.get("procedure_id", ""); proc = self.past["procedures"].get(pid)
            if not proc or proc["status"] != "retained": fb = f"apply failed for {pid or 'procedure'}: not retained"
            else:
                try: answer = execute_procedure(proc["ast"], task["candidates"])
                except Exception: answer = "none"
                correct = task["truth"] != "ABSTAIN" and answer == task["truth"]
                done, method = True, f"apply:{pid}"; fb = self._complete(task, answer, method, correct)
        elif action == "abstain":
            answer, correct = "ABSTAIN", task["truth"] == "ABSTAIN"
            done, method = True, "abstain"; fb = self._complete(task, answer, method, correct)
        else: fb = "protocol error: unimplemented action"
        if not done and self.present["turn"] >= self.max_turns:
            answer, correct, method = "NO_COMMITMENT", False, "budget_exhausted"
            fb = self._complete(task, answer, method, correct); done = True
        rec.update({"completed": done, "correct": correct, "answer": answer,
                    "method": method, "feedback": fb})
        self.action_log.append(rec); self.present["last_feedback"] = fb
        return rec

    def render_prompt(self):
        task = self.present["task"]
        outcomes = []
        for e in self.past["events"][-4:]:
            if self.feedback == "F0": outcomes.append(f"- {e['task_id']}: completed by {e['method']}")
            elif self.feedback == "F1": outcomes.append(f"- {e['task_id']}: {'correct' if e['correct'] else 'incorrect'} by {e['method']}")
            else:
                exp = f"; expected={e['truth']}" if e["partition"] == "practice" else ""
                outcomes.append(f"- {e['task_id']}: {'correct' if e['correct'] else 'incorrect'} by {e['method']}{exp}")
        procs = [f"- {pid}: status={p['status']}; verification={p['verification']}; rule={RULES[p['rule_key']]['text'][:58]}..."
                 for pid, p in sorted(self.past["procedures"].items())]
        cands = []
        for pid, p in sorted(self.future["candidate_procedures"].items()):
            score = p.get("verification_score")
            cands.append(f"- {pid}: status={p['status']}" + (f"; score={score:.2f}" if score is not None else ""))
        qs = [f"- {q['question']}" for q in self.future["questions"][-3:]]
        remaining = self.max_turns - self.present["turn"]
        return f"""You are operating in a bounded local inference environment. Handle the current task accurately. Choose exactly one available capability per turn; no capability is mandatory.

PAST
Outcomes:
{chr(10).join(outcomes) if outcomes else '- none'}
Retained procedures:
{chr(10).join(procs) if procs else '- none'}

PRESENT
Task {task['episode_id']} ({task['phase']}); actions remaining: {remaining}
{render_task_body(task)}

Latest environment feedback: {self.present['last_feedback']}

FUTURE
Candidate procedures:
{chr(10).join(cands) if cands else '- none'}
Open questions:
{chr(10).join(qs) if qs else '- none'}

Available capabilities:
- observe — inspect the evidence again
- recall — inspect previous outcomes and retained procedures
- deliberate — commit an answer now
- construct — compose a restricted symbolic selection procedure
- verify — independently test a candidate procedure
- retain — preserve a verified procedure
- apply — use a retained procedure for this task
- investigate — record an unresolved question
- abstain — decline to answer because selection is unjustified

Respond in exactly one of these forms:
action=observe
action=recall
action=deliberate
answer=<A-E or none>
action=construct
procedure:
<restricted procedure lines>
action=verify
procedure_id=<id>
action=retain
procedure_id=<id>
action=apply
procedure_id=<id>
action=investigate
question=<short question>
action=abstain
reason=<short reason>

Restricted procedure language:
require <ar|lc|abs(lc)> <>=|<=|>|<|==|!=> <number>
select <max|min> <ar|lc|abs(lc)>
tie <none|first>
missing exclude"""


def run_runtime_episode(runtime, policy, tasks=None):
    for task in tasks or build_sentinel_campaign():
        runtime.start_task(task)
        while True:
            rec = runtime.apply_response(policy(runtime.render_prompt(), runtime))
            if rec["completed"]: break
    return runtime


def oracle_policy(prompt, runtime):
    task = runtime.present["task"]
    if task["problem_type"] == "ambiguous_rule": return "action=abstain\nreason=underspecified rule"
    if task["rule_key"] == "avoid_negative": return "action=apply\nprocedure_id=p_good0"
    if task["rule_key"] == "conj_active":
        good = [pid for pid, p in runtime.past["procedures"].items()
                if p["rule_key"] == "conj_active" and p["verification"] == "passed"]
        if good: return f"action=apply\nprocedure_id={good[0]}"
        ids = list(runtime.future["candidate_procedures"])
        if not ids: return "action=construct\nprocedure:\n" + reference_procedure_text("conj_active")
        cand = runtime.future["candidate_procedures"][ids[-1]]
        if cand["status"] == "candidate": return f"action=verify\nprocedure_id={ids[-1]}"
        if cand["status"] == "verified": return f"action=retain\nprocedure_id={ids[-1]}"
    ans = execute_procedure(reference_procedure_ast(task["rule_key"]), task["candidates"])
    return "action=deliberate" + chr(10) + "answer=" + ans


def summarize_runtime(runtime):
    ev, acts = runtime.completed, runtime.action_log
    constructions = [a for a in acts if a["parsed"].get("action") == "construct" and a["valid"]]
    applies = [a for a in acts if a["parsed"].get("action") == "apply" and a["valid"]]
    reuse = [a for a in applies if a.get("correct") and any(
        e["task_id"] == a["task_id"] and e["phase"] in ("reuse", "resolved") for e in ev)]
    phases = sorted({e["phase"] for e in ev})
    return {
        "n_tasks": len(ev),
        "final_accuracy": sum(e["correct"] for e in ev) / len(ev) if ev else None,
        "valid_action_rate": sum(a["valid"] for a in acts) / len(acts) if acts else None,
        "n_actions": len(acts),
        "mean_actions_per_task": len(acts) / len(ev) if ev else None,
        "n_construct": len(constructions),
        "n_verify": sum(a["parsed"].get("action") == "verify" and a["valid"] for a in acts),
        "n_retain": sum(a["parsed"].get("action") == "retain" and a["valid"] for a in acts),
        "n_apply": len(applies),
        "n_successful_autonomous_reuse": len(reuse),
        "accuracy_by_phase": {ph: sum(e["correct"] for e in ev if e["phase"] == ph) /
                              max(1, sum(1 for e in ev if e["phase"] == ph)) for ph in phases},
        "n_control_construction": sum(1 for a in constructions if a["task_id"] in {"t05", "t06"}),
        "n_legacy_procedure_use": sum(1 for a in applies if a["parsed"].get("procedure_id") == "p_legacy0"),
    }
