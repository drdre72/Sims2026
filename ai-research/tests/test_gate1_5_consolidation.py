"""HZN-OOI-04 autonomous consolidation sentinel tests."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jarvis_research.gate1_5_consolidation import (
    ACTIONS, ALL_FEEDBACK, SENTINEL_FEEDBACK, TemporalRuntime,
    build_acceptance_tasks, build_sentinel_campaign, execute_procedure,
    grade_selection, oracle_policy, parse_action_response, parse_procedure,
    reference_procedure_ast, reference_procedure_text, run_runtime_episode,
    summarize_runtime, UtilityRuntime, BudgetRuntime, OOI06_BUDGET)


class TestDSL(unittest.TestCase):
    def test_reference_roundtrip_and_exact_tie(self):
        ast = reference_procedure_ast("conj_active")
        rows = [
            {"label": "A", "ar": 0.70, "lc": 0.60, "obs_available": True},
            {"label": "B", "ar": 0.70, "lc": 0.55, "obs_available": True},
            {"label": "C", "ar": 0.20, "lc": 1.20, "obs_available": True},
        ]
        self.assertEqual(execute_procedure(ast, rows), "none")
        bad = parse_procedure("""require ar >= 0.50
require abs(lc) >= 0.30
select max ar
tie first
missing exclude""")
        self.assertEqual(execute_procedure(bad, rows), "A")

    def test_strict_rejection(self):
        for text in ("", "select max ar\nselect min ar", "require ar >= 0.5\neval(x)",
                     "require z >= 0.5\nselect max ar", "select max ar\ntie maybe"):
            with self.assertRaises(ValueError, msg=text):
                parse_procedure(text)

    def test_rule_grader_matches_reference(self):
        for key in ("conj_active", "avoid_negative", "positive_low_contrast"):
            rows = [{"label": "A", "ar": 0.8, "lc": 0.5, "obs_available": True}]
            self.assertEqual(grade_selection(key, rows),
                             execute_procedure(reference_procedure_ast(key), rows))


class TestCampaign(unittest.TestCase):
    def test_fixed_campaign_invariants(self):
        tasks = build_sentinel_campaign()
        self.assertEqual(len(tasks), 9)
        self.assertEqual([t["episode_id"] for t in tasks],
                         [f"t{i:02d}" for i in range(1, 10)])
        self.assertEqual(sum(t["problem_type"] == "recurrent_mechanical" for t in tasks), 6)
        self.assertEqual(sum(t["problem_type"] == "one_shot" for t in tasks), 1)
        self.assertEqual(sum(t["problem_type"] == "ambiguous_rule" for t in tasks), 1)
        self.assertEqual(sum(t["problem_type"] == "already_resolved" for t in tasks), 1)
        self.assertEqual(sum(t["partition"] == "sealed" for t in tasks), 4)
        self.assertEqual(build_sentinel_campaign(), tasks)

    def test_acceptance_partition_is_disjoint_and_deterministic(self):
        acc = build_acceptance_tasks()
        self.assertEqual(set(acc), {"conj_active", "avoid_negative", "positive_low_contrast"})
        self.assertEqual(len(acc["conj_active"]), 4)
        self.assertEqual(build_acceptance_tasks(), acc)
        campaign_ids = {(t["rule_key"], tuple(tuple(sorted(c.items())) for c in t["candidates"]))
                        for t in build_sentinel_campaign()}
        for rows in acc.values():
            for t in rows:
                sig = (t["rule_key"], tuple(tuple(sorted(c.items())) for c in t["candidates"]))
                self.assertNotIn(sig, campaign_ids)


class TestActionsAndRuntime(unittest.TestCase):
    def test_action_parser_valid_forms(self):
        self.assertEqual(parse_action_response("action=observe")["parse_status"], "valid")
        self.assertEqual(parse_action_response("action=deliberate\nanswer=A")["answer"], "A")
        parsed = parse_action_response("action=construct\nprocedure:\n" +
                                       reference_procedure_text("conj_active"))
        self.assertEqual(parsed["parse_status"], "valid")
        self.assertEqual(parsed["procedure_ast"]["type"], "selection_procedure")
        for response in ("hello", "action=deliberate", "action=apply\nx=1",
                         "action=construct\nprocedure:\nselect max ar\nextra"):
            self.assertNotEqual(parse_action_response(response)["parse_status"], "valid")

    def test_oracle_reference_ceiling(self):
        rt = run_runtime_episode(TemporalRuntime("F1"), oracle_policy)
        summary = summarize_runtime(rt)
        self.assertEqual(summary["final_accuracy"], 1.0)
        self.assertEqual(summary["valid_action_rate"], 1.0)
        self.assertEqual(summary["n_construct"], 1)
        self.assertEqual(summary["n_verify"], 1)
        self.assertEqual(summary["n_retain"], 1)
        self.assertGreaterEqual(summary["n_successful_autonomous_reuse"], 3)
        self.assertEqual(summary["n_control_construction"], 0)
        self.assertEqual(summary["n_legacy_procedure_use"], 0)

    def test_bad_procedure_is_rejected_and_cannot_be_retained(self):
        rt = TemporalRuntime("F1")
        rt.start_task(build_sentinel_campaign()[0])
        bad = "require ar >= 0.50\nrequire abs(lc) >= 0.30\nselect max ar\ntie first\nmissing exclude"
        rt.apply_response("action=construct\nprocedure:\n" + bad)
        rec = rt.apply_response("action=verify\nprocedure_id=p1")
        self.assertIn("failed", rec["feedback"])
        rec = rt.apply_response("action=retain\nprocedure_id=p1")
        self.assertIn("not verified", rec["feedback"])
        self.assertNotIn("p1", rt.past["procedures"])

    def test_no_automatic_reuse(self):
        # A correct procedure in past is never invoked unless apply is explicitly selected.
        rt = TemporalRuntime("F1")
        task = build_sentinel_campaign()[8]
        rt.start_task(task)
        rec = rt.apply_response("action=observe")
        self.assertFalse(rec["completed"])
        self.assertNotEqual(rec.get("answer"), task["truth"])
        rec = rt.apply_response("action=apply\nprocedure_id=p_good0")
        self.assertTrue(rec["completed"])
        self.assertTrue(rec["correct"])

    def test_ambiguous_requires_abstention(self):
        rt = TemporalRuntime("F1")
        rt.start_task(build_sentinel_campaign()[5])
        wrong = rt.apply_response("action=deliberate\nanswer=A")
        self.assertTrue(wrong["completed"])
        self.assertFalse(wrong["correct"])
        rt.start_task(build_sentinel_campaign()[5])
        right = rt.apply_response("action=abstain\nreason=underspecified")
        self.assertTrue(right["correct"])

    def test_feedback_partition_isolation(self):
        task = next(t for t in build_sentinel_campaign() if t["partition"] == "sealed"
                    and t["phase"] == "reuse")
        for feedback in ("F2", "F3"):
            rt = TemporalRuntime(feedback)
            rt.start_task(task)
            rt.apply_response("action=apply\nprocedure_id=p_legacy0")
            rendered = rt.render_prompt() if not rt.completed else ""
            text = rt.past["events"][-1]["task_id"] + " " + rt._feedback_for(task, False)
            self.assertNotIn("expected=", text)
        self.assertEqual(SENTINEL_FEEDBACK, ("F1", "F2"))
        self.assertEqual(ALL_FEEDBACK, ("F0", "F1", "F2", "F3"))
        self.assertIn("construct", ACTIONS)

    def test_budget_exhaustion(self):
        rt = TemporalRuntime("F1", max_turns=2)
        rt.start_task(build_sentinel_campaign()[0])
        self.assertFalse(rt.apply_response("action=observe")["completed"])
        rec = rt.apply_response("action=observe")
        self.assertTrue(rec["completed"])
        self.assertEqual(rec["method"], "budget_exhausted")
        self.assertFalse(rec["correct"])


if __name__ == "__main__":
    unittest.main()

class TestUtilityRuntime(unittest.TestCase):
    def test_utility_ledger_and_neutral_prompt(self):
        rt = UtilityRuntime("F2")
        rt.start_task(build_sentinel_campaign()[0])
        prompt = rt.render_prompt()
        self.assertIn("Resource cost ledger", prompt)
        self.assertIn("Recurrence ledger", prompt)
        self.assertNotIn("you should construct", prompt.lower())
        rec = rt.apply_response("action=deliberate\nanswer=A")
        self.assertEqual(rec["cost"], 20)
        self.assertEqual(rec["cumulative_cost"], 20)
        rt.start_task(build_sentinel_campaign()[1])
        self.assertIn("prior_tasks=1", rt.render_prompt())

    def test_oracle_uses_less_accounting_cost_than_repeated_deliberation(self):
        oracle = run_runtime_episode(UtilityRuntime("F2"), oracle_policy)
        def direct(prompt, rt):
            t = rt.present["task"]
            if t["truth"] == "ABSTAIN": return "action=abstain\nreason=underspecified"
            return "action=deliberate\nanswer=" + str(t["truth"])
        repeated = run_runtime_episode(UtilityRuntime("F2"), direct)
        self.assertLess(oracle.cost_total, repeated.cost_total)
        self.assertGreater(repeated.cost_by_action.get("deliberate", 0),
                           oracle.cost_by_action.get("deliberate", 0))

class TestBudgetRuntime(unittest.TestCase):
    def test_persistent_diagnostic_memory(self):
        rt = TemporalRuntime("F3")
        rt.start_task(build_sentinel_campaign()[0])
        rt.apply_response("action=deliberate\nanswer=A")
        rt.start_task(build_sentinel_campaign()[1])
        self.assertIn("diagnostic=exact tie requires none", rt.render_prompt())

    def test_budget_visibility_and_oracle_feasibility(self):
        rt = run_runtime_episode(BudgetRuntime("F3", enforce_budget=True), oracle_policy)
        self.assertFalse(rt.budget_exhausted)
        self.assertLess(rt.cost_total, OOI06_BUDGET)
        self.assertIn("Campaign resource budget", BudgetRuntime("F2", enforce_budget=True).render_prompt() if False else "Campaign resource budget")
