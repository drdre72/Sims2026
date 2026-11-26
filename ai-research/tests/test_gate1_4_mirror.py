"""HZN-IR-12 mirror construction and grading tests."""
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jarvis_research.gate1_4_benchmark import (
    FAMILIES, grade_task, make_task, render_task)
from jarvis_research.gate1_4_mirror import (
    MIRROR_CAUSES, candidate_evaluation, grade_mirror_response,
    parse_mirror_response, permuted_relabel_task, reference_failure_cause,
    render_evidence_expansion, render_mirror_prompt)


class TestPermutedRelabel(unittest.TestCase):
    def test_nonidentity_deterministic_relabel(self):
        task = make_task(random.Random(3), "misleading_magnitude")
        a = permuted_relabel_task(task, 123)
        b = permuted_relabel_task(task, 123)
        self.assertEqual(a, b)
        self.assertNotEqual(a["label_map"],
                            {label: label for label in "ABCDE"})
        self.assertEqual([c["label"] for c in a["candidates"]],
                         list("ABCDE"))
        # Same multiset of observational evidence, new positional labels.
        key = lambda c: (c["ar"], c["lc"], c["obs_available"])
        self.assertEqual(sorted(map(key, task["candidates"])),
                         sorted(map(key, a["candidates"])))
        self.assertEqual(a["truth"], grade_task(a["rule_key"],
                                                a["candidates"]))

    def test_truth_tracks_evidence_not_original_label(self):
        task = make_task(random.Random(3), "misleading_magnitude")
        m0 = permuted_relabel_task(task, 999)
        if task["truth"]["answer"] != "none":
            old_truth = task["truth"]["answer"]
            expected_new = m0["label_map"][old_truth]
            self.assertEqual(m0["truth"]["answer"], expected_new)


class TestMirrorRendering(unittest.TestCase):
    def setUp(self):
        self.task = make_task(random.Random(4), "conflicting")

    def test_prompt_information_conditions(self):
        base = render_mirror_prompt(self.task, "C")
        self.assertIn("Previous answer: C", base)
        self.assertNotIn("Independent correctness verdict", base)
        self.assertNotIn("Deterministic evidence expansion", base)
        m2 = render_mirror_prompt(self.task, "C", verdict="incorrect")
        self.assertIn("Independent correctness verdict: incorrect", m2)
        self.assertNotIn("correct answer", m2.lower())
        m3 = render_mirror_prompt(self.task, "C", verdict="incorrect",
                                  expanded_evidence=True)
        self.assertIn("Deterministic evidence expansion", m3)
        self.assertIn("qualifies=", m3)
        # The mirror output contract must not conflict with the original
        # single-answer instruction.
        self.assertNotIn("Reply with ONLY the candidate letter", m3)
        self.assertIn("cause:", m3)
        self.assertIn("answer:", m3)

    def test_original_render_unchanged(self):
        txt = render_task(self.task)
        self.assertIn("Reply with ONLY the candidate letter", txt)

    def test_expansion_matches_grader_qualifiers(self):
        for seed in range(80):
            task = make_task(random.Random(seed), FAMILIES[seed % len(FAMILIES)])
            ev_q = sorted(c["label"] for c in task["candidates"]
                          if candidate_evaluation(task["rule_key"], c)["qualifies"])
            self.assertEqual(ev_q, task["truth"]["qualifying"], seed)
            txt = render_evidence_expansion(task["rule_key"], task["candidates"])
            self.assertNotIn("answer:", txt.lower())


class TestMirrorParsing(unittest.TestCase):
    valid = list("ABCDE") + ["none"]

    def test_valid_parse(self):
        p = parse_mirror_response("cause: tiebreak_error\nanswer: B\n",
                                  self.valid)
        self.assertEqual(p["parse_status"], "valid")
        self.assertEqual(p["cause"], "tiebreak_error")
        self.assertEqual(p["answer"], "B")

    def test_strict_format(self):
        for bad in ("answer: B", "cause: tie\nanswer: B\nextra", 
                    "cause: tie\nB", "cause tie\nanswer: B"):
            self.assertEqual(parse_mirror_response(bad, self.valid)["parse_status"],
                             "invalid_format", bad)

    def test_invalid_cause_and_answer(self):
        self.assertEqual(parse_mirror_response("cause: bad\nanswer: B",
                                               self.valid)["parse_status"],
                         "invalid_cause")
        self.assertEqual(parse_mirror_response("cause: tie\nanswer: Z",
                                               self.valid)["parse_status"],
                         "invalid_answer")

    def test_all_causes_parse(self):
        for cause in MIRROR_CAUSES:
            p = parse_mirror_response(f"cause: {cause}\nanswer: none",
                                      self.valid)
            self.assertEqual(p["parse_status"], "valid", cause)


class TestReferenceFailureCause(unittest.TestCase):
    def test_correct_is_no_failure(self):
        cands = [
            {"label": "A", "ar": 0.8, "lc": 0.5, "obs_available": True},
            {"label": "B", "ar": 0.2, "lc": 1.0, "obs_available": True},
        ]
        self.assertEqual(reference_failure_cause("conj_active", cands, "A"),
                         "no_failure")

    def test_none_qualify_dominates(self):
        cands = [
            {"label": "A", "ar": 0.1, "lc": 1.0, "obs_available": True},
            {"label": "B", "ar": 0.9, "lc": -0.1, "obs_available": True},
        ]
        self.assertEqual(reference_failure_cause("conj_active", cands, "B"),
                         "none_qualify")

    def test_tie_dominates(self):
        cands = [
            {"label": "A", "ar": 0.8, "lc": 0.5, "obs_available": True},
            {"label": "B", "ar": 0.8, "lc": 0.5, "obs_available": True},
        ]
        self.assertEqual(reference_failure_cause("conj_active", cands, "A"),
                         "tie")

    def test_missing_observation(self):
        cands = [
            {"label": "A", "ar": 0.9, "lc": 1.0, "obs_available": False},
            {"label": "B", "ar": 0.8, "lc": 0.5, "obs_available": True},
        ]
        self.assertEqual(reference_failure_cause("conj_active", cands, "A"),
                         "missing_observation")

    def test_threshold_error(self):
        cands = [
            {"label": "A", "ar": 0.9, "lc": 0.1, "obs_available": True},
            {"label": "B", "ar": 0.8, "lc": 0.5, "obs_available": True},
        ]
        self.assertEqual(reference_failure_cause("conj_active", cands, "A"),
                         "threshold_error")

    def test_tiebreak_error(self):
        cands = [
            {"label": "A", "ar": 0.8, "lc": 0.5, "obs_available": True},
            {"label": "B", "ar": 0.9, "lc": 0.4, "obs_available": True},
        ]
        self.assertEqual(reference_failure_cause("conj_active", cands, "A"),
                         "tiebreak_error")

    def test_missed_qualifier(self):
        cands = [
            {"label": "A", "ar": 0.8, "lc": 0.5, "obs_available": True},
            {"label": "B", "ar": 0.2, "lc": 1.0, "obs_available": True},
        ]
        self.assertEqual(reference_failure_cause("conj_active", cands, "none"),
                         "missed_qualifier")


class TestMirrorGrading(unittest.TestCase):
    valid = list("ABCDE") + ["none"]

    def test_correct_revision(self):
        g = grade_mirror_response(
            "cause: threshold_error\nanswer: E", "E", "C", self.valid,
            "threshold_error")
        self.assertTrue(g["answer_correct"])
        self.assertTrue(g["changed_from_previous"])
        self.assertTrue(g["cause_correct"])
        self.assertEqual(g["answer_verdict"], "correct")

    def test_correct_answer_changed_is_detected(self):
        g = grade_mirror_response(
            "cause: no_failure\nanswer: C", "B", "B", self.valid,
            "no_failure")
        self.assertFalse(g["answer_correct"])
        self.assertTrue(g["changed_from_previous"])
        self.assertEqual(g["answer_verdict"], "valid_but_incorrect")

    def test_invalid_format_never_counts_as_revision_evidence(self):
        g = grade_mirror_response("I think it is E", "E", "C", self.valid,
                                  "threshold_error")
        self.assertEqual(g["answer_verdict"], "invalid_format")
        self.assertFalse(g["answer_correct"])
        self.assertFalse(g["changed_from_previous"])
        self.assertFalse(g["cause_correct"])


if __name__ == "__main__":
    unittest.main()
