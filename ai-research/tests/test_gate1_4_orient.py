"""HZN-ORIENT-01 construction and metric tests."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jarvis_research.gate1_4_orient import (
    CONDITIONS, ORIENT_PREAMBLE, build_orient_tasks, render_condition,
    summarize_orient_cells)


class TestOrientBattery(unittest.TestCase):
    def test_fixed_battery_invariants(self):
        tasks = build_orient_tasks()
        self.assertEqual(len(tasks), 18)
        self.assertEqual(len({t['family'] for t in tasks}), 6)
        rules = {}
        for t in tasks:
            rules[t['rule_key']] = rules.get(t['rule_key'], 0) + 1
        self.assertEqual(sorted(rules.values()), [6, 6, 6])
        none_ct = sum(t['truth']['answer'] == 'none' for t in tasks)
        mag_ct = sum(t['truth']['answer'] == t['magnitude_answer']
                     for t in tasks)
        self.assertEqual(none_ct, 7)
        self.assertEqual(mag_ct, 1)
        # Fixed seed/version regression anchor.
        self.assertEqual(tasks[0]['truth']['answer'], 'B')
        self.assertEqual(tasks[17]['truth']['answer'], 'D')

    def test_deterministic(self):
        self.assertEqual(build_orient_tasks(), build_orient_tasks())


class TestOrientRendering(unittest.TestCase):
    def setUp(self):
        self.task = build_orient_tasks()[0]

    def test_all_conditions_share_output_contract(self):
        for cond in CONDITIONS:
            txt = render_condition(self.task, cond)
            self.assertTrue(txt.endswith(
                'Reply with ONLY the candidate letter, or the word none.'), cond)

    def test_orient_topology_present_only_in_orient(self):
        orient = render_condition(self.task, 'ORIENT')
        careful = render_condition(self.task, 'CAREFUL')
        base = render_condition(self.task, 'BASE')
        for needle in ('center:', 'who/which:', 'what:', 'where:', 'when:',
                       'how:', 'why:'):
            self.assertIn(needle, orient)
            self.assertNotIn(needle, careful)
            self.assertNotIn(needle, base)

    def test_careful_is_generic_control(self):
        careful = render_condition(self.task, 'CAREFUL')
        self.assertIn('carefully review', careful)
        self.assertIn('check whether none is required', careful)

    def test_preamble_does_not_reveal_truth(self):
        for cond in ('CAREFUL', 'ORIENT'):
            preamble = render_condition(self.task, cond).split('Rule:')[0]
            self.assertNotIn(self.task['truth']['answer'] + '\n', preamble)


class TestOrientSummary(unittest.TestCase):
    def cell(self, truth, mag, answers):
        return {'truth': truth, 'magnitude_answer': mag,
                'conditions': {
                    cond: {'answer': ans,
                           'answer_correct': ans == truth,
                           'format_valid': True}
                    for cond, ans in answers.items()}}

    def test_metric_definitions(self):
        cells = [
            self.cell('B', 'A', {'BASE': 'A', 'CAREFUL': 'B', 'ORIENT': 'B'}),
            self.cell('none', 'C', {'BASE': 'C', 'CAREFUL': 'none', 'ORIENT': 'none'}),
        ]
        s = summarize_orient_cells(cells)
        self.assertEqual(s['BASE']['accuracy'], 0.0)
        self.assertEqual(s['ORIENT']['accuracy'], 1.0)
        self.assertEqual(s['BASE']['magnitude_capture_rate_conflict'], 1.0)
        self.assertEqual(s['ORIENT']['magnitude_capture_rate_conflict'], 0.0)
        self.assertEqual(s['BASE']['selection_accuracy_answer_present'], 0.0)
        self.assertEqual(s['ORIENT']['abstention_accuracy_none'], 1.0)
        comp = s['paired_comparisons']['BASE_vs_ORIENT']
        self.assertEqual(comp['ORIENT_only_correct'], 2)
        self.assertEqual(comp['BASE_only_correct'], 0)

    def test_invalid_counts_wrong(self):
        cell = self.cell('B', 'A', {'BASE': 'junk', 'CAREFUL': 'B', 'ORIENT': 'B'})
        cell['conditions']['BASE']['format_valid'] = False
        s = summarize_orient_cells([cell])
        self.assertEqual(s['BASE']['accuracy'], 0.0)
        self.assertEqual(s['BASE']['format_valid_rate'], 0.0)


if __name__ == '__main__':
    unittest.main()
