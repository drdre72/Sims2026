"""HZN-OOI-01 operational orientation and inhibition tests."""
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jarvis_research.gate1_4_benchmark import FAMILIES, grade_task, make_task
from jarvis_research.gate1_4_mirror import candidate_evaluation
from jarvis_research.gate1_4_operational import (
    CONDITIONS, OOI_FAMILIES, build_operational_tasks,
    grade_structured_artifact, parse_structured_artifact,
    render_operational_condition, render_reference_artifact, rule_summary,
    summarize_operational_cells, apply_exact_tie_guard)


class TestOperationalBattery(unittest.TestCase):
    def test_fixed_battery_invariants(self):
        tasks = build_operational_tasks()
        self.assertEqual(len(tasks), 36)
        self.assertEqual(len({t['scenario_index'] for t in tasks}), 24)
        self.assertEqual(Counter(t['family'] for t in tasks), Counter({
            'exact_tie': 12, 'near_tie': 12,
            'insufficient': 6, 'unique_control': 6}))
        self.assertEqual(Counter(t['rule_key'] for t in tasks), Counter({
            'conj_active': 12, 'conj_strong': 12, 'avoid_negative': 12}))
        self.assertEqual(sum(t['truth']['answer'] == 'none' for t in tasks), 18)
        self.assertEqual(sum(t['truth']['answer'] != 'none' for t in tasks), 18)
        self.assertEqual(sum(t['truth']['answer'] == t['magnitude_answer']
                             for t in tasks), 0)
        self.assertEqual(Counter(t['presentation'] for t in tasks), Counter({
            'order_a': 24, 'pair_swapped': 12}))

    def test_deterministic(self):
        self.assertEqual(build_operational_tasks(), build_operational_tasks())

    def test_family_truth_shapes(self):
        tasks = build_operational_tasks()
        for task in tasks:
            if task['family'] == 'exact_tie':
                self.assertEqual(task['truth']['answer'], 'none')
                self.assertEqual(task['truth']['reason'], 'tie')
                self.assertEqual(task['mechanics']['state'], 'tie')
            elif task['family'] == 'near_tie':
                self.assertNotEqual(task['truth']['answer'], 'none')
                self.assertEqual(task['mechanics']['state'], 'unique')
            elif task['family'] == 'insufficient':
                self.assertEqual(task['truth']['answer'], 'none')
                self.assertEqual(task['truth']['reason'], 'none_qualify')
            elif task['family'] == 'unique_control':
                self.assertNotEqual(task['truth']['answer'], 'none')
                self.assertEqual(task['mechanics']['state'], 'unique')

    def test_near_tie_margin_is_exactly_one_cent(self):
        for task in build_operational_tasks():
            if task['family'] != 'near_tie' or task['presentation'] != 'order_a':
                continue
            vals = []
            for item in task['critical_items']:
                cand = next(c for c in task['candidates'] if c['item'] == item)
                vals.append(candidate_evaluation(
                    task['rule_key'], cand)['tiebreak_value'])
            self.assertAlmostEqual(vals[1] - vals[0], 0.01, places=8)

    def test_swapped_pair_preserves_evidence_and_reverses_item_order(self):
        tasks = build_operational_tasks()
        by_scenario = {}
        for task in tasks:
            if task['family'] in ('exact_tie', 'near_tie'):
                by_scenario.setdefault(task['scenario_index'], {})[
                    task['presentation']] = task
        for variants in by_scenario.values():
            a, b = variants['order_a'], variants['pair_swapped']
            key = lambda c: (c['item'], c['ar'], c['lc'], c['obs_available'])
            self.assertEqual(sorted(map(key, a['candidates'])),
                             sorted(map(key, b['candidates'])))
            self.assertEqual([c['label'] for c in a['candidates']], list('ABCDE'))
            self.assertEqual([c['label'] for c in b['candidates']], list('ABCDE'))
            self.assertEqual(a['critical_positions'], b['critical_positions'])
            self.assertNotEqual(a['first_critical_item'], b['first_critical_item'])


class TestRuleMechanics(unittest.TestCase):
    def test_rule_summary_agrees_with_ir11_grader(self):
        for seed in range(100):
            task = make_task(random.Random(seed), FAMILIES[seed % len(FAMILIES)])
            mech = rule_summary(task['rule_key'], task['candidates'])
            truth = grade_task(task['rule_key'], task['candidates'])
            self.assertEqual(mech['answer'], truth['answer'], seed)
            self.assertEqual(mech['qualifiers'], truth['qualifying'], seed)
            self.assertEqual(mech['state'], truth['reason'], seed)

    def test_exact_tie_guard_only_inhibits_tie_states(self):
        exact = next(t for t in build_operational_tasks()
                     if t['family'] == 'exact_tie')
        near = next(t for t in build_operational_tasks()
                    if t['family'] == 'near_tie')
        self.assertEqual(apply_exact_tie_guard(
            'A', exact['rule_key'], exact['candidates']), 'none')
        self.assertEqual(apply_exact_tie_guard(
            'A', near['rule_key'], near['candidates']), 'A')


class TestOperationalRenderingAndParsing(unittest.TestCase):
    def setUp(self):
        self.tasks = build_operational_tasks()
        self.task = self.tasks[0]
        self.labels = [c['label'] for c in self.task['candidates']]

    def test_conditions(self):
        self.assertEqual(CONDITIONS, ('BASE', 'STRUCT'))
        base = render_operational_condition(self.task, 'BASE')
        struct = render_operational_condition(self.task, 'STRUCT')
        self.assertIn('Reply with ONLY the candidate letter', base)
        self.assertNotIn('Reply with ONLY the candidate letter', struct)
        self.assertIn('operational orientation artifact', struct)
        self.assertIn('tiebreak=<ar or abs_lc>', struct)
        self.assertIn('summary|qualifiers=', struct)

    def test_reference_artifact_roundtrip(self):
        for task in self.tasks:
            artifact = render_reference_artifact(task)
            labels = [c['label'] for c in task['candidates']]
            parsed = parse_structured_artifact(artifact, labels)
            self.assertEqual(parsed['parse_status'], 'valid', task['task_id'])
            graded = grade_structured_artifact(artifact, task)
            self.assertTrue(graded['answer_correct'], task['task_id'])
            self.assertTrue(graded['delegated_correct'], task['task_id'])
            self.assertTrue(graded['raw_extraction_exact'], task['task_id'])
            self.assertTrue(graded['summary_correct'], task['task_id'])
            self.assertEqual(graded['eval_row_accuracy'], 1.0, task['task_id'])
            self.assertEqual(graded['internal_eval_row_accuracy'], 1.0,
                             task['task_id'])

    def test_strict_format(self):
        artifact = render_reference_artifact(self.task)
        for bad in ('answer=none',
                    artifact + '\nextra',
                    artifact.replace('tiebreak=ar\n', '', 1),
                    artifact.replace('label|ar|lc|ar_pass|contrast_pass|qualifies|tb',
                                     'label|ar|lc', 1)):
            self.assertEqual(parse_structured_artifact(
                bad, self.labels)['parse_status'], 'invalid_format')

    def test_invalid_answer_status(self):
        artifact = render_reference_artifact(self.task)
        bad = artifact.rsplit('answer=', 1)[0] + 'answer=Z'
        parsed = parse_structured_artifact(bad, self.labels)
        self.assertEqual(parsed['parse_status'], 'invalid_answer')

    def test_observed_rows_require_numeric_evidence(self):
        artifact = render_reference_artifact(self.task)
        bad = artifact.replace('A|yes|', 'A|yes|', 1)
        lines = bad.splitlines()
        parts = lines[2].split('|')
        parts[1] = 'na'
        lines[2] = '|'.join(parts)
        parsed = parse_structured_artifact('\n'.join(lines), self.labels)
        self.assertEqual(parsed['parse_status'], 'invalid_format')


class TestOperationalGrading(unittest.TestCase):
    def test_tie_neural_error_can_still_delegate_correctly(self):
        task = next(t for t in build_operational_tasks()
                    if t['family'] == 'exact_tie')
        artifact = render_reference_artifact(task)
        wrong = artifact.rsplit('answer=', 1)[0] + 'answer=A'
        graded = grade_structured_artifact(wrong, task)
        self.assertFalse(graded['answer_correct'])
        self.assertEqual(graded['answer'], 'A')
        self.assertEqual(graded['delegated_answer'], 'none')
        self.assertTrue(graded['delegated_correct'])
        self.assertTrue(graded['summary_correct'])

    def test_invalid_format_never_counts_as_correct_or_delegated(self):
        task = build_operational_tasks()[0]
        graded = grade_structured_artifact('I choose none', task)
        self.assertEqual(graded['parse_status'], 'invalid_format')
        self.assertFalse(graded['answer_correct'])
        self.assertFalse(graded['delegated_correct'])
        self.assertFalse(graded['format_valid'])

    def test_summary_error_is_independently_visible(self):
        task = next(t for t in build_operational_tasks()
                    if t['family'] == 'exact_tie')
        artifact = render_reference_artifact(task)
        bad = artifact.replace('state=tie', 'state=unique')
        graded = grade_structured_artifact(bad, task)
        self.assertTrue(graded['format_valid'])
        self.assertFalse(graded['summary_correct'])
        self.assertTrue(graded['raw_extraction_exact'])


class TestOperationalSummary(unittest.TestCase):
    def test_metric_definitions(self):
        tasks = build_operational_tasks()[:2]
        cells = []
        for task in tasks:
            artifact = render_reference_artifact(task)
            struct = grade_structured_artifact(artifact, task)
            cells.append({
                'family': task['family'], 'rule_key': task['rule_key'],
                'truth': task['truth']['answer'],
                'candidates': task['candidates'],
                'critical_labels': task['critical_labels'],
                'label_to_item': task['label_to_item'],
                'first_critical_item': task['first_critical_item'],
                'conditions': {
                    'BASE': {'answer': task['truth']['answer'],
                             'answer_correct': True,
                             'format_valid': True},
                    'STRUCT': struct,
                },
            })
        summary = summarize_operational_cells(cells)
        self.assertEqual(summary['BASE']['accuracy'], 1.0)
        self.assertEqual(summary['STRUCT']['accuracy'], 1.0)
        self.assertEqual(summary['STRUCT_DELEGATED']['accuracy'], 1.0)
        self.assertEqual(summary['STRUCT_ARTIFACT']['raw_extraction_exact_rate'], 1.0)
        self.assertIn('BASE_TIE_GUARD', summary)
        self.assertIn('BASE_vs_STRUCT', summary['paired_comparisons'])


if __name__ == '__main__':
    unittest.main()
