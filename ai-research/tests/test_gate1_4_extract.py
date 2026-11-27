"""HZN-OOI-03 extraction-mediated delegation tests."""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jarvis_research.gate1_4_extract import (
    CONDITIONS, build_extract_tasks, grade_extract_artifact,
    parse_extract_artifact, render_extract_condition,
    render_reference_extract, summarize_extract_cells)


class TestExtractBattery(unittest.TestCase):
    def test_fixed_battery_invariants(self):
        tasks = build_extract_tasks()
        self.assertEqual(len(tasks), 36)
        self.assertEqual(len({t['scenario_index'] for t in tasks}), 24)
        self.assertEqual(Counter(t['family'] for t in tasks), Counter({
            'exact_tie': 12, 'near_tie': 12,
            'insufficient': 6, 'unique_control': 6}))
        self.assertEqual(Counter(t['rule_key'] for t in tasks), Counter({
            'conj_active': 12, 'conj_strong': 12, 'avoid_negative': 12}))
        self.assertEqual(sum(t['truth']['answer'] == 'none' for t in tasks), 18)
        self.assertEqual(sum(t['truth']['answer'] == t['magnitude_answer']
                             for t in tasks), 0)

    def test_deterministic(self):
        self.assertEqual(build_extract_tasks(), build_extract_tasks())


class TestExtractRenderingAndParsing(unittest.TestCase):
    def setUp(self):
        self.task = build_extract_tasks()[0]
        self.labels = [c['label'] for c in self.task['candidates']]

    def test_conditions_and_contract(self):
        self.assertEqual(CONDITIONS, ('BASE', 'XTRACT'))
        base = render_extract_condition(self.task, 'BASE')
        xtract = render_extract_condition(self.task, 'XTRACT')
        self.assertIn('Reply with ONLY the candidate letter', base)
        self.assertNotIn('Reply with ONLY the candidate letter', xtract)
        self.assertIn('evidence-orientation artifact', xtract)
        self.assertIn('label|ar|lc', xtract)
        self.assertIn('Do not add a header', xtract)

    def test_reference_roundtrip(self):
        for task in build_extract_tasks():
            artifact = render_reference_extract(task)
            labels = [c['label'] for c in task['candidates']]
            parsed = parse_extract_artifact(artifact, labels)
            self.assertEqual(parsed['parse_status'], 'valid', task['task_id'])
            graded = grade_extract_artifact(artifact, task)
            self.assertTrue(graded['format_valid'], task['task_id'])
            self.assertTrue(graded['answer_correct'], task['task_id'])
            self.assertTrue(graded['raw_extraction_exact'], task['task_id'])
            self.assertTrue(graded['delegated_correct'], task['task_id'])

    def test_strict_format(self):
        artifact = render_reference_extract(self.task)
        for bad in ('answer=none', artifact + '\nextra',
                    artifact.replace('A|', 'A:', 1),
                    artifact.rsplit('answer=', 1)[0] + 'answer=Z'):
            self.assertNotEqual(parse_extract_artifact(
                bad, self.labels)['parse_status'], 'valid', bad[:30])


class TestExtractGrading(unittest.TestCase):
    def test_neural_tie_error_can_still_delegate_correctly(self):
        task = next(t for t in build_extract_tasks()
                    if t['family'] == 'exact_tie')
        artifact = render_reference_extract(task)
        wrong = artifact.rsplit('answer=', 1)[0] + 'answer=A'
        graded = grade_extract_artifact(wrong, task)
        self.assertFalse(graded['answer_correct'])
        self.assertTrue(graded['raw_extraction_exact'])
        self.assertEqual(graded['delegated_answer'], 'none')
        self.assertTrue(graded['delegated_correct'])

    def test_invalid_never_counts_as_correct_or_delegated(self):
        graded = grade_extract_artifact('I choose none', build_extract_tasks()[0])
        self.assertEqual(graded['parse_status'], 'invalid_format')
        self.assertFalse(graded['answer_correct'])
        self.assertFalse(graded['delegated_correct'])

    def test_summary_metrics(self):
        task = build_extract_tasks()[0]
        graded = grade_extract_artifact(render_reference_extract(task), task)
        cell = {
            'family': task['family'], 'rule_key': task['rule_key'],
            'truth': task['truth']['answer'], 'candidates': task['candidates'],
            'critical_labels': task['critical_labels'],
            'label_to_item': task['label_to_item'],
            'first_critical_item': task['first_critical_item'],
            'conditions': {
                'BASE': {'answer': task['truth']['answer'],
                         'answer_correct': True, 'format_valid': True},
                'XTRACT': graded,
            },
        }
        summary = summarize_extract_cells([cell])
        self.assertEqual(summary['BASE']['accuracy'], 1.0)
        self.assertEqual(summary['XTRACT']['accuracy'], 1.0)
        self.assertEqual(summary['XTRACT_DELEGATED']['accuracy'], 1.0)
        self.assertEqual(summary['XTRACT_ARTIFACT']['raw_extraction_exact_rate'], 1.0)


if __name__ == '__main__':
    unittest.main()
