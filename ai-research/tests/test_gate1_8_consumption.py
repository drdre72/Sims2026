import sys,unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from jarvis_research.gate1_8_consumption import *
from jarvis_research.gate1_7_substrate import render_hybrid_matrix

class TestConsumptionHarness(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.matrix=build_matrix_state_scenarios(); cls.epis=build_epistemic_decision_scenarios()
 def test_battery_shapes_and_policies(self):
  self.assertEqual(len(self.matrix),37)
  self.assertEqual(Counter(s['state_group'] for s in self.matrix),Counter({'resolved':18,'nonnormal':18,'stale':1}))
  nonnormal=[s for s in self.matrix if s['state_group']=='nonnormal']
  self.assertEqual(Counter(s['family'] for s in nonnormal),Counter({
   'incomplete_missing_bit':3,'malformed_bit':3,'contradictory_duplicate_label':3,
   'incomplete_missing_row':3,'unknown_bit':3,'malformed_type':3}))
  self.assertTrue(all(s['expected_action']=='answer' for s in self.matrix if s['state_group']=='resolved'))
  self.assertEqual(Counter(s['expected_action'] for s in nonnormal),Counter({'query':9,'reject_premises':9}))
  self.assertEqual(self.matrix[-1]['expected_action'],'refresh')
  self.assertEqual(len(self.epis),12)
  self.assertEqual(Counter(s['expected_action'] for s in self.epis),Counter({'query':6,'abstain':4,'insufficient_evidence':2}))
 def test_nonnormal_states_do_not_dispatch(self):
  nonnormal=[s for s in self.matrix if s['state_group']=='nonnormal']
  self.assertTrue(all(s['n_dispatch_events']==0 for s in nonnormal))
  self.assertTrue(all(s['effective_status']==s['case']['expected_status'] for s in nonnormal))
  self.assertEqual(self.matrix[-1]['effective_status'],'stale')
 def test_baseline_prompt_is_epi5_hybrid_prompt(self):
  s=self.matrix[0]
  expected=render_hybrid_matrix(s['case'],s['bus'])
  self.assertIn('You are the neural interpretation layer above a typed working-state runtime.',expected)
  self.assertIn('Reply with ONLY target=<row label, none, or status:STATUS>.',expected)
 def test_readout_strict_relaxed_and_failure_classes(self):
  s=self.matrix[0]
  good=grade_readout('target='+s['expected_report'],s)
  self.assertTrue(good['strict_correct']); self.assertTrue(good['state_respect'])
  colon=grade_readout('target:'+s['expected_report'],s)
  self.assertFalse(colon['strict_correct']); self.assertTrue(colon['relaxed_correct'])
  self.assertEqual(colon['failure_class'],'format')
  wrong=grade_readout('target=C',s)
  self.assertEqual(wrong['failure_class'],'state-ignore')
  empty=grade_readout('the answer is unavailable',s)
  self.assertEqual(empty['failure_class'],'hallucinated-content')
 def test_stale_readout_distinguishes_stale_copy_and_recompute(self):
  s=self.matrix[-1]
  self.assertEqual(s['cached_unique'],'C'); self.assertEqual(s['naive_answer'],'D')
  stale_copy=grade_readout('target=C',s)
  self.assertTrue(stale_copy['stale_value_emitted']); self.assertEqual(stale_copy['failure_class'],'state-ignore')
  recompute=grade_readout('target=D',s)
  self.assertEqual(recompute['failure_class'],'premise-recompute')
  correct=grade_readout('target=status:stale',s)
  self.assertTrue(correct['strict_correct'])
 def test_repaired_prompt_contains_nonnormal_contract_examples(self):
  p=render_readout_repaired(self.matrix[-1])
  for x in ('target=none','target=status:unknown','target=status:invalid','target=status:ambiguous','target=status:stale'):
   self.assertIn(x,p)
  self.assertIn('never :',p)
  self.assertIn('Typed working state:',p)
 def test_matrix_decision_grading(self):
  stale=self.matrix[-1]
  good=grade_matrix_decision('action=refresh',stale)
  self.assertTrue(good['action_correct']); self.assertTrue(good['state_respect'])
  colon=grade_matrix_decision('action:refresh',stale)
  self.assertEqual(colon['failure_class'],'format'); self.assertTrue(colon['relaxed_correct'])
  answer=grade_matrix_decision('action=answer',stale)
  self.assertEqual(answer['failure_class'],'premise-recompute')
  bad=grade_matrix_decision('action=fly',stale)
  self.assertEqual(bad['failure_class'],'hallucinated-content')
 def test_epistemic_decision_grading_and_ties(self):
  tie=next(s for s in self.epis if s['family']=='need_multiple_equal')
  self.assertTrue(grade_epistemic_decision('action=query choice='+tie['acceptable_choices'][0],tie)['action_correct'])
  self.assertTrue(grade_epistemic_decision('action=query choice='+tie['acceptable_choices'][1],tie)['action_correct'])
  wrong=grade_epistemic_decision('action=query choice=Q4',tie)
  if 'Q4' in tie['acceptable_choices']:
   self.assertTrue(wrong['action_correct'])
  else:
   self.assertTrue(wrong['unauthorized_choice']); self.assertEqual(wrong['failure_class'],'state-ignore')
  resolved=next(s for s in self.epis if s['family']=='resolved_still_diff')
  premature=grade_epistemic_decision('action=query choice=Q1',resolved)
  self.assertEqual(premature['failure_class'],'premise-recompute')
  self.assertTrue(grade_epistemic_decision('action=abstain',resolved)['restraint_correct'])
  insufficient=next(s for s in self.epis if s['family']=='consistent_nodisc')
  self.assertTrue(grade_epistemic_decision('action=insufficient_evidence',insufficient)['action_correct'])
 def test_mechanical_matrix_renderer(self):
  renders=[render_mechanical_matrix(s) for s in self.matrix]
  self.assertEqual(sum(r['fidelity_correct'] for r in renders),37)
  self.assertTrue(all(r['neural_output_tokens']==0 for r in renders))
  self.assertTrue(all(not r['stale_value_emitted'] for r in renders))
  stale=renders[-1]
  self.assertEqual(stale['render_key'],'stale'); self.assertFalse(stale['stale_value_emitted'])
  self.assertNotIn('Row C',stale['rendered_text'])
 def test_mechanical_epistemic_renderer(self):
  renders=[render_mechanical_epistemic(s) for s in self.epis]
  self.assertEqual(sum(r['fidelity_correct'] for r in renders),12)
  ties=[r for r in renders if r['requires_neural_choice']]
  self.assertEqual(len(ties),2)
  self.assertEqual({r['family'] for r in ties},{'need_multiple_equal'})
  self.assertTrue(all(not r['end_to_end_correct'] for r in ties))
  nonties=[r for r in renders if not r['requires_neural_choice']]
  self.assertTrue(all(r['end_to_end_correct'] for r in nonties))
  p=render_tie_interpretation_prompt(ties[0])
  self.assertIn('The lowest-cost discriminating observations are tied:',p)
  self.assertNotIn('Typed working state:',p); self.assertNotIn('cost=',p)
  self.assertEqual(parse_tie_choice('choice='+ties[0]['candidates'][0]),ties[0]['candidates'][0])
  self.assertIsNone(parse_tie_choice('choice=no_query'))
  self.assertIn('Selected tied observation:',finalize_tie_render(ties[0],ties[0]['candidates'][0]))
 def test_summary_smoke(self):
  s=self.matrix[0]
  readout=dict(grade_readout('target='+s['expected_report'],s),family=s['family'],state_group=s['state_group'],latency_s=1,output_tokens=5)
  decision=dict(grade_matrix_decision('action=answer',s),family=s['family'],latency_s=1,output_tokens=3)
  payload={'readout_baseline':[readout],'readout_repaired':[readout],'decision_matrix':[decision],
   'decision_epistemic':[],'mechanical_matrix_render':[render_mechanical_matrix(s)],
   'mechanical_epistemic_render':[render_mechanical_epistemic(self.epis[0])]}
  out=summarize_epi6(payload)
  self.assertEqual(out['READOUT_BASELINE']['strict_accuracy'],1.0)
  self.assertEqual(out['DECISION_CONSUMPTION']['decision_appropriateness'],1.0)
  self.assertEqual(out['MECHANICAL_RENDER']['matrix_fidelity_rate'],1.0)

if __name__=='__main__': unittest.main()
