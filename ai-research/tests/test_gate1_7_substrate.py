import sys,unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from jarvis_research.gate1_7_substrate import *
from jarvis_research.cognitive_bus import AMBIGUOUS,RESOLVED

class TestSubstrateHarness(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.matrix=build_matrix_cases(); cls.epis=build_epi_tasks(); cls.robust=build_robustness_cases()
 def test_battery_shapes(self):
  self.assertEqual(len(self.matrix),18); self.assertEqual(len(self.epis),12); self.assertEqual(len(self.robust),6)
  self.assertEqual(len(build_composition_cases()),4); self.assertEqual(len(build_rule_runtime_tasks()),6)
  self.assertEqual(Counter(x['expected_status'] for x in self.robust),Counter({'unknown':3,'invalid':2,'ambiguous':1}))
 def test_auto_matrix_reference_is_executed(self):
  rows=[run_auto_matrix_case(c) for c in self.matrix]
  self.assertEqual(sum(r['answer_correct'] for r in rows),18)
  self.assertTrue(all(r['n_dispatch_events']==3 for r in rows))
  self.assertTrue(all(r['status']==RESOLVED for r in rows))
  self.assertTrue(all(r['output_id'] for r in rows))
 def test_composition_generalization(self):
  rows=[run_auto_matrix_case(c) for c in build_composition_cases()]
  self.assertEqual(sum(r['answer_correct'] for r in rows),4)
  self.assertEqual(sorted({len(r['rows']) for r in rows}),[3,4,6,7])
 def test_robustness_statuses(self):
  rows=[run_auto_robustness_case(c) for c in self.robust]
  self.assertEqual(sum(r['answer_correct'] for r in rows),6)
  self.assertTrue(all(r['n_dispatch_events']==0 for r in rows))
  self.assertEqual({r['family']:r['input_status'] for r in rows},{r['family']:r['expected_status'] for r in rows})
 def test_stale_probe(self):
  p=run_stale_invalidation_probe()
  self.assertEqual(p['initial_decision'],'C'); self.assertEqual(p['status_after_premise_update'],'stale')
  self.assertTrue(p['invalidated']); self.assertEqual(p['refreshed_decision'],'D'); self.assertTrue(p['refresh_correct'])
  self.assertGreater(p['refreshed_output_version'],p['initial_output_version'])
  c,bus,input_obj,events=build_stale_hybrid_case()
  self.assertEqual(bus.effective_status(bus.binding('decision.unique')),'stale')
  g=grade_hybrid_matrix('target=status:stale',c,bus); self.assertTrue(g['answer_correct'])
 def test_epistemic_auto_states(self):
  rows=[run_auto_epistemic_case(t) for t in self.epis]
  self.assertEqual(sum(r['auto_correct'] for r in rows),12)
  ties=[r for r in rows if r['family']=='need_multiple_equal']
  self.assertEqual(len(ties),2); self.assertTrue(all(r['status']==AMBIGUOUS for r in ties))
  self.assertTrue(all(len(r['candidates'])==2 for r in ties))
 def test_rule_runtime(self):
  rows=[run_auto_rule_task(t) for t in build_rule_runtime_tasks()]
  self.assertEqual(sum(r['answer_correct'] for r in rows),6)
  self.assertTrue(all(r['n_dispatch_events']==2 for r in rows))
 def test_direct_grading(self):
  c=self.matrix[0]
  good=grade_matrix_final('final='+c['truth'],c)
  self.assertTrue(good['format_valid']); self.assertTrue(good['answer_correct'])
  self.assertIsNone(parse_matrix_final('answer='+c['truth'],c))
 def test_invocation_execution_and_m02_copy_error(self):
  c=next(x for x in self.matrix if x['case_id']=='m02')
  response='action=compare\nrows=A:0/0;B:1/1;C:0/1;D:0/0;E:0/0'
  action=parse_delegate_action(response,c)
  self.assertTrue(action['format_valid']); self.assertFalse(operand_exact(action['rows'],c))
  bus,input_obj,events,result=execute_matrix_invocation(action['rows'],c)
  self.assertEqual(result['state']['unique'],'C'); self.assertEqual(result['decision'],'C')
  self.assertEqual(len(events),3)
 def test_hybrid_parsing(self):
  c=self.matrix[0]; bus,input_obj,events=load_matrix_bus(c)
  good=grade_hybrid_matrix('target='+c['truth'],c,bus)
  self.assertTrue(good['answer_correct']); self.assertTrue(good['state_consistent'])
  bad=grade_hybrid_matrix('target=none',c,bus)
  self.assertFalse(bad['state_consistent'])
  x=self.robust[0]; xbus,xinput,xevents=load_robustness_bus(x)
  xh=grade_hybrid_matrix('target=status:unknown',x,xbus)
  self.assertTrue(xh['answer_correct'])
 def test_epistemic_hybrid_grading(self):
  t=self.epis[0]
  g=grade_hybrid_epistemic('choice='+t['acceptable'][0],t)
  self.assertTrue(g['answer_correct'])
 def test_summary_smoke(self):
  c=self.matrix[0]; auto=run_auto_matrix_case(c)
  payload={'matrix_auto':[auto], 'composition_auto':[run_auto_matrix_case(x) for x in build_composition_cases()],
           'robustness_auto':[run_auto_robustness_case(x) for x in self.robust],
           'stale_probe':run_stale_invalidation_probe(),
           'epistemic_auto':[run_auto_epistemic_case(t) for t in self.epis],
           'rule_runtime':[run_auto_rule_task(t) for t in build_rule_runtime_tasks()]}
  s=summarize_epi5(payload)
  self.assertEqual(s['AUTO_DISPATCH']['accuracy'],1.0)
  self.assertEqual(s['ROBUSTNESS_AUTO']['status_accuracy'],1.0)
  self.assertTrue(s['STALE_INVALIDATION']['invalidated'])
  self.assertEqual(s['EPISTEMIC_AUTO']['mechanical_state_accuracy'],1.0)

if __name__=='__main__': unittest.main()
