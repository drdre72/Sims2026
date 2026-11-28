import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from jarvis_research.cognitive_bus import *


def matrix_rows():
    return {"expected_rows":3,"rows":[
        {"label":"X","h1":False,"h2":False},
        {"label":"Y","h1":True,"h2":False},
        {"label":"Z","h1":True,"h2":True},
    ]}

class TestCognitiveBus(unittest.TestCase):
 def test_versioned_dispatch_and_binding(self):
  bus=CognitiveBus(); src=bus.put('binary_row_set',matrix_rows(),provenance={'source':'test'},context_id='case')
  self.assertEqual(src.status,VALID)
  events=auto_dispatch(bus)
  self.assertEqual([e['primitive'] for e in events],['DIFF','COUNT','UNIQUE'])
  flags=bus.binding('comparison.flags'); count=bus.binding('comparison.count'); unique=bus.binding('decision.unique')
  self.assertEqual(thaw(flags.value)['flags'],{'X':False,'Y':True,'Z':False})
  self.assertEqual(thaw(count.value)['count'],1)
  self.assertEqual(thaw(unique.value)['unique'],'Y')
  self.assertEqual(bus.effective_status(unique),RESOLVED)
  self.assertEqual(len(unique.depends_on),2)
  self.assertIn('deterministic_primitive',thaw(unique.provenance)['source'])
 def test_no_dispatch_for_nonexecutable_evidence(self):
  bad=[({'expected_rows':2,'rows':[{'label':'A','h1':0}],},UNKNOWN),
       ({'expected_rows':2,'rows':[{'label':'A','h1':2,'h2':0},{'label':'B','h1':0,'h2':0}]},INVALID),
       ({'expected_rows':2,'rows':[{'label':'A','h1':0,'h2':0},{'label':'A','h1':1,'h2':0}]},AMBIGUOUS)]
  for value,status in bad:
   bus=CognitiveBus(); obj=bus.put('binary_row_set',value)
   self.assertEqual(obj.status,status)
   self.assertEqual(auto_dispatch(bus),[])
   self.assertIsNone(bus.binding('decision.unique'))
 def test_stale_invalidation_and_refresh(self):
  bus=CognitiveBus(); src=bus.put('binary_row_set',matrix_rows(),context_id='case')
  auto_dispatch(bus); unique=bus.binding('decision.unique')
  self.assertEqual(bus.effective_status(unique),RESOLVED)
  changed=matrix_rows(); changed['rows'][0]['h2']=True; changed['rows'][1]['h2']=True
  bus.update(src.object_id,changed,{'source':'test_update'})
  self.assertEqual(bus.effective_status(bus.binding('decision.unique')),STALE)
  events=auto_dispatch(bus)
  self.assertEqual([e['primitive'] for e in events],['DIFF','COUNT','UNIQUE'])
  self.assertEqual(bus.effective_status(bus.binding('decision.unique')),RESOLVED)
  self.assertEqual(thaw(bus.binding('decision.unique').value)['unique'],'X')
 def test_update_proposals_are_validated(self):
  bus=CognitiveBus(); src=bus.put('binary_row_set',matrix_rows())
  auto_dispatch(bus); derived=bus.binding('decision.unique')
  p=propose_premise_update(bus,derived.object_id,matrix_rows(),{'proposer':'test','rationale':'bad'})
  self.assertFalse(p.valid); self.assertIn('derived_object_not_a_premise',p.errors)
  p=propose_premise_update(bus,src.object_id,matrix_rows(),{'proposer':'test'})
  self.assertFalse(p.valid); self.assertIn('missing_rationale',p.errors)
  p=propose_premise_update(bus,src.object_id,matrix_rows(),{'proposer':'test','rationale':'ok'})
  self.assertTrue(p.valid)
 def test_context_isolation(self):
  bus=CognitiveBus()
  a=bus.put('binary_row_set',matrix_rows(),context_id='a')
  bvalue=matrix_rows(); bvalue['rows'][1]['h2']=True
  b=bus.put('binary_row_set',bvalue,context_id='b')
  events=auto_dispatch(bus)
  uniq=[e for e in events if e['primitive']=='UNIQUE']
  self.assertEqual(len(uniq),2)
  self.assertEqual(len({e['output_id'] for e in uniq}),2)
 def test_epistemic_matrix_cost_and_tie(self):
  value={'consistent':[True,True],'queries':[
   {'label':'Q1','cost':1,'h1':True,'h2':False},
   {'label':'Q2','cost':4,'h1':True,'h2':False},
   {'label':'Q3','cost':1,'h1':False,'h2':True}]}
  bus=CognitiveBus(); bus.put('epistemic_matrix_problem',value)
  auto_dispatch(bus); obj=bus.binding('epistemic.choice')
  self.assertEqual(bus.effective_status(obj),AMBIGUOUS)
  self.assertEqual(sorted(thaw(obj.value)['candidates']),['Q1','Q3'])
 def test_ambiguous_derived_result_stales_before_refresh(self):
  value={'consistent':[True,True],'queries':[
   {'label':'Q1','cost':1,'h1':True,'h2':False},
   {'label':'Q2','cost':4,'h1':True,'h2':False},
   {'label':'Q3','cost':1,'h1':False,'h2':True}]}
  bus=CognitiveBus(); src=bus.put('epistemic_matrix_problem',value)
  auto_dispatch(bus)
  self.assertEqual(bus.effective_status(bus.binding('epistemic.choice')),AMBIGUOUS)
  changed=copy.deepcopy(value); changed['queries'][2]['cost']=2
  bus.update(src.object_id,changed,{'source':'test_update'})
  self.assertEqual(bus.effective_status(bus.binding('epistemic.choice')),STALE)
  auto_dispatch(bus)
  obj=bus.binding('epistemic.choice')
  self.assertEqual(bus.effective_status(obj),RESOLVED)
  self.assertEqual(thaw(obj.value)['choice'],'Q1')
 def test_rule_consistency_and_discrimination(self):
  value={'rules':[{'kind':'conj','a':.5,'b':.5},{'kind':'either','a':.5,'b':.5}],
   'observations':[{'x':.7,'y':.7,'outcome':True},{'x':.2,'y':.2,'outcome':False}],
   'queries':[{'label':'Q1','x':.7,'y':.2},{'label':'Q2','x':.7,'y':.7}]}
  bus=CognitiveBus(); bus.put('rule_epistemic_problem',value)
  events=auto_dispatch(bus)
  self.assertEqual([e['primitive'] for e in events],['CONSISTENT','DISCRIMINATE_RULES'])
  self.assertEqual(thaw(bus.binding('epistemic.consistency').value)['fit'],[True,True])
  self.assertEqual(thaw(bus.binding('epistemic.choice').value)['choice'],'Q1')
 def test_snapshot_is_read_only_data(self):
  bus=CognitiveBus(); bus.put('binary_row_set',matrix_rows())
  auto_dispatch(bus); snap=bus.snapshot()
  self.assertIn('objects',snap); self.assertIn('bindings',snap)
  self.assertTrue(all('effective_status' in x for x in snap['objects']))
  snap['objects'][0]['value']['rows'][0]['h1']='corrupted'
  self.assertEqual(thaw(bus.binding('comparison.flags').value)['flags']['Y'],True)

if __name__=='__main__': unittest.main()
