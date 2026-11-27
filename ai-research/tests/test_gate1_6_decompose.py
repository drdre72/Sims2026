import sys,unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from jarvis_research.gate1_6_decompose import *

class TestDecompose(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.cases=build_matrix_cases(); cls.singles=build_single_tasks(); cls.epis=build_epi_tasks()
 def test_matrix_battery(self):
  self.assertEqual(len(self.cases),18); self.assertEqual(self.cases,build_matrix_cases())
  self.assertEqual(Counter(c['family'] for c in self.cases),Counter({'unique_diff':8,'no_diff':5,'multiple_diff':5}))
  for c in self.cases:
   self.assertEqual(oracle_count(c),sum(r['diff'] for r in c['rows']))
   self.assertEqual(oracle_unique(c),c['truth'])
   self.assertEqual(len({r['label'] for r in c['rows']}),5)
   if c['family']=='unique_diff':
    self.assertEqual(c['rows'][c['target_display_index']]['label'],c['truth'])
  # Stratified controls are all represented.
  for key in ('labelset','colrev','permute','yesno','csv'):
   self.assertGreater(len({str(c['variant'][key]) for c in self.cases}),1)
 def test_single(self):
  self.assertEqual(len(self.singles),12); self.assertEqual(sum(t['truth'] for t in self.singles),6)
  for t in self.singles:
   pref='answer=' if t['variant']['yesno'] else 'relation='
   word=('yes' if t['truth'] else 'no') if t['variant']['yesno'] else ('different' if t['truth'] else 'same')
   g=grade_single(pref+word,t); self.assertTrue(g['format_valid']); self.assertTrue(g['answer_correct'])
   self.assertIsNone(parse_single('bad',t))
 def test_vector_count_select(self):
  for c in self.cases:
   if c['variant']['yesno']: body=','.join(f"{r['label']}={yn(r['diff'])}" for r in c['rows'])
   else: body=','.join(f"{r['label']}={de(r['diff'])}" for r in c['rows'])
   g=grade_vector('flags='+body,c); self.assertTrue(g['format_valid']); self.assertTrue(g['answer_correct']); self.assertEqual(g['row_accuracy'],1.0)
   cg=grade_count('count='+str(oracle_count(c)),c); self.assertTrue(cg['answer_correct'])
   sg=grade_select('select='+oracle_unique(c),c); self.assertTrue(sg['answer_correct'])
   self.assertIsNone(parse_vector('flags=A=diff',c))
 def test_routes(self):
  r=build_route_tasks(self.cases); self.assertEqual(len(r),12); self.assertEqual(Counter(x['truth'] for x in r),Counter({'comparator':6,'manual':6}))
  for t in r:
   g=grade_route('route='+t['truth'],t); self.assertTrue(g['answer_correct'])
 def test_delegate(self):
  c=self.cases[0]
  rows=';'.join(f"{r['label']}:{int(r['h1'])}/{int(r['h2'])}" for r in c['rows'])
  a=parse_delegate_action('action=compare\nrows='+rows,c)
  self.assertTrue(a['format_valid']); self.assertEqual(a['action'],'compare'); self.assertTrue(operand_exact(a['rows'],c))
  res=comparator_result(a['rows']); self.assertEqual(res['unique'],oracle_unique(c))
  fg=grade_final('final='+oracle_unique(c),c); self.assertTrue(fg['answer_correct'])
  da=parse_delegate_action('action=answer\nanswer='+oracle_unique(c),c); self.assertTrue(da['format_valid'])
  bad=parse_delegate_action('action=compare\nrows=A:2/0',c); self.assertFalse(bad['format_valid'])
 def test_epi_logic(self):
  self.assertEqual(len(self.epis),12)
  self.assertEqual(Counter(t['family'] for t in self.epis),Counter({'need_unique':2,'need_multiple_equal':2,'need_multiple_utility':2,'resolved_still_diff':2,'consistent_nodisc':2,'resolved_nodisc':2}))
  for t in self.epis:
   if t['family']=='resolved_still_diff':
    self.assertEqual(t['acceptable'],['no_query']); self.assertTrue(any(r['diff'] for r in t['rows']))
   if t['family']=='need_multiple_equal': self.assertEqual(len(t['acceptable']),2)
   if t['family']=='need_multiple_utility': self.assertEqual(len(t['acceptable']),1)
   g=grade_epi('choice='+t['acceptable'][0],t); self.assertTrue(g['answer_correct'])
 def test_summary(self):
  c=self.cases[0]
  payload={'singles':[grade_single('relation=different',self.singles[1])],
           'vector':[dict({'family':c['family']},**grade_vector('flags='+','.join(f"{r['label']}={yn(r['diff']) if c['variant']['yesno'] else de(r['diff'])}" for r in c['rows']),c))],
           'count':[dict({'family':c['family']},**grade_count('count='+str(oracle_count(c)),c))],
           'select':[dict({'family':c['family']},**grade_select('select='+oracle_unique(c),c))],
           'epis':[dict({'family':self.epis[0]['family']},**grade_epi('choice='+self.epis[0]['acceptable'][0],self.epis[0]))]}
  s=summarize_epi4(payload); self.assertEqual(s['AUTO_REFERENCE']['accuracy'],1.0); self.assertEqual(s['VECTOR']['accuracy'],1.0)
if __name__=='__main__': unittest.main()
