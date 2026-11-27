import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from jarvis_research.gate1_6_epistemic_matrix import *

class TestMatrix(unittest.TestCase):
 def test_reference_roundtrip_all_tasks(self):
  for t in build_matrix_tasks():
   art=render_reference_matrix(t)
   g=grade_matrix_artifact(art,t)
   self.assertTrue(g['format_valid'],t['task_id'])
   self.assertTrue(g['matrix_exact'],t['task_id'])
   self.assertTrue(g['answer_correct'],t['task_id'])
   self.assertTrue(g['delegated_correct'],t['task_id'])
   self.assertTrue(g['reference_matrix_exact'],t['task_id'])
 def test_wrong_neural_choice_can_be_repaired(self):
  t=next(x for x in build_matrix_tasks() if x['family']=='need_query')
  art=render_reference_matrix(t,choice='no_query')
  g=grade_matrix_artifact(art,t)
  self.assertFalse(g['answer_correct']); self.assertTrue(g['delegated_correct'])
 def test_bad_matrix_delegates_wrong_or_no_query(self):
  t=next(x for x in build_matrix_tasks() if x['family']=='need_query')
  lines=render_reference_matrix(t).splitlines(); lines[1]=lines[1].replace('yes','no',1) if 'yes' in lines[1] else lines[1].replace('no','yes',1)
  g=grade_matrix_artifact('\n'.join(lines),t)
  self.assertTrue(g['format_valid']); self.assertFalse(g['matrix_exact'])
 def test_invalid_contract(self):
  t=build_matrix_tasks()[0]
  for bad in ('choice=Q1',render_reference_matrix(t)+'\nextra'):
   self.assertNotEqual(parse_matrix_artifact(bad,t)['parse_status'],'valid')
 def test_summary(self):
  cells=[]
  for t in build_matrix_tasks()[:2]:
   g=grade_matrix_artifact(render_reference_matrix(t),t)
   cells.append({'family':t['family'],'conditions':{'BASE':{'answer_correct':False,'format_valid':True},'XTRACT':g,'ORACLEPRED':{'answer_correct':True,'format_valid':True}}})
  s=summarize_matrix_cells(cells)
  self.assertEqual(s['XTRACT_DELEGATED']['accuracy'],1.0)
  self.assertEqual(s['XTRACT_ARTIFACT']['matrix_exact_rate'],1.0)
if __name__=='__main__': unittest.main()
