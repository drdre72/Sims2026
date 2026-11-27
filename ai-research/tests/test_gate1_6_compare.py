import sys,unittest
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from jarvis_research.gate1_6_compare import *

class TestCompare(unittest.TestCase):
 def test_battery(self):
  t=build_comparison_tasks(); self.assertEqual(len(t),24); self.assertEqual(t,build_comparison_tasks())
  self.assertEqual(Counter(x['family'] for x in t),Counter({'unique_diff':12,'no_diff':6,'multiple_diff':6}))
  self.assertEqual(sum(x['truth']=='none' for x in t),12)
  for x in t:
   nd=sum(r['diff'] for r in x['rows'])
   if x['family']=='unique_diff': self.assertEqual(nd,1); self.assertTrue(x['rows'][x['target_index']]['diff'])
   elif x['family']=='no_diff': self.assertEqual(nd,0)
   else: self.assertEqual(nd,2)
 def test_render_parse_grade(self):
  for cond in COMPARE_CONDITIONS:
   for t in build_comparison_tasks()[:3]:
    p=render_compare_task(t,cond)
    if cond=='RAW': resp='answer='+t['truth']
    else: resp='choice='+(t['truth'] if t['truth']!='none' else 'no_query')
    g=grade_compare_response(resp,t,cond)
    self.assertTrue(g['format_valid']); self.assertTrue(g['answer_correct'])
   self.assertIsNone(parse_compare_response('hello',cond))
 def test_summary(self):
  t=build_comparison_tasks()[0]; cell={'family':t['family'],'truth':t['truth'],'conditions':{}}
  for cond in COMPARE_CONDITIONS: cell['conditions'][cond]=grade_compare_response(('answer=' if cond=='RAW' else 'choice=')+t['truth'],t,cond)
  s=summarize_compare_cells([cell]); self.assertEqual(s['RAW']['accuracy'],1.0)
if __name__=='__main__': unittest.main()
