import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from jarvis_research.gate1_6_epistemic import *

class TestEpistemic(unittest.TestCase):
 def test_battery(self):
  t=build_epistemic_tasks(); self.assertEqual(len(t),24); self.assertEqual(t,build_epistemic_tasks())
  self.assertEqual(sum(x['family']=='need_query' for x in t),8)
  self.assertEqual(sum(x['truth']=='no_query' for x in t),16)
  for x in t:
   div=[q for q in x['queries'] if q['discriminates']]
   if x['family']=='need_query': self.assertEqual(len(div),1); self.assertEqual(div[0]['label'],x['truth'])
   else: self.assertEqual(len(div),0)
 def test_render_parse(self):
  t=build_epistemic_tasks()[0]; p=render_epistemic_task(t,'NOCOST')
  self.assertIn('Rule 1:',p); self.assertIn('A query costs',render_epistemic_task(t,'COST'))
  self.assertEqual(parse_choice('choice='+t['truth'],t),t['truth'])
  self.assertIsNone(parse_choice('I choose '+t['truth'],t))
