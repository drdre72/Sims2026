"""HZN-EPI-04 comparative decomposition/invocation probe."""
import hashlib,json,sys,time
sys.path.insert(0,r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from jarvis_research.gate1_pilot import MODEL_ID,DTYPE,local_provenance
from jarvis_research.gate1_6_decompose import *
OUT=Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research\results\gate1_6_decompose_v1.json")
SYSTEM="You are performing exact comparison and routing tasks. Follow the requested output format exactly — no extra text."
DELIB_SYSTEM="You may reason internally. End with exactly the requested output line."
MAX_GENERATIONS=150; MAX_WALL_SECONDS=20*60

def extract_line(resp,prefix):
 lines=[x.strip() for x in str(resp).splitlines() if x.strip().startswith(prefix)]
 return lines[-1] if lines else resp

def main():
 matrix=build_matrix_cases(); singles=build_single_tasks(); routes=build_route_tasks(matrix); epis=build_epi_tasks()
 report={"model":MODEL_ID,"harness":"epi4-decompose-1","manifest":{},"singles":[],"vector":[],"count":[],"select":[],"routes":[],"delegates":[],"vector_delib":[],"epis":[],"summary":{}}
 if OUT.exists(): report=json.loads(OUT.read_text(encoding='utf-8'))
 prov=local_provenance(MODEL_ID); report['manifest']={"model_revision":prov['revision'],"decompose_version":DECOMPOSE_VERSION,"matrix_cases":len(matrix),"single_tasks":len(singles),"route_tasks":len(routes),"epi_tasks":len(epis)}
 tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=prov['revision'],local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=prov['revision'],dtype=DTYPE,device_map={"":"cuda"},local_files_only=True); model.eval()
 t0=time.perf_counter(); state={'ng':0}
 def over(): return state['ng']>=MAX_GENERATIONS or time.perf_counter()-t0>MAX_WALL_SECONDS
 def gen(prompt,max_tokens=16,thinking=False):
  system=DELIB_SYSTEM if thinking else SYSTEM
  enc=tok.apply_chat_template([{"role":"system","content":system},{"role":"user","content":prompt}],add_generation_prompt=True,return_tensors="pt",return_dict=True,enable_thinking=thinking)
  enc={k:v.to(model.device) for k,v in enc.items()}; ni=int(enc['input_ids'].shape[1]); t=time.perf_counter()
  with torch.no_grad(): out=model.generate(**enc,max_new_tokens=max_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
  lat=time.perf_counter()-t; ids=out[0][enc['input_ids'].shape[1]:]
  state['ng']+=1
  return tok.decode(ids,skip_special_tokens=True),ni,int(ids.shape[0]),lat
 def meta(resp,ni,no,lat,prompt):
  return {"response":resp,"input_tokens":ni,"output_tokens":no,"latency_s":round(lat,2),"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()}
 def save(): OUT.write_text(json.dumps(report,indent=1),encoding='utf-8')
 # SINGLE
 have={r['task_id'] for r in report['singles']}
 for task in singles:
  if task['task_id'] in have or over(): continue
  p=render_single(task); resp,ni,no,lat=gen(p,16); g=grade_single(resp,task); g.update(meta(resp,ni,no,lat,p)); g.update({"task_id":task['task_id'],"truth":task['truth'],"variant":task['variant']})
  report['singles'].append(g); save(); print('[single]',task['task_id'],g['answer'],g['expected'],g['answer_correct'],flush=True)
 # VECTOR / COUNT / SELECT on shared matrix cases
 for cond,renderer,grader,maxtok in (("vector",render_vector,grade_vector,64),("count",render_count,grade_count,16),("select",render_select,grade_select,16)):
  have={r['case_id'] for r in report[cond]}
  for case in matrix:
   if case['case_id'] in have or over(): continue
   p=renderer(case); resp,ni,no,lat=gen(p,maxtok); g=grader(resp,case); g.update(meta(resp,ni,no,lat,p)); g.update({"case_id":case['case_id'],"family":case['family'],"truth":case['truth'],"variant":case['variant'],"rows":case['rows']})
   report[cond].append(g); save(); print('['+cond+']',case['case_id'],case['family'],g['answer_correct'],flush=True)
 # ROUTE
 have={r['task_id'] for r in report['routes']}
 for task in routes:
  if task['task_id'] in have or over(): continue
  p=render_route_task(task); resp,ni,no,lat=gen(p,16); g=grade_route(resp,task); g.update(meta(resp,ni,no,lat,p)); g.update({"task_id":task['task_id'],"kind":task['kind']})
  report['routes'].append(g); save(); print('[route]',task['task_id'],task['kind'],g['answer'],g['answer_correct'],flush=True)
 # DELEGATE: first 8 unique plus first two no/multiple cases.
 unique=[c for c in matrix if c['family']=='unique_diff'][:8]
 controls=[c for c in matrix if c['family']=='no_diff'][:2]+[c for c in matrix if c['family']=='multiple_diff'][:2]
 dcases=unique+controls
 have={r['case_id'] for r in report['delegates']}
 for case in dcases:
  if case['case_id'] in have or over(): continue
  p=render_delegate(case); resp,ni,no,lat=gen(p,80); action=parse_delegate_action(resp,case)
  rec={"case_id":case['case_id'],"family":case['family'],"truth":case['truth'],"rows":case['rows'],"variant":case['variant'],
       "first_response":resp,"first_format_valid":action['format_valid'],"action":action.get('action'),
       "first_input_tokens":ni,"first_output_tokens":no,"first_latency_s":round(lat,2),"prompt_sha256":hashlib.sha256(p.encode()).hexdigest()}
  if action.get('action')=='answer':
   rec.update({"answer":action['answer'],"final":action['answer'],"final_correct":action['answer']==case['truth'],"stage":"direct"})
  elif action.get('action')=='compare':
   rec['operand_exact']=operand_exact(action['rows'],case); result=comparator_result(action['rows']); rec['comparator_result']=result
   if not over():
    p2=render_delegate_result(result); resp2,ni,no,lat=gen(p2,16); fg=grade_final(resp2,case)
    rec.update({"second_response":resp2,"final":fg['final'],"final_correct":fg['answer_correct'],"second_format_valid":fg['format_valid'],
                "second_input_tokens":ni,"second_output_tokens":no,"second_latency_s":round(lat,2),"stage":"consumed"})
   else: rec.update({"final_correct":False,"stage":"budget_stopped"})
  else: rec.update({"final_correct":False,"stage":"invalid_action"})
  report['delegates'].append(rec); save(); print('[delegate]',case['case_id'],rec['action'],rec.get('final'),rec['truth'],rec['final_correct'],flush=True)
 # Deliberative VECTOR reference: six unique cases only.
 have={r['case_id'] for r in report['vector_delib']}
 for case in [c for c in matrix if c['family']=='unique_diff'][:6]:
  if case['case_id'] in have or over(): continue
  p=render_vector(case); resp,ni,no,lat=gen(p,300,thinking=True); cleaned=extract_line(resp,'flags=')
  g=grade_vector(cleaned,case); g.update(meta(resp,ni,no,lat,p)); g.update({"cleaned_response":cleaned,"case_id":case['case_id'],"family":case['family'],"truth":case['truth'],"variant":case['variant'],"rows":case['rows']})
  report['vector_delib'].append(g); save(); print('[vector-delib]',case['case_id'],g['answer_correct'],flush=True)
 # Epistemic consistency/discrimination/utility.
 have={r['task_id'] for r in report['epis']}
 for task in epis:
  if task['task_id'] in have or over(): continue
  p=render_epi_task(task); resp,ni,no,lat=gen(p,16); g=grade_epi(resp,task); g.update(meta(resp,ni,no,lat,p)); g.update({"task_id":task['task_id'],"family":task['family'],"rows":task['rows'],"consistent":task['consistent']})
  report['epis'].append(g); save(); print('[epi]',task['task_id'],task['family'],g['answer'],g['acceptable'],g['answer_correct'],flush=True)
 report['summary']=summarize_epi4(report)
 # Add lightweight cost summaries for generation-bearing lists.
 keymap={'singles':'SINGLE','vector':'VECTOR','count':'COUNT','select':'SELECT','routes':'ROUTE','vector_delib':'VECTOR_DELIB','epis':'EPIUTIL'}
 for key in keymap:
  rows=report.get(key,[])
  if rows and keymap[key] in report['summary']:
   report['summary'][keymap[key]]['mean_latency_s']=sum(r['latency_s'] for r in rows)/len(rows)
 report['wall_seconds']=round(time.perf_counter()-t0,1); report['n_generations']=state['ng']; save()
 print(json.dumps(report['summary'],indent=1),flush=True)
if __name__=='__main__': main()
