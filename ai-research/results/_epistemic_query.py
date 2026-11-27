"""HZN-EPI-01 discriminating-query probe: 24 tasks x 2 cost conditions."""
import hashlib,json,sys,time
sys.path.insert(0,r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from jarvis_research.gate1_pilot import MODEL_ID,DTYPE,local_provenance
from jarvis_research.gate1_6_epistemic import CONDITIONS,EPI_VERSION,build_epistemic_tasks,parse_choice,render_epistemic_task
OUT=Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research\results\gate1_6_epistemic_query_v1.json")
SYSTEM="You are evaluating epistemic uncertainty. Follow the requested output format exactly."
MAX_GENERATIONS=56; MAX_WALL_SECONDS=10*60

def gen(model,tok,prompt):
 enc=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],add_generation_prompt=True,return_tensors="pt",return_dict=True,enable_thinking=False)
 enc={k:v.to(model.device) for k,v in enc.items()}; n_in=int(enc['input_ids'].shape[1]); t=time.perf_counter()
 with torch.no_grad(): out=model.generate(**enc,max_new_tokens=16,do_sample=False,pad_token_id=tok.eos_token_id)
 lat=time.perf_counter()-t; ids=out[0][enc['input_ids'].shape[1]:]
 return tok.decode(ids,skip_special_tokens=True),n_in,int(ids.shape[0]),lat

def main():
 tasks=build_epistemic_tasks(); report={"model":MODEL_ID,"harness":"epi-query-1","manifest":{},"cells":{},"summary":{}}
 if OUT.exists(): report=json.loads(OUT.read_text(encoding='utf-8'))
 prov=local_provenance(MODEL_ID); report['manifest']={"model_revision":prov['revision'],"epi_version":EPI_VERSION,"n_tasks":len(tasks),"conditions":list(CONDITIONS)}
 tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=prov['revision'],local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=prov['revision'],dtype=DTYPE,device_map={"":"cuda"},local_files_only=True); model.eval()
 t0=time.perf_counter(); ng=0
 for task in tasks:
  cell=report['cells'].setdefault(task['task_id'],{"family":task['family'],"truth":task['truth'],"pair":task['pair'],"conditions":{}})
  for cond in CONDITIONS:
   if cond in cell['conditions'] or ng>=MAX_GENERATIONS or time.perf_counter()-t0>MAX_WALL_SECONDS: continue
   prompt=render_epistemic_task(task,cond); resp,ni,no,lat=gen(model,tok,prompt); ng+=1
   choice=parse_choice(resp,task)
   cell['conditions'][cond]={"response":resp,"choice":choice,"correct":choice==task['truth'],"format_valid":choice is not None,"input_tokens":ni,"output_tokens":no,"latency_s":round(lat,2),"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()}
   OUT.write_text(json.dumps(report,indent=1),encoding='utf-8')
   print(f"[{task['task_id']}:{cond}] {choice} exp={task['truth']} ok={choice==task['truth']}",flush=True)
 for cond in CONDITIONS:
  rows=[]; fams={}
  for c in report['cells'].values():
   if cond in c['conditions']:
    rows.append((c,c['conditions'][cond])); fams.setdefault(c['family'],[]).append(c['conditions'][cond]['correct'])
  report['summary'][cond]={"n":len(rows),"accuracy":sum(r['correct'] for c,r in rows)/len(rows),"format_valid_rate":sum(r['format_valid'] for c,r in rows)/len(rows),"family_accuracy":{k:sum(v)/len(v) for k,v in fams.items()},"query_rate_need":sum(r['choice']!='no_query' for c,r in rows if c['family']=='need_query')/8,"query_rate_control":sum(r['choice']!='no_query' for c,r in rows if c['family']!='need_query')/16,"mean_input_tokens":sum(r['input_tokens'] for c,r in rows)/len(rows),"mean_latency_s":sum(r['latency_s'] for c,r in rows)/len(rows)}
 report['wall_seconds']=round(time.perf_counter()-t0,1); report['n_generations']=ng
 OUT.write_text(json.dumps(report,indent=1),encoding='utf-8')
 print(json.dumps(report['summary'],indent=1),flush=True)
if __name__=='__main__': main()
