"""HZN-EPI-03 comparative primitive probe: 24 tasks x 3 framings."""
import hashlib,json,sys,time
sys.path.insert(0,r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from jarvis_research.gate1_pilot import MODEL_ID,DTYPE,local_provenance
from jarvis_research.gate1_6_compare import COMPARE_CONDITIONS,build_comparison_tasks,grade_compare_response,render_compare_task,summarize_compare_cells
OUT=Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research\results\gate1_6_compare_primitive_v1.json")
SYSTEM="You are evaluating binary comparisons. Follow the requested output format exactly — no reasoning, no extra text."
MAX_GENERATIONS=80; MAX_WALL_SECONDS=12*60

def gen(model,tok,prompt):
 enc=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],add_generation_prompt=True,return_tensors="pt",return_dict=True,enable_thinking=False)
 enc={k:v.to(model.device) for k,v in enc.items()}; n_in=int(enc['input_ids'].shape[1]); t=time.perf_counter()
 with torch.no_grad(): out=model.generate(**enc,max_new_tokens=14,do_sample=False,pad_token_id=tok.eos_token_id)
 lat=time.perf_counter()-t; ids=out[0][enc['input_ids'].shape[1]:]
 return tok.decode(ids,skip_special_tokens=True),n_in,int(ids.shape[0]),lat

def main():
 tasks=build_comparison_tasks(); report={"model":MODEL_ID,"harness":"compare-primitive-1","manifest":{},"cells":{},"summary":{}}
 if OUT.exists(): report=json.loads(OUT.read_text(encoding='utf-8'))
 prov=local_provenance(MODEL_ID); report['manifest']={"model_revision":prov['revision'],"compare_version":1,"n_tasks":len(tasks),"conditions":list(COMPARE_CONDITIONS)}
 tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=prov['revision'],local_files_only=True)
 model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=prov['revision'],dtype=DTYPE,device_map={"":"cuda"},local_files_only=True); model.eval()
 t0=time.perf_counter(); ng=0
 for task in tasks:
  cell=report['cells'].setdefault(task['task_id'],{"family":task['family'],"truth":task['truth'],"rows":task['rows'],"conditions":{}})
  for cond in COMPARE_CONDITIONS:
   if cond in cell['conditions'] or ng>=MAX_GENERATIONS or time.perf_counter()-t0>MAX_WALL_SECONDS: continue
   prompt=render_compare_task(task,cond); resp,ni,no,lat=gen(model,tok,prompt); ng+=1
   rec=grade_compare_response(resp,task,cond); rec.update({"response":resp,"input_tokens":ni,"output_tokens":no,"latency_s":round(lat,2),"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()})
   cell['conditions'][cond]=rec; OUT.write_text(json.dumps(report,indent=1),encoding='utf-8')
   print(f"[{task['task_id']}:{cond}] ans={rec['answer']} exp={task['truth']} ok={rec['answer_correct']}",flush=True)
 report['summary']=summarize_compare_cells(report['cells'].values())
 for cond in COMPARE_CONDITIONS:
  rows=[c['conditions'][cond] for c in report['cells'].values() if cond in c['conditions']]
  report['summary'][cond]['mean_input_tokens']=sum(r['input_tokens'] for r in rows)/len(rows)
  report['summary'][cond]['mean_latency_s']=sum(r['latency_s'] for r in rows)/len(rows)
 report['wall_seconds']=round(time.perf_counter()-t0,1); report['n_generations']=ng
 OUT.write_text(json.dumps(report,indent=1),encoding='utf-8')
 print(json.dumps(report['summary'],indent=1),flush=True)
if __name__=='__main__': main()
