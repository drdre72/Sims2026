"""Posthoc HZN-EPI-04 supplementary deliberative-budget probe (600 tokens)."""
import hashlib,json,sys,time
sys.path.insert(0,r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from jarvis_research.gate1_pilot import MODEL_ID,DTYPE,local_provenance
from jarvis_research.gate1_6_decompose import build_matrix_cases,render_vector,grade_vector
OUT=Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research\results\gate1_6_decompose_delib600.json")
SYSTEM="You may reason internally. End with exactly the requested output line."
def extract_line(resp,prefix):
 lines=[x.strip() for x in str(resp).splitlines() if x.strip().startswith(prefix)]
 return lines[-1] if lines else resp
prov=local_provenance(MODEL_ID)
tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=prov['revision'],local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=prov['revision'],dtype=DTYPE,device_map={"":"cuda"},local_files_only=True); model.eval()
rows=[]; t0=time.perf_counter()
for case in [c for c in build_matrix_cases() if c['family']=='unique_diff'][:6]:
 p=render_vector(case)
 enc=tok.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":p}],add_generation_prompt=True,return_tensors="pt",return_dict=True,enable_thinking=True)
 enc={k:v.to(model.device) for k,v in enc.items()}; ni=int(enc['input_ids'].shape[1]); t=time.perf_counter()
 with torch.no_grad(): out=model.generate(**enc,max_new_tokens=600,do_sample=False,pad_token_id=tok.eos_token_id)
 lat=time.perf_counter()-t; ids=out[0][enc['input_ids'].shape[1]:]; resp=tok.decode(ids,skip_special_tokens=True); clean=extract_line(resp,'flags=')
 g=grade_vector(clean,case); g.update({"case_id":case['case_id'],"family":case['family'],"truth":case['truth'],"rows":case['rows'],"response":resp,"cleaned_response":clean,"input_tokens":ni,"output_tokens":int(ids.shape[0]),"latency_s":round(lat,2),"prompt_sha256":hashlib.sha256(p.encode()).hexdigest()})
 rows.append(g); print(case['case_id'],g['format_valid'],g['answer_correct'],int(ids.shape[0]),flush=True)
OUT.write_text(json.dumps({"model":MODEL_ID,"harness":"epi4-delib600-posthoc","rows":rows,"summary":{"n":len(rows),"accuracy":sum(r['answer_correct'] for r in rows)/len(rows),"format_valid_rate":sum(r['format_valid'] for r in rows)/len(rows),"mean_output_tokens":sum(r['output_tokens'] for r in rows)/len(rows),"mean_latency_s":sum(r['latency_s'] for r in rows)/len(rows)},"wall_seconds":round(time.perf_counter()-t0,1)},indent=1),encoding='utf-8')
print('done',flush=True)
