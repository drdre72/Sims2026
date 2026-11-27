"""HZN-EPI-05 typed cognitive substrate/structural-binding probe."""
import hashlib,json,sys,time
sys.path.insert(0,r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from jarvis_research.gate1_pilot import MODEL_ID,DTYPE,local_provenance
from jarvis_research.gate1_6_decompose import render_delegate,parse_delegate_action,operand_exact
from jarvis_research.gate1_7_substrate import *

OUT=Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research\results\gate1_7_substrate_v1.json")
SYSTEM="You are operating in an exact typed cognitive architecture experiment. Follow the requested output format exactly — no extra text."
MAX_GENERATIONS=130; MAX_WALL_SECONDS=20*60


def sha(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def row_copy_accuracy(call_rows,case):
    if not call_rows: return 0.0
    vals=[]
    for r in case['rows']:
        got=call_rows.get(r['label'])
        vals.append(got==(r['h1'],r['h2']))
    return sum(vals)/len(vals)


def main():
    matrix=build_matrix_cases(); epis=build_epi_tasks(); robust=build_robustness_cases()
    composition=build_composition_cases(); rule_tasks=build_rule_runtime_tasks()
    report={"model":MODEL_ID,"harness":"epi5-substrate-1","manifest":{},
            "matrix_direct":[],"matrix_invocations":[],"matrix_hybrid":[],
            "matrix_auto":[],"robustness_auto":[],"robustness_hybrid":[],"stale_hybrid":[],"composition_auto":[],
            "epistemic_direct":[],"epistemic_auto":[],"epistemic_hybrid":[],"rule_runtime":[],
            "stale_probe":None,"summary":{}}
    if OUT.exists(): report=json.loads(OUT.read_text(encoding='utf-8'))
    def save(): OUT.write_text(json.dumps(report,indent=1),encoding='utf-8')
    t0=time.perf_counter(); state={'ng':0}
    def over(): return state['ng']>=MAX_GENERATIONS or time.perf_counter()-t0>MAX_WALL_SECONDS
    # Deterministic runtime references are executed and graded, not declared.
    if not report['matrix_auto']:
        report['matrix_auto']=[run_auto_matrix_case(c) for c in matrix]; save()
    if not report['robustness_auto']:
        report['robustness_auto']=[run_auto_robustness_case(c) for c in robust]; save()
    if not report['composition_auto']:
        report['composition_auto']=[run_auto_matrix_case(c) for c in composition]; save()
    if not report['epistemic_auto']:
        report['epistemic_auto']=[run_auto_epistemic_case(t) for t in epis]; save()
    if not report['rule_runtime']:
        report['rule_runtime']=[run_auto_rule_task(t) for t in rule_tasks]; save()
    if report['stale_probe'] is None:
        report['stale_probe']=run_stale_invalidation_probe(); save()
    prov=local_provenance(MODEL_ID)
    report['manifest']={"model_revision":prov['revision'],"substrate_version":SUBSTRATE_VERSION,
                        "matrix_cases":len(matrix),"robustness_cases":len(robust),"composition_cases":len(composition),
                        "epistemic_tasks":len(epis),"rule_runtime_tasks":len(rule_tasks),
                        "primary_arms":list(PRIMARY_ARMS),"diagnostic_arms":list(DIAGNOSTIC_ARMS)}
    tok=AutoTokenizer.from_pretrained(MODEL_ID,revision=prov['revision'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=prov['revision'],dtype=DTYPE,device_map={"":"cuda"},local_files_only=True); model.eval()
    def gen_messages(messages,max_tokens=16):
        enc=tok.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt",return_dict=True,enable_thinking=False)
        enc={k:v.to(model.device) for k,v in enc.items()}; ni=int(enc['input_ids'].shape[1]); t=time.perf_counter()
        with torch.no_grad(): out=model.generate(**enc,max_new_tokens=max_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
        lat=time.perf_counter()-t; ids=out[0][enc['input_ids'].shape[1]:]
        state['ng']+=1
        return tok.decode(ids,skip_special_tokens=True),ni,int(ids.shape[0]),lat
    def gen(prompt,max_tokens=16):
        return gen_messages([{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],max_tokens)
    def meta(resp,ni,no,lat,prompt):
        return {"response":resp,"input_tokens":ni,"output_tokens":no,"latency_s":round(lat,2),"prompt_sha256":sha(prompt)}
    # A — neural-only matrix decisions.
    have={r['case_id'] for r in report['matrix_direct']}
    for case in matrix:
        if case['case_id'] in have or over(): continue
        p=render_direct_matrix(case); resp,ni,no,lat=gen(p,16); g=grade_matrix_final(resp,case)
        g.update(meta(resp,ni,no,lat,p)); g.update({"case_id":case['case_id'],"family":case['family'],"truth":case['truth'],"variant":case['variant']})
        report['matrix_direct'].append(g); save(); print('[direct]',case['case_id'],g['answer'],g['expected'],g['answer_correct'],flush=True)
    # B/C — shared typed invocation; neural retranscription versus structural binding.
    have={r['case_id'] for r in report['matrix_invocations']}
    for case in matrix:
        if case['case_id'] in have or over(): continue
        p=render_delegate(case); resp,ni,no,lat=gen(p,80); action=parse_delegate_action(resp,case)
        rec={"case_id":case['case_id'],"family":case['family'],"truth":case['truth'],"variant":case['variant'],
             "first_response":resp,"first_format_valid":action['format_valid'],"action":action.get('action'),
             "first_input_tokens":ni,"first_output_tokens":no,"first_latency_s":round(lat,2),"prompt_sha256":sha(p),
             "tool_context_total_latency_s":round(lat,2),"tool_context_total_output_tokens":no}
        if action.get('action')=='answer':
            rec.update({"answer":action['answer'],"bound_final":action['answer'],"bound_correct":action['answer']==case['truth'],
                        "context_final":action['answer'],"context_correct":action['answer']==case['truth'],"stage":"direct"})
        elif action.get('action')=='compare':
            bus,input_obj,events,result=execute_matrix_invocation(action['rows'],case)
            public=comparator_public_result(result)
            truth_rows={r['label']:(r['h1'],r['h2']) for r in case['rows']}
            copy_hits=sum(action['rows'].get(k)==v for k,v in truth_rows.items())/len(truth_rows)
            rec.update({"operand_exact":operand_exact(action['rows'],case),"row_copy_accuracy":copy_hits,
                        "comparator_result":public,"comparator_unique":public['unique'],
                        "dispatch_events":len(events),"bound_final":result['decision'],
                        "bound_correct":result['decision']==case['truth'],"stage":"invoked"})
            # Context-preserving neural retranscription: the original user prompt and the model's own call remain in chat.
            if not over():
                p2=render_tool_result(result,case,context_preserving=True)
                messages=[{"role":"system","content":SYSTEM},{"role":"user","content":p},
                          {"role":"assistant","content":resp},{"role":"user","content":p2}]
                resp2,ni2,no2,lat2=gen_messages(messages,16); fg=grade_matrix_final(resp2,case)
                rec.update({"context_response":resp2,"context_final":fg['answer'],"context_correct":fg['answer_correct'],
                            "context_format_valid":fg['format_valid'],"context_input_tokens":ni2,"context_output_tokens":no2,
                            "context_latency_s":round(lat2,2),"context_prompt_sha256":sha(p2)})
                rec['tool_context_total_latency_s']=round(lat+lat2,2); rec['tool_context_total_output_tokens']=no+no2
            # Fresh-context standalone readout diagnostic, matching the EPI-04 context-loss interpretation.
            if not over():
                p3=render_tool_result(result,case,context_preserving=False)
                resp3,ni3,no3,lat3=gen(p3,16); fg3=grade_matrix_final(resp3,case)
                rec.update({"fresh_response":resp3,"fresh_final":fg3['answer'],"fresh_correct":fg3['answer_correct'],
                            "fresh_format_valid":fg3['format_valid'],"fresh_input_tokens":ni3,"fresh_output_tokens":no3,
                            "fresh_latency_s":round(lat3,2),"fresh_prompt_sha256":sha(p3)})
        else:
            rec.update({"bound_correct":False,"context_correct":False,"stage":"invalid_action"})
        report['matrix_invocations'].append(rec); save()
        print('[invoke]',case['case_id'],rec['action'],rec.get('comparator_unique'),rec.get('bound_final'),rec.get('context_final'),rec['truth'],flush=True)
    # E — hybrid interpretation over read-only automatically bound matrix state.
    have={r['case_id'] for r in report['matrix_hybrid']}
    for case in matrix:
        if case['case_id'] in have or over(): continue
        bus,input_obj,events=load_matrix_bus(case); p=render_hybrid_matrix(case,bus)
        resp,ni,no,lat=gen(p,16); g=grade_hybrid_matrix(resp,case,bus)
        g.update(meta(resp,ni,no,lat,p)); g.update({"case_id":case['case_id'],"family":case['family'],"truth":case['truth'],"n_dispatch_events":len(events)})
        report['matrix_hybrid'].append(g); save(); print('[hybrid-matrix]',case['case_id'],g['answer'],g['expected'],g['answer_correct'],flush=True)
    # Robustness hybrid: malformed/incomplete/contradictory evidence typed by the substrate.
    have={r['case_id'] for r in report['robustness_hybrid']}
    for case in robust:
        if case['case_id'] in have or over(): continue
        bus,input_obj,events=load_robustness_bus(case); p=render_hybrid_matrix(case,bus)
        resp,ni,no,lat=gen(p,16); g=grade_hybrid_matrix(resp,case,bus)
        g.update(meta(resp,ni,no,lat,p)); g.update({"case_id":case['case_id'],"family":case['family'],"expected_status":case['expected_status'],"n_dispatch_events":len(events)})
        report['robustness_hybrid'].append(g); save(); print('[hybrid-robust]',case['case_id'],g['answer'],g['expected'],g['answer_correct'],flush=True)
    # Stale-evidence hybrid reference: dependencies have been invalidated but not yet recomputed.
    if not report['stale_hybrid'] and not over():
        case,bus,input_obj,events=build_stale_hybrid_case(); p=render_hybrid_matrix(case,bus)
        resp,ni,no,lat=gen(p,16); g=grade_hybrid_matrix(resp,case,bus)
        g.update(meta(resp,ni,no,lat,p)); g.update({"case_id":case['case_id'],"family":case['family'],"expected_status":"stale","n_dispatch_events":len(events)})
        report['stale_hybrid'].append(g); save(); print('[hybrid-stale]',g['answer'],g['expected'],g['answer_correct'],flush=True)
    # Epistemic neural-only baseline.
    have={r['task_id'] for r in report['epistemic_direct']}
    for task in epis:
        if task['task_id'] in have or over(): continue
        p=render_epi_task(task); resp,ni,no,lat=gen(p,16); g=grade_epi(resp,task)
        g.update(meta(resp,ni,no,lat,p)); g.update({"task_id":task['task_id'],"family":task['family'],"rows":task['rows'],"consistent":task['consistent']})
        report['epistemic_direct'].append(g); save(); print('[epi-direct]',task['task_id'],task['family'],g['answer'],g['acceptable'],g['answer_correct'],flush=True)
    # Epistemic hybrid over automatic consistency/discrimination/cost state.
    have={r['task_id'] for r in report['epistemic_hybrid']}
    for task in epis:
        if task['task_id'] in have or over(): continue
        bus,input_obj,events=load_epistemic_bus(task); p=render_hybrid_epistemic(task,bus)
        resp,ni,no,lat=gen(p,16); g=grade_hybrid_epistemic(resp,task)
        g.update(meta(resp,ni,no,lat,p)); g.update({"task_id":task['task_id'],"family":task['family'],"n_dispatch_events":len(events)})
        report['epistemic_hybrid'].append(g); save(); print('[epi-hybrid]',task['task_id'],task['family'],g['answer'],g['acceptable'],g['answer_correct'],flush=True)
    report['summary']=summarize_epi5(report)
    report['wall_seconds']=round(time.perf_counter()-t0,1); report['n_generations']=state['ng']; save()
    print(json.dumps(report['summary'],indent=1),flush=True)

if __name__=='__main__': main()
