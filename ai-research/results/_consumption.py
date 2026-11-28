"""HZN-EPI-06 typed-state consumption probe: report versus decision versus mechanical render."""
import hashlib,json,sys,time
sys.path.insert(0,r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from jarvis_research.gate1_pilot import MODEL_ID,DTYPE,local_provenance
from jarvis_research.gate1_7_substrate import render_hybrid_matrix
from jarvis_research.gate1_8_consumption import *

OUT=Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research\results\gate1_8_consumption_v1.json")
SYSTEM="You are operating in an exact typed cognitive architecture experiment. Follow the requested output format exactly — no extra text."
MAX_GENERATIONS=130; MAX_WALL_SECONDS=20*60


def sha(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def matrix_meta(sc):
    return {"scenario_id":sc['scenario_id'],"family":sc['family'],"state_group":sc['state_group'],
            "expected_report":sc['expected_report'],"expected_action":sc['expected_action'],
            "effective_status":sc['effective_status'],"input_status":sc['input_status'],
            "n_dispatch_events":sc['n_dispatch_events'],"naive_answer":sc['naive_answer'],
            "cached_unique":sc['cached_unique']}


def epistemic_meta(sc):
    return {"scenario_id":sc['scenario_id'],"family":sc['family'],"expected_action":sc['expected_action'],
            "acceptable_choices":list(sc['acceptable_choices']),"n_dispatch_events":sc['n_dispatch_events']}


def main():
    matrix_scenarios=build_matrix_state_scenarios(); epistemic_scenarios=build_epistemic_decision_scenarios()
    report={"model":MODEL_ID,"harness":"epi6-consumption-1","manifest":{},
            "readout_baseline":[],"readout_repaired":[],"decision_matrix":[],"decision_epistemic":[],
            "mechanical_matrix_render":[],"mechanical_epistemic_render":[],"summary":{}}
    if OUT.exists(): report=json.loads(OUT.read_text(encoding='utf-8'))
    def save(): OUT.write_text(json.dumps(report,indent=1),encoding='utf-8')
    t0=time.perf_counter(); state={'ng':0}
    def over(): return state['ng']>=MAX_GENERATIONS or time.perf_counter()-t0>MAX_WALL_SECONDS
    # Arm D deterministic rendering is executed before any neural interpretation.
    if not report['mechanical_matrix_render']:
        report['mechanical_matrix_render']=[render_mechanical_matrix(s) for s in matrix_scenarios]; save()
    if not report['mechanical_epistemic_render']:
        report['mechanical_epistemic_render']=[render_mechanical_epistemic(s) for s in epistemic_scenarios]; save()
    prov=local_provenance(MODEL_ID)
    report['manifest']={"model_revision":prov['revision'],"consumption_version":CONSUMPTION_VERSION,
                        "matrix_state_scenarios":len(matrix_scenarios),"resolved_matrix_cases":18,
                        "nonnormal_cases":18,"stale_cases":1,"epistemic_decision_tasks":len(epistemic_scenarios),
                        "robust_base_case_ids":list(ROBUST_BASE_CASE_IDS),"arms":list(CONSUMPTION_ARMS),
                        "actions":list(ACTIONS),"substrate_note":"ambiguous derived states now stale on dependency invalidation before recomputation"}
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
    # Arm A — exact EPI-05 hybrid readout prompts.
    have={r['scenario_id'] for r in report['readout_baseline']}
    for sc in matrix_scenarios:
        if sc['scenario_id'] in have or over(): continue
        p=render_hybrid_matrix(sc['case'],sc['bus']); resp,ni,no,lat=gen(p,16); g=grade_readout(resp,sc)
        g.update(meta(resp,ni,no,lat,p)); g.update(matrix_meta(sc))
        report['readout_baseline'].append(g); save()
        print('[readout-base]',sc['scenario_id'],g['strict_answer'],g['expected'],g['strict_correct'],g['failure_class'],flush=True)
    # Arm B — few-shot output-contract repair over identical typed states.
    have={r['scenario_id'] for r in report['readout_repaired']}
    for sc in matrix_scenarios:
        if sc['scenario_id'] in have or over(): continue
        p=render_readout_repaired(sc); resp,ni,no,lat=gen(p,16); g=grade_readout(resp,sc)
        g.update(meta(resp,ni,no,lat,p)); g.update(matrix_meta(sc))
        report['readout_repaired'].append(g); save()
        print('[readout-repaired]',sc['scenario_id'],g['strict_answer'],g['expected'],g['strict_correct'],g['failure_class'],flush=True)
    # Arm C — decision-only consumption of matrix state; no mechanical content is reported.
    have={r['scenario_id'] for r in report['decision_matrix']}
    for sc in matrix_scenarios:
        if sc['scenario_id'] in have or over(): continue
        p=render_matrix_decision(sc); resp,ni,no,lat=gen(p,12); g=grade_matrix_decision(resp,sc)
        g.update(meta(resp,ni,no,lat,p)); g.update(matrix_meta(sc))
        report['decision_matrix'].append(g); save()
        print('[decision-matrix]',sc['scenario_id'],g['action'],g['expected_action'],g['action_correct'],g['failure_class'],flush=True)
    # Arm C — decision-only consumption of epistemic state, including tied candidates and restraint controls.
    have={r['scenario_id'] for r in report['decision_epistemic']}
    for sc in epistemic_scenarios:
        if sc['scenario_id'] in have or over(): continue
        p=render_epistemic_decision(sc); resp,ni,no,lat=gen(p,14); g=grade_epistemic_decision(resp,sc)
        g.update(meta(resp,ni,no,lat,p)); g.update(epistemic_meta(sc))
        report['decision_epistemic'].append(g); save()
        print('[decision-epistemic]',sc['scenario_id'],g['action'],g.get('choice'),g['expected_action'],g['action_correct'],g['failure_class'],flush=True)
    # Arm D — only genuine equal-cost ties require neural interpretation; all mechanical facts are rendered mechanically.
    for i,sc in enumerate(epistemic_scenarios):
        if i>=len(report['mechanical_epistemic_render']): break
        rec=report['mechanical_epistemic_render'][i]
        if not rec.get('requires_neural_choice') or rec.get('tie_choice') is not None or over(): continue
        p=render_tie_interpretation_prompt(rec); resp,ni,no,lat=gen(p,8); choice=parse_tie_choice(resp)
        correct=choice in rec['candidates']
        if correct:
            final_text=finalize_tie_render(rec,choice); failure=None
        else:
            final_text=rec['rendered_text']+"\nNo authorized tied observation was selected."
            failure='hallucinated-content' if choice is None else 'state-ignore'
        rec.update({"tie_choice":choice,"tie_choice_correct":correct,"tie_format_valid":choice is not None,
                    "failure_class":failure,"final_rendered_text":final_text,"end_to_end_correct":correct,
                    "response":resp,"input_tokens":ni,"output_tokens":no,"latency_s":round(lat,2),
                    "prompt_sha256":sha(p),"neural_input_tokens":ni,"neural_output_tokens":no})
        save(); print('[mechanical-tie]',sc['scenario_id'],choice,rec['candidates'],correct,failure,flush=True)
    report['summary']=summarize_epi6(report)
    report['wall_seconds']=round(time.perf_counter()-t0,1); report['n_generations']=state['ng']; save()
    print(json.dumps(report['summary'],indent=1),flush=True)

if __name__=='__main__': main()
