"""Encoding probe: is the English rule prose necessary?
Same 12 tasks as the candidate-reasoning sentinel, T2 condition only,
under two alternative renderings. Paired with existing T2 results.
  T2sym: symbolic rule + compact labeled data
  T2pos: symbolic rule + positional data (schema declared once)
"""
import hashlib, json, random, sys, time
sys.path.insert(0, r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path

AI_RESEARCH = Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research")
OUT = AI_RESEARCH / "results" / "gate1_4_encoding_probe.json"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jarvis_research.gate1_4_benchmark import RULES, FAMILIES, grade_task, make_task
from jarvis_research.gate1_pilot import MODEL_ID, DTYPE, local_provenance
from jarvis_research.run_manifest import validate_resume_manifest

N_TASKS = 12
HARNESS_VERSION = "encoding-probe-1"
SYSTEM = ("You are evaluating intervention candidates in a self-editing "
          "experiment. Apply the stated rule exactly. Reply with ONLY "
          "the requested format — no reasoning, no extra text.")

SYM_RULES = {
    "conj_active": "select:ar>={ar}&|lc|>={lc};tie-break:highest ar;exact-tie-or-none:none",
    "conj_strong": "select:ar>={ar}&|lc|>={lc};tie-break:largest|lc|;exact-tie-or-none:none",
    "avoid_negative": "select:lc>=0&ar>={ar};among-qualifying:largest|lc|;none-qualify:none",
}

def render_sym(task):
    r = RULES[task["rule_key"]]
    rule = SYM_RULES[task["rule_key"]].format(ar=r["ar_thr"], lc=r["lc_thr"])
    lines = []
    for c in task["candidates"]:
        if c["obs_available"]:
            lines.append(f"{c['label']}|ar={c['ar']:.2f}|lc={c['lc']:.2f}")
        else:
            lines.append(f"{c['label']}|ar=NA|lc=NA")
    return (rule + "\n" + "\n".join(lines) +
            "\nReply with ONLY the candidate letter, or the word none.")

def render_pos(task):
    r = RULES[task["rule_key"]]
    rule = SYM_RULES[task["rule_key"]].format(ar=r["ar_thr"], lc=r["lc_thr"])
    lines = ["fields:ar,lc"]
    for c in task["candidates"]:
        if c["obs_available"]:
            lines.append(f"{c['label']}|{c['ar']:.2f}|{c['lc']:.2f}")
        else:
            lines.append(f"{c['label']}|NA|NA")
    return (rule + "\n" + "\n".join(lines) +
            "\nReply with ONLY the candidate letter, or the word none.")

def build_tasks():
    tasks = []
    for i in range(N_TASKS):
        fam = FAMILIES[i % len(FAMILIES)]
        rng = random.Random(int.from_bytes(hashlib.sha256(
            f"t5-task:{i}".encode()).digest()[:8], "little"))
        tasks.append(make_task(rng, fam))
    return tasks

def gen_text(model, tok, user_text, max_new=16):
    enc = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user_text}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    n_in = int(enc["input_ids"].shape[1])
    sdpa = torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
         torch.nn.attention.SDPBackend.MATH])
    t0 = time.perf_counter()
    with sdpa, torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    lat = time.perf_counter() - t0
    return (tok.decode(out[0][enc["input_ids"].shape[1]:],
                       skip_special_tokens=True), n_in, lat)

def grade(resp, expected):
    t = resp.strip()
    if expected == "none":
        return "correct" if t.lower() == "none" else "invalid_pick"
    if t.lower() == "none":
        return "abstain_when_answer"
    return "correct" if t.upper() == str(expected) else "valid_but_incorrect"

def main():
    report = {"model": MODEL_ID, "harness": HARNESS_VERSION,
              "manifest": {}, "cells": {}}
    stored = {}
    if OUT.exists():
        report = json.loads(OUT.read_text())
        stored = dict(report.get("manifest", {}))
        print(f"[resume] {len(report['cells'])} cells", flush=True)
    provenance = local_provenance(MODEL_ID)
    report["manifest"] = {"model_revision": provenance["revision"],
                          "harness_version": HARNESS_VERSION,
                          "n_tasks": N_TASKS}
    if stored:
        errs = validate_resume_manifest(stored, report["manifest"])
        if errs:
            print("REFUSING resume - manifest mismatch", flush=True)
            raise SystemExit(1)
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=provenance["revision"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=provenance["revision"], dtype=DTYPE,
        device_map={"": "cuda"}, local_files_only=True)
    model.eval()

    tasks = build_tasks()
    for ti, task in enumerate(tasks):
        truth = task["truth"]["answer"]
        for cond, renderer in (("T2sym", render_sym), ("T2pos", render_pos)):
            key = f"task{ti}:{cond}"
            if key in report["cells"] and "verdict" in report["cells"][key]:
                continue
            resp, n_in, lat = gen_text(model, tok, renderer(task))
            v = grade(resp, truth)
            report["cells"][key] = {
                "task": ti, "family": task["family"], "cond": cond,
                "truth": truth, "response": resp, "verdict": v,
                "input_tokens": n_in, "latency_s": round(lat, 2)}
            OUT.write_text(json.dumps(report, indent=1))
            print(f"[{key}] {v} exp={truth} resp={resp.strip()[:10]!r} "
                  f"tok={n_in}", flush=True)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
