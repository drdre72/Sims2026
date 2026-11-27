"""HZN-ORIENT-01 — orientation scaffold / salience-capture probe.

18 paired tasks x 3 prompt conditions = 54 planned generations.
Inference-only; no intervention authority or parameter modification.
"""
import hashlib, json, sys, time
sys.path.insert(0, r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path

AI_RESEARCH = Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research")
OUT = AI_RESEARCH / "results" / "gate1_4_orient_salience.json"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jarvis_research.gate1_4_mirror import answer_verdict, normalize_answer
from jarvis_research.gate1_4_orient import (
    CONDITIONS, N_TASKS, ORIENT_VERSION, build_orient_tasks,
    render_condition, summarize_orient_cells)
from jarvis_research.gate1_pilot import MODEL_ID, DTYPE, local_provenance
from jarvis_research.run_manifest import validate_resume_manifest

MAX_GENERATIONS = 60
MAX_WALL_SECONDS = 20 * 60
HARNESS_VERSION = "orient-salience-1"
SYSTEM = ("You are evaluating intervention candidates in a self-editing "
          "experiment. Apply the stated rule exactly. Reply with ONLY "
          "the requested format — no reasoning, no extra text.")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
    lat = time.perf_counter() - t0
    text = tok.decode(out[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True)
    return text, n_in, lat


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def augment_summary(report):
    summary = summarize_orient_cells(report["cells"].values(), CONDITIONS)
    for cond in CONDITIONS:
        rows = [c["conditions"][cond] for c in report["cells"].values()
                if cond in c["conditions"] and "answer_correct" in c["conditions"][cond]]
        if rows:
            summary[cond]["mean_input_tokens"] = mean(
                r["input_tokens"] for r in rows)
            summary[cond]["mean_latency_s"] = mean(
                r["latency_s"] for r in rows)
    return summary


def main():
    tasks = build_orient_tasks()
    report = {"model": MODEL_ID, "harness": HARNESS_VERSION,
              "manifest": {}, "cells": {}, "summary": {}}
    stored_manifest = {}
    if OUT.exists():
        report = json.loads(OUT.read_text(encoding="utf-8"))
        stored_manifest = dict(report.get("manifest", {}))
        print(f"[resume] {len(report['cells'])} cells", flush=True)

    provenance = local_provenance(MODEL_ID)
    report["manifest"] = {
        "model_revision": provenance["revision"],
        "harness_version": HARNESS_VERSION,
        "n_tasks": len(tasks),
        "orient_version": ORIENT_VERSION,
        "conditions": list(CONDITIONS),
    }
    if stored_manifest:
        errs = validate_resume_manifest(stored_manifest, report["manifest"])
        if errs:
            print("REFUSING resume - manifest mismatch:", flush=True)
            for e in errs:
                print("  - " + e, flush=True)
            raise SystemExit(1)

    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=provenance["revision"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=provenance["revision"], dtype=DTYPE,
        device_map={"": "cuda"}, local_files_only=True)
    model.eval()

    t0_wall = time.perf_counter()
    n_gen = 0

    def budget_ok():
        return n_gen < MAX_GENERATIONS and \
            (time.perf_counter() - t0_wall) < MAX_WALL_SECONDS

    for ti, task in enumerate(tasks):
        key = f"task{ti}"
        truth = task["truth"]["answer"]
        valid = [c["label"] for c in task["candidates"]] + ["none"]
        cell = report["cells"].setdefault(key, {
            "family": task["family"], "rule_key": task["rule_key"],
            "truth": truth, "magnitude_answer": task["magnitude_answer"],
            "candidates": task["candidates"], "conditions": {},
        })
        conds = cell["conditions"]

        for cond in CONDITIONS:
            if cond in conds and "answer_correct" in conds[cond]:
                continue
            if not budget_ok():
                conds[cond] = {"skipped": "budget_cap"}
                continue
            prompt = render_condition(task, cond)
            resp, n_in, lat = gen_text(model, tok, prompt)
            n_gen += 1
            ans = normalize_answer(resp)
            verdict = answer_verdict(ans, truth, valid)
            rec = {
                "response": resp, "answer": ans, "expected": truth,
                "answer_verdict": verdict,
                "answer_correct": verdict == "correct",
                "format_valid": ans in valid,
                "input_tokens": n_in, "latency_s": round(lat, 2),
                "prompt_sha256": sha256_text(prompt),
            }
            conds[cond] = rec
            OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
            print(f"[{key}:{cond}] {verdict} exp={truth} ans={ans!r} "
                  f"{lat:.1f}s", flush=True)

        OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    report["summary"] = augment_summary(report)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"done: {n_gen} generations, "
          f"{time.perf_counter() - t0_wall:.0f}s wall", flush=True)
    print(json.dumps(report["summary"], indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
