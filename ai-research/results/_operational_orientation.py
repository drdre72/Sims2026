"""HZN-OOI-01 — operational orientation and exact-tie inhibition probe.

24 deterministic scenarios expanded to 36 paired presentations x 2 generative
conditions = 72 planned generations. STRUCT additionally yields a deterministic
delegation condition computed from the model's extracted raw evidence.
Inference-only; no intervention authority, parameter modification, or training.
"""
import hashlib, json, sys, time
sys.path.insert(0, r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path

AI_RESEARCH = Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research")
OUT = AI_RESEARCH / "results" / "gate1_4_operational_orientation.json"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jarvis_research.gate1_4_mirror import answer_verdict, normalize_answer
from jarvis_research.gate1_4_operational import (
    CONDITIONS, OOI_VERSION, apply_exact_tie_guard, build_operational_tasks,
    grade_structured_artifact, render_operational_condition,
    summarize_operational_cells)
from jarvis_research.gate1_pilot import MODEL_ID, DTYPE, local_provenance
from jarvis_research.run_manifest import validate_resume_manifest

MAX_GENERATIONS = 80
MAX_WALL_SECONDS = 20 * 60
STRUCT_MAX_NEW = 192
HARNESS_VERSION = "operational-orientation-1"
SYSTEM = ("You are evaluating intervention candidates in a self-editing "
          "experiment. Apply the stated rule exactly. Follow the requested "
          "output format exactly — no reasoning, no extra text.")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj):
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


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
    gen_ids = out[0][enc["input_ids"].shape[1]:]
    text = tok.decode(gen_ids, skip_special_tokens=True)
    return text, n_in, int(gen_ids.shape[0]), lat


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def augment_summary(report):
    summary = summarize_operational_cells(report["cells"].values(), CONDITIONS)
    for cond in CONDITIONS:
        rows = [c["conditions"][cond] for c in report["cells"].values()
                if cond in c["conditions"] and "answer_correct" in c["conditions"][cond]]
        if rows:
            summary[cond]["mean_input_tokens"] = mean(
                r["input_tokens"] for r in rows)
            summary[cond]["mean_output_tokens"] = mean(
                r["output_tokens"] for r in rows)
            summary[cond]["mean_latency_s"] = mean(
                r["latency_s"] for r in rows)
    return summary


def main():
    tasks = build_operational_tasks()
    report = {"model": MODEL_ID, "harness": HARNESS_VERSION,
              "manifest": {}, "cells": {}, "summary": {}}
    stored_manifest = {}
    if OUT.exists():
        report = json.loads(OUT.read_text(encoding="utf-8"))
        stored_manifest = dict(report.get("manifest", {}))
        print(f"[resume] {len(report['cells'])} cells", flush=True)

    provenance = local_provenance(MODEL_ID)
    battery_fp = sha256_obj([
        {k: t[k] for k in ("task_id", "family", "rule_key", "presentation",
                           "candidates", "truth", "critical_items")}
        for t in tasks
    ])
    report["manifest"] = {
        "model_revision": provenance["revision"],
        "harness_version": HARNESS_VERSION,
        "n_presentations": len(tasks),
        "n_scenarios": len({t["scenario_index"] for t in tasks}),
        "ooi_version": OOI_VERSION,
        "conditions": list(CONDITIONS),
        "battery_sha256": battery_fp,
        "struct_max_new": STRUCT_MAX_NEW,
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

    for task in tasks:
        key = task["task_id"]
        truth = task["truth"]["answer"]
        valid = [c["label"] for c in task["candidates"]] + ["none"]
        cell = report["cells"].setdefault(key, {
            "family": task["family"], "rule_key": task["rule_key"],
            "scenario_index": task["scenario_index"],
            "presentation": task["presentation"],
            "truth": truth, "mechanics": task["mechanics"],
            "magnitude_answer": task["magnitude_answer"],
            "candidates": task["candidates"],
            "critical_items": task["critical_items"],
            "critical_labels": task["critical_labels"],
            "critical_positions": task["critical_positions"],
            "first_critical_item": task["first_critical_item"],
            "first_critical_label": task["first_critical_label"],
            "item_to_label": task["item_to_label"],
            "label_to_item": task["label_to_item"],
            "conditions": {},
        })
        conds = cell["conditions"]

        for cond in CONDITIONS:
            if cond in conds and "answer_correct" in conds[cond]:
                continue
            if not budget_ok():
                conds[cond] = {"skipped": "budget_cap"}
                continue
            prompt = render_operational_condition(task, cond)
            max_new = STRUCT_MAX_NEW if cond == "STRUCT" else 16
            resp, n_in, n_out, lat = gen_text(model, tok, prompt,
                                              max_new=max_new)
            n_gen += 1
            if cond == "BASE":
                ans = normalize_answer(resp)
                verdict = answer_verdict(ans, truth, valid)
                guarded = apply_exact_tie_guard(
                    ans, task["rule_key"], task["candidates"])
                rec = {
                    "response": resp, "answer": ans, "expected": truth,
                    "answer_verdict": verdict,
                    "answer_correct": verdict == "correct",
                    "format_valid": ans in valid,
                    "tie_guard_answer": guarded,
                    "tie_guard_correct": guarded == truth,
                    "input_tokens": n_in, "output_tokens": n_out,
                    "latency_s": round(lat, 2),
                    "prompt_sha256": sha256_text(prompt),
                }
            else:
                rec = grade_structured_artifact(resp, task)
                rec.update({
                    "response": resp,
                    "input_tokens": n_in, "output_tokens": n_out,
                    "latency_s": round(lat, 2),
                    "prompt_sha256": sha256_text(prompt),
                })
            conds[cond] = rec
            OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
            print(f"[{key}:{cond}] {rec['answer_verdict']} exp={truth} "
                  f"ans={rec.get('answer')!r} del={rec.get('delegated_answer')!r} "
                  f"{lat:.1f}s", flush=True)

        OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    report["summary"] = augment_summary(report)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"done: {n_gen} generations, "
          f"{time.perf_counter() - t0_wall:.0f}s wall", flush=True)
    print(json.dumps(report["summary"], indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
