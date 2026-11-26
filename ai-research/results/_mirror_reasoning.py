"""HZN-IR-12 — candidate-reasoning mirror experiment (capped).

Uses the committed IR-11 tasks and graded T2 responses as the mirror
surface. 12 tasks x 4 conditions = 48 planned generations:

M0  independent retry under candidate-order permutation + relabeling
M1  task + previous response
M2  task + previous response + correctness verdict
M3  task + previous response + correctness verdict + expanded evidence

Inference-only. No intervention authority, weight modification, evaluator
modification, or feedback into any training loop.
"""
import hashlib, json, sys, time
sys.path.insert(0, r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path

AI_RESEARCH = Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research")
SOURCE = AI_RESEARCH / "research" / "evidence" / "gate1_4_candidate_reasoning.json"
OUT = AI_RESEARCH / "results" / "gate1_4_mirror.json"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jarvis_research.gate1_4_benchmark import render_task
from jarvis_research.gate1_4_mirror import (
    answer_verdict, grade_mirror_response, normalize_answer,
    permuted_relabel_task, reference_failure_cause, render_mirror_prompt)
from jarvis_research.gate1_pilot import MODEL_ID, DTYPE, local_provenance
from jarvis_research.run_manifest import validate_resume_manifest

MAX_GENERATIONS = 60
MAX_WALL_SECONDS = 20 * 60
HARNESS_VERSION = "mirror-reasoning-1"
CONDITIONS = ("M0", "M1", "M2", "M3")
SYSTEM = ("You are evaluating intervention candidates in a self-editing "
          "experiment. Apply the stated rule exactly. Reply with ONLY "
          "the requested format — no reasoning, no extra text.")


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def gen_text(model, tok, user_text, max_new=48):
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


def task_key_order(key):
    return int(key[4:])


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def summarize(report):
    """Compute the preregistered behavioral metrics from recorded cells."""
    cells = [report["cells"][k] for k in
             sorted(report["cells"], key=task_key_order)]
    summary = {}
    for cond in CONDITIONS:
        rows = []
        for c in cells:
            r = c["conditions"].get(cond)
            if r and "answer_correct" in r:
                rows.append((c, r))
        inc = [(c, r) for c, r in rows if not c["initially_correct"]]
        cor = [(c, r) for c, r in rows if c["initially_correct"]]
        fmt = [bool(r.get("format_valid")) for c, r in rows]
        acc = [bool(r["answer_correct"]) for c, r in rows]
        correction = [bool(r["answer_correct"]) for c, r in inc]
        stability = [bool(r["answer_correct"]) for c, r in cor]
        explicit_flip = [bool(r.get("format_valid")) and
                         not bool(r["answer_correct"]) for c, r in cor]
        invalid_cor = [not bool(r.get("format_valid")) for c, r in cor]
        cause_inc = [bool(r.get("cause_correct")) for c, r in inc]
        cause_all = [bool(r.get("cause_correct")) for c, r in rows
                     if "cause_correct" in r]
        s = {
            "n": len(rows),
            "n_initially_correct": len(cor),
            "n_initially_incorrect": len(inc),
            "format_valid_rate": mean(fmt),
            "accuracy": mean(acc),
            "correction_rate_incorrect": mean(correction),
            "stability_rate_correct": mean(stability),
            "explicit_flip_rate_correct": mean(explicit_flip),
            "invalid_rate_correct": mean(invalid_cor),
        }
        corr_rate = s["correction_rate_incorrect"]
        flip_rate = s["explicit_flip_rate_correct"]
        if corr_rate is not None and flip_rate is not None:
            s["correction_minus_flip"] = corr_rate - flip_rate
        if cond != "M0":
            s["cause_accuracy_incorrect"] = mean(cause_inc)
            s["cause_accuracy_all"] = mean(cause_all)
        summary[cond] = s
    mirror_rows = []
    for c in cells:
        for cond in ("M1", "M2", "M3"):
            r = c["conditions"].get(cond)
            if r and "answer_correct" in r:
                mirror_rows.append(r)
    summary["M1_M3_aggregate"] = {
        "n": len(mirror_rows),
        "format_valid_rate": mean([bool(r.get("format_valid"))
                                   for r in mirror_rows]),
    }
    return summary


def main():
    source_bytes_hash = file_sha256(SOURCE)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source_keys = sorted(source["cells"], key=task_key_order)

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
        "n_tasks": len(source_keys),
        "conditions": list(CONDITIONS),
        "source_evidence_sha256": source_bytes_hash,
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

    for key in source_keys:
        src = source["cells"][key]
        t2 = src["conditions"]["T2"]
        prev = normalize_answer(t2["response"])
        truth = src["truth"]
        initially_correct = t2["verdict"] == "correct"
        verdict_word = "correct" if initially_correct else "incorrect"
        task = {"family": src["family"], "rule_key": src["rule_key"],
                "candidates": src["candidates"]}
        ref_cause = reference_failure_cause(src["rule_key"],
                                            src["candidates"], prev)
        valid = [c["label"] for c in src["candidates"]] + ["none"]

        cell = report["cells"].setdefault(key, {
            "family": src["family"], "rule_key": src["rule_key"],
            "truth": truth, "candidates": src["candidates"],
            "previous_answer": prev,
            "previous_verdict": t2["verdict"],
            "initially_correct": initially_correct,
            "reference_cause": ref_cause,
            "conditions": {},
        })
        conds = cell["conditions"]

        def write():
            OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

        def run_m0():
            nonlocal n_gen
            if "M0" in conds and "answer_correct" in conds["M0"]:
                return
            if not budget_ok():
                conds["M0"] = {"skipped": "budget_cap"}
                return
            seed = int.from_bytes(hashlib.sha256(
                f"m0:{key}".encode()).digest()[:8], "little")
            m0task = permuted_relabel_task(task, seed)
            prompt = render_task(m0task)
            resp, n_in, lat = gen_text(model, tok, prompt, max_new=16)
            n_gen += 1
            ans = normalize_answer(resp)
            expected = m0task["truth"]["answer"]
            av = answer_verdict(ans, expected, valid)
            mapped_prev = (m0task["label_map"].get(prev, prev)
                           if prev != "none" else "none")
            rec = {
                "response": resp, "answer": ans, "expected": expected,
                "answer_verdict": av,
                "answer_correct": av == "correct",
                "format_valid": ans in valid,
                "input_tokens": n_in, "latency_s": round(lat, 2),
                "prompt_sha256": sha256_text(prompt),
                "label_map": m0task["label_map"],
                "mapped_previous_answer": mapped_prev,
                "changed_from_previous": ans != mapped_prev,
                "m0_truth": expected,
            }
            conds["M0"] = rec
            write()
            print(f"[{key}:M0] {av} exp={expected} resp={ans!r} "
                  f"{lat:.1f}s", flush=True)

        def run_mirror(cond, verdict, expanded):
            nonlocal n_gen
            if cond in conds and "answer_correct" in conds[cond]:
                return
            if not budget_ok():
                conds[cond] = {"skipped": "budget_cap"}
                return
            prompt = render_mirror_prompt(
                task, prev, verdict=verdict, expanded_evidence=expanded)
            resp, n_in, lat = gen_text(model, tok, prompt, max_new=48)
            n_gen += 1
            g = grade_mirror_response(resp, truth, prev, valid, ref_cause)
            rec = {
                "response": resp,
                "expected": truth,
                "input_tokens": n_in,
                "latency_s": round(lat, 2),
                "prompt_sha256": sha256_text(prompt),
                "verdict_shown": verdict,
                "expanded_evidence": expanded,
                "format_valid": g["parse_status"] == "valid",
            }
            rec.update(g)
            conds[cond] = rec
            write()
            print(f"[{key}:{cond}] {g['answer_verdict']} "
                  f"cause={g.get('cause')!r} exp={truth} "
                  f"ans={g.get('answer')!r} {lat:.1f}s", flush=True)

        run_m0()
        run_mirror("M1", None, False)
        run_mirror("M2", verdict_word, False)
        run_mirror("M3", verdict_word, True)
        write()

    report["summary"] = summarize(report)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"done: {n_gen} generations, "
          f"{time.perf_counter() - t0_wall:.0f}s wall", flush=True)
    print(json.dumps(report["summary"], indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
