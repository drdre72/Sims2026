"""HZN-IR-11 D5 — candidate-conditioned reasoning sentinel (capped).

12 tasks x 5 model conditions = 60 generations (T1 deterministic).
Budget cap enforced: 120 generations / 20 min, whichever first.

T0  full 248-row catalog, candidates embedded as rows, answer by id
T1  deterministic magnitude ranking (no model generation)
T2  model over retrieved labeled candidates (letters)
T3  model over random size-matched candidate subset
T4  model over T2 candidates with obs correspondence shuffled
T5  known-answer subset (diagnostic ceiling)
"""
import hashlib, json, random, sys, time
sys.path.insert(0, r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path

AI_RESEARCH = Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research")
OUT = AI_RESEARCH / "results" / "gate1_4_candidate_reasoning.json"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jarvis_research.gate1_4_benchmark import (
    FAMILIES, RULES, grade_task, magnitude_answer, make_task, render_task)
from jarvis_research.gate1_3_comprehension import synthetic_catalog
from jarvis_research.gate1_pilot import MODEL_ID, DTYPE, local_provenance
from jarvis_research.run_manifest import validate_resume_manifest

N_TASKS = 12
MAX_GENERATIONS = 120
MAX_WALL_SECONDS = 20 * 60
HARNESS_VERSION = "candidate-reasoning-1"
SYSTEM = ("You are evaluating intervention candidates in a self-editing "
          "experiment. Apply the stated rule exactly. Reply with ONLY "
          "the requested format — no reasoning, no extra text.")


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


def build_tasks():
    tasks = []
    for i in range(N_TASKS):
        fam = FAMILIES[i % len(FAMILIES)]
        rng = random.Random(int.from_bytes(hashlib.sha256(
            f"t5-task:{i}".encode()).digest()[:8], "little"))
        tasks.append(make_task(rng, fam))
    return tasks


def embed_in_catalog(task, cat_rows, rng):
    """Inject the task's candidates as rows of the synthetic catalog.
    Returns (modified_rows, id_to_label)."""
    rows = [dict(r) for r in cat_rows]
    ids = rng.sample(range(len(rows)), len(task["candidates"]))
    id_to_label = {}
    for c, idx in zip(task["candidates"], ids):
        rows[idx]["obs_act_rank"] = c["ar"]
        rows[idx]["obs_log_contrast"] = c["lc"]
        rows[idx]["obs_available"] = c["obs_available"]
        id_to_label[rows[idx]["param_id"]] = c["label"]
    return rows, id_to_label


def catalog_truth(rule_key, rows):
    """Apply the rule mechanically to catalog rows (T0 full-catalog
    truth). Returns winning param_id or 'none'."""
    cands = []
    for r in rows:
        cands.append({"label": str(r["param_id"]),
                      "ar": round(r["obs_act_rank"], 2),
                      "lc": round(r["obs_log_contrast"], 2),
                      "obs_available": r.get("obs_available", True)})
    t = grade_task(rule_key, cands)
    return t["answer"]


def render_catalog_rows(rows):
    lines = []
    for r in rows:
        if r.get("obs_available", True):
            lines.append(f"{r['param_id']}|L{r['layer']}"
                         f"|ar={round(r['obs_act_rank'],2):.2f}"
                         f"|lc={round(r['obs_log_contrast'],2):.2f}")
        else:
            lines.append(f"{r['param_id']}|L{r['layer']}|no-obs")
    return "\n".join(lines)


def grade_answer(resp, expected, valid):
    import re
    t = resp.strip()
    if expected == "none":
        if t.lower() == "none":
            return "correct"
        return "invalid_pick"
    if t.lower() == "none":
        return "abstain_when_answer"
    if str(expected) == t.upper().strip():
        return "correct"
    if t.upper() in valid or t in valid:
        return "valid_but_incorrect"
    return "invalid_reference"


def main():
    report = {"model": MODEL_ID, "harness": HARNESS_VERSION,
              "manifest": {}, "cells": {}}
    stored_manifest = {}
    if OUT.exists():
        report = json.loads(OUT.read_text())
        stored_manifest = dict(report.get("manifest", {}))
        print(f"[resume] {len(report['cells'])} cells", flush=True)

    provenance = local_provenance(MODEL_ID)
    report["manifest"] = {"model_revision": provenance["revision"],
                          "harness_version": HARNESS_VERSION,
                          "n_tasks": N_TASKS}
    if stored_manifest:
        errs = validate_resume_manifest(stored_manifest, report["manifest"])
        if errs:
            print("REFUSING resume - manifest mismatch:", flush=True)
            raise SystemExit(1)

    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=provenance["revision"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=provenance["revision"], dtype=DTYPE,
        device_map={"": "cuda"}, local_files_only=True)
    model.eval()

    cat = synthetic_catalog(248, seed=104)
    tasks = build_tasks()
    t0_wall = time.perf_counter()
    n_gen = 0

    def budget_ok():
        return n_gen < MAX_GENERATIONS and \
            (time.perf_counter() - t0_wall) < MAX_WALL_SECONDS

    for ti, task in enumerate(tasks):
        rng = random.Random(int.from_bytes(hashlib.sha256(
            f"t5-run:{ti}".encode()).digest()[:8], "little"))
        truth = task["truth"]["answer"]
        rule = RULES[task["rule_key"]]
        rule_text = rule["text"].format(ar_thr=rule["ar_thr"],
                                        lc_thr=rule["lc_thr"])
        mag = magnitude_answer(task["candidates"])
        key = f"task{ti}"
        cell = report["cells"].setdefault(key, {
            "family": task["family"], "rule_key": task["rule_key"],
            "truth": truth, "magnitude_answer": mag,
            "candidates": task["candidates"], "conditions": {}})
        conds = cell["conditions"]

        # T1 — deterministic baseline, no generation
        conds["T1"] = {"answer": mag, "correct": mag == truth}

        def run(cond, prompt, expected, valid, extra=None):
            nonlocal n_gen
            if cond in conds and "verdict" in conds[cond]:
                return
            if not budget_ok():
                conds[cond] = {"skipped": "budget_cap"}
                return
            resp, n_in, lat = gen_text(model, tok, prompt)
            n_gen += 1
            verdict = grade_answer(resp, expected, valid)
            rec = {"response": resp, "expected": expected,
                   "verdict": verdict, "input_tokens": n_in,
                   "latency_s": round(lat, 2)}
            if extra:
                rec.update(extra)
            conds[cond] = rec
            OUT.write_text(json.dumps(report, indent=1))
            print(f"[{key}:{cond}] {verdict} exp={expected} "
                  f"resp={resp.strip()[:16]!r} {lat:.1f}s", flush=True)

        letters = [c["label"] for c in task["candidates"]] + ["none"]

        # T2 — retrieved labeled candidates
        run("T2", render_task(task) , truth, letters)

        # T0 — full catalog, candidates embedded
        rows, id2label = embed_in_catalog(task, cat, rng)
        full_truth = catalog_truth(task["rule_key"], rows)
        q0 = (render_catalog_rows(rows) +
              f"\n\nRule: {rule_text}\n\nWhich row id satisfies the "
              f"rule? Reply with ONLY the row id number, or the word "
              f"none.")
        valid_ids = [str(r["param_id"]) for r in rows] + ["none"]
        run("T0", q0, full_truth, valid_ids,
            extra={"candidate_truth": truth,
                   "full_catalog_truth": full_truth})

        # T3 — random size-matched subset; correct candidate in w.p. .5
        t3_cands = []
        include_correct = rng.random() < 0.5
        drng = random.Random(int.from_bytes(hashlib.sha256(
            f"t3:{ti}".encode()).digest()[:8], "little"))
        pool = [c for c in task["candidates"]
                if c["label"] != truth]
        drng.shuffle(pool)
        t3_cands = [dict(c) for c in pool[:4]]
        if include_correct and truth != "none":
            corr = next(c for c in task["candidates"]
                        if c["label"] == truth)
            t3_cands.append(dict(corr))
        while len(t3_cands) < 5:
            t3_cands.append({"label": "X", "ar": round(drng.uniform(0.05, 0.9), 2),
                             "lc": round(drng.uniform(-1.5, 1.5), 2),
                             "obs_available": True})
        drng.shuffle(t3_cands)
        for i, c in enumerate(t3_cands):
            c["label"] = "ABCDE"[i]
        t3_truth = grade_task(task["rule_key"], t3_cands)["answer"]
        t3_task = {"rule_key": task["rule_key"], "candidates": t3_cands}
        run("T3", render_task(t3_task), t3_truth,
            [c["label"] for c in t3_cands] + ["none"],
            extra={"correct_present": include_correct})

        # T4 — T2 candidates with obs values shuffled across labels
        vals = [(c["ar"], c["lc"], c["obs_available"])
                for c in task["candidates"]]
        sh = vals[:]
        while True:
            rng.shuffle(sh)
            if sh != vals:
                break
        t4_cands = []
        for c, (a, l, o) in zip(task["candidates"], sh):
            t4_cands.append({"label": c["label"], "ar": a, "lc": l,
                             "obs_available": o})
        t4_truth = grade_task(task["rule_key"], t4_cands)["answer"]
        t4_task = {"rule_key": task["rule_key"], "candidates": t4_cands}
        run("T4", render_task(t4_task), t4_truth,
            letters, extra={"t2_truth": truth})

        # T5 — known-answer subset
        if truth != "none":
            corr = next(c for c in task["candidates"] if c["label"] == truth)
            others = [c for c in task["candidates"] if c["label"] != truth]
            t5_cands = [dict(corr), dict(rng.choice(others))]
        else:
            nonq = [c for c in task["candidates"]
                    if c["label"] not in task["truth"]["qualifying"]]
            t5_cands = [dict(c) for c in rng.sample(nonq, min(2, len(nonq)))]
        rng.shuffle(t5_cands)
        for i, c in enumerate(t5_cands):
            c["label"] = "AB"[i]
        t5_truth = grade_task(task["rule_key"], t5_cands)["answer"]
        t5_task = {"rule_key": task["rule_key"], "candidates": t5_cands}
        run("T5", render_task(t5_task), t5_truth,
            [c["label"] for c in t5_cands] + ["none"])

        OUT.write_text(json.dumps(report, indent=1))

    print(f"done: {n_gen} generations, "
          f"{time.perf_counter() - t0_wall:.0f}s wall", flush=True)


if __name__ == "__main__":
    sys.exit(main())
