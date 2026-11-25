"""HZN-OC-01 comprehension pretest — operational GPU run (HZN-IR-08 d6).

Runs the C1-C4 gate against the real model on campaign-realistic
synthetic catalogs (248 rows, near-zero contrast mass, rounded ties,
repeated family/layer ambiguity). C4 probes carry the FULL rendered
catalog in context — grounding is resolving an address against the
catalog, not against a row in isolation. Preregistered counts:
MIN_PROBES C1:40 C2:40 C3:20 C4:60 across >=3 catalogs; thresholds
95/90/90/100; strict whole-response grading; fail-closed gate.

Encodings under test: linguistic (baseline) + s1 + s3 (s2 is dominated
on tokens — no efficiency justification). Per-catalog probe budget is
set so the 3-catalog aggregate meets MIN_PROBES with margin.

RESUMABLE: per-(level,catalog) results are written incrementally to
gate1_3_comprehension.json after every probe, and completed cells are
skipped on restart — a killed run loses at most one probe.

Output: results/gate1_3_comprehension.json + printed summary.
"""
import json, sys, time
sys.path.insert(0, r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path

AI_RESEARCH = Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research")
OUT = AI_RESEARCH / "results" / "gate1_3_comprehension.json"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jarvis_research.gate1_3_channel import encode_catalog_text
from jarvis_research.gate1_3_comprehension import (
    MIN_PROBES, PRETEST_ORDER, c4_coverage, gen_items, pretest_passed,
    prompt_for, score_items, synthetic_catalog)
from jarvis_research.gate1_pilot import MODEL_ID, DTYPE, local_provenance

LEVELS = ["linguistic", "s1", "s3"]
CATALOG_SEEDS = [101, 202, 303]
# C4: 21 unambiguous probes/catalog -> 63 total >= preregistered 60,
# plus ~7 ambiguity-refusal probes per catalog on top.
COUNTS = {"C1": 16, "C2": 16, "C3": 8, "C4": 21}
MAX_NEW = {"C1": 48, "C2": 16, "C3": 64, "C4": 24}

SYSTEM = ("You are reading a compressed observation catalog from a "
          "self-editing experiment. Answer each probe precisely. Reply "
          "with ONLY the requested format — no reasoning, no extra "
          "text, no repetition of the catalog row.")


def gen_text(model, tok, user_text, max_new):
    """Prompt exactly as the campaign proposer does: chat template,
    generation prompt, thinking disabled, EFFICIENT_ATTENTION backend,
    greedy decode."""
    enc = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user_text}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
        enable_thinking=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    sdpa = torch.nn.attention.sdpa_kernel(
        [torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
         torch.nn.attention.SDPBackend.MATH])
    with sdpa, torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new,
                             do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def save(report):
    OUT.write_text(json.dumps(report, indent=1))


def main():
    report = {"model": MODEL_ID, "counts_per_catalog": COUNTS,
              "cells": {}, "resumed_from": None}
    if OUT.exists():
        try:
            prior = json.loads(OUT.read_text())
            report["cells"] = prior.get("cells", {})
            report["resumed_from"] = "gate1_3_comprehension.json"
            print(f"[resume] {len(report['cells'])} completed cells")
        except Exception:
            pass

    provenance = local_provenance(MODEL_ID)
    report["revision"] = provenance["revision"]
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=provenance["revision"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=provenance["revision"], dtype=DTYPE,
        device_map={"": "cuda"}, local_files_only=True)
    model.eval()

    for level in LEVELS:
        for cseed in CATALOG_SEEDS:
            key = f"{level}:{cseed}"
            if key in report["cells"]:
                print(f"[skip] {key} already done", flush=True)
                continue
            t0 = time.perf_counter()
            cat = synthetic_catalog(248, seed=cseed)
            fam_code = {f: f"f{i}" for i, f in
                        enumerate(sorted({r["family"] for r in cat}))}
            catalog_text = encode_catalog_text(
                cat, fam_code, True, True, level,
                arm_mag=(-0.005, 0.005), arm_count=(1, 512))
            items = gen_items(cat, fam_code, True, level,
                              seed=cseed, counts=COUNTS)
            responses = {lvl: [] for lvl in PRETEST_ORDER}
            for lvl in PRETEST_ORDER:
                for j, (prompt, exp) in enumerate(items[lvl]):
                    q = prompt_for(lvl, prompt)
                    if lvl == "C4":
                        # Grounding requires the rendered catalog in
                        # context — the question addresses it.
                        q = catalog_text + "\n\n" + q
                    responses[lvl].append(
                        gen_text(model, tok, q, MAX_NEW[lvl]))
                    print(f"  {key} {lvl}[{j + 1}/{len(items[lvl])}] "
                          f"{time.perf_counter() - t0:.0f}s", flush=True)
            sc = score_items(items, responses)
            report["cells"][key] = {
                "scores": sc, "items": items,
                "responses": responses,
                "seconds": round(time.perf_counter() - t0, 1)}
            save(report)
            print(f"[{key}] " + " ".join(
                f"{l}={sc[l]['correct']}/{sc[l]['n']}"
                for l in PRETEST_ORDER), flush=True)

    # Aggregate per level across catalogs; apply the fail-closed gate.
    summary = {}
    for level in LEVELS:
        items_agg = {lvl: [] for lvl in PRETEST_ORDER}
        resp_agg = {lvl: [] for lvl in PRETEST_ORDER}
        n_cat = 0
        for cseed in CATALOG_SEEDS:
            cell = report["cells"].get(f"{level}:{cseed}")
            if not cell:
                continue
            n_cat += 1
            for lvl in PRETEST_ORDER:
                items_agg[lvl] += cell["items"][lvl]
                resp_agg[lvl] += cell["responses"][lvl]
        agg = score_items(items_agg, resp_agg)
        summary[level] = {
            "n_catalogs": n_cat,
            "aggregate_scores": agg,
            "pretest_passed": pretest_passed(agg) and n_cat >= 3,
        }
        print(f"[{level}] AGG " + " ".join(
            f"{l}:{agg[l]['correct']}/{agg[l]['n']}({agg[l]['acc']:.0%})"
            for l in PRETEST_ORDER) +
            f" -> {'PASS' if summary[level]['pretest_passed'] else 'FAIL'}",
            flush=True)
    report["summary"] = summary
    save(report)
    print("wrote results/gate1_3_comprehension.json")


if __name__ == "__main__":
    sys.exit(main())
