"""HZN-G1.3-R — grounding isolation sweep (HZN-IR-09 §4, inference only).

Question: do grounding failures come from context SCALE, semantic
interpretation, or identifier binding? No weight edits anywhere.

Nested catalogs at 16/32/64/128/248 rows (each a seeded subset of the
next), linguistic encoding only, 2 seeds. Conditions:
  R0  full catalog, descriptive lookup            (248 = baseline replication)
  R1  reduced catalog, descriptive lookup         (the sweep below 248)
  R2  select-a-row-and-copy-id                    (campaign-realistic copy)
  R3  deterministic address resolution            (model describes, store resolves)
  R4  store-retrieved candidate subset            (family filter -> subset)
  R5  random subset of equal size                 (reduced-context control)
  R6  oracle subset containing the target         (diagnostic upper bound)

Per probe recorded: condition, size, target kind (unique/ambiguous/
missing), correct, expected, response, input tokens, latency, peak VRAM.
Resumable: cells saved incrementally to gate1_3_grounding.json.
"""
import json, random, sys, time
sys.path.insert(0, r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path

AI_RESEARCH = Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research")
OUT = AI_RESEARCH / "results" / "gate1_3_grounding.json"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jarvis_research.gate1_3_channel import encode_catalog_text
from jarvis_research.gate1_3_comprehension import synthetic_catalog
from jarvis_research.gate1_3_store import ObservationStore
from jarvis_research.gate1_pilot import MODEL_ID, DTYPE, local_provenance
from collections import defaultdict

LEVEL = "linguistic"
SIZES = [16, 32, 64, 128, 248]
SEEDS = [101, 202]
N_LOOKUP = 10       # unique-target lookups per arm per size
N_AMBIG = 4         # ambiguous-target lookups where available
N_MISS = 4          # missing-target lookups
N_COPY = 8          # R2 copy probes
N_DESCRIBE = 8      # R3 descriptor probes
SYSTEM = ("You are reading a compressed observation catalog from a "
          "self-editing experiment. Answer each probe precisely. Reply "
          "with ONLY the requested format — no reasoning, no extra "
          "text, no repetition of the catalog row.")


def gen_text(model, tok, user_text, max_new):
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


def subset_ids(rows, n, rng):
    return sorted(r["param_id"] for r in rng.sample(rows, min(n, len(rows))))


def lookup_items(cat_subset, fam_code, rng, n_u, n_a, n_m):
    """Unique/ambiguous/missing (family,layer) lookup probes against a
    subset catalog."""
    by_pair = defaultdict(list)
    for r in cat_subset:
        by_pair[(r["family"], r["layer"])].append(r["param_id"])
    uniq = sorted(p for p, ids in by_pair.items() if len(ids) == 1)
    amb = sorted(p for p, ids in by_pair.items() if len(ids) > 1)
    rng.shuffle(uniq); rng.shuffle(amb)
    items = []
    for fam, lay in uniq[:n_u]:
        items.append({"kind": "unique", "pair": (fam, lay),
                      "expected": by_pair[(fam, lay)][0]})
    for fam, lay in amb[:n_a]:
        items.append({"kind": "ambiguous", "pair": (fam, lay),
                      "expected": "ambiguous"})
    # Missing: (fam,layer) pairs absent from the subset.
    fams = sorted({r["family"] for r in cat_subset})
    miss = [(f, l) for f in fams for l in range(0, 32)
            if (f, l) not in by_pair]
    rng.shuffle(miss)
    for fam, lay in miss[:n_m]:
        items.append({"kind": "missing", "pair": (fam, lay),
                      "expected": "not present"})
    return items


def grade_lookup(expected, resp):
    t = resp.strip()
    if expected == "ambiguous":
        return t.lower() == "ambiguous"
    if expected == "not present":
        return t.lower() in ("not present", "none", "missing")
    import re
    return bool(re.fullmatch(r"[+-]?\d+", t)) and int(t) == expected


def classify(expected, resp):
    t = resp.strip()
    import re
    if expected == "ambiguous":
        return ("refused" if t.lower() == "ambiguous"
                else "invalid_pick")
    if expected == "not present":
        return ("correct_missing" if t.lower() in
                ("not present", "none", "missing") else "hallucinated")
    if re.fullmatch(r"[+-]?\d+", t):
        return "correct" if int(t) == expected else "wrong_id_valid"
    if t.lower() in ("ambiguous", "not present", "none", "missing"):
        return "refusal_when_present"
    return "malformed"


def run_cell(model, tok, cell, cat_full, level="linguistic"):
    """cell = {seed, size, condition}. Returns probes + summary."""
    seed, size, cond = cell["seed"], cell["size"], cell["cond"]
    import hashlib
    rng = random.Random(int.from_bytes(
        hashlib.sha256(f"{seed}:{size}:{cond}".encode()).digest()[:8],
        "little"))
    # Nested: deterministic subset ordering of the full catalog.
    order = sorted(cat_full, key=lambda r: rng.Random(seed * 7919
                                                    + r["param_id"]).random())
    cat = order[:size]
    fam_code = {f: f"f{i}" for i, f in
                enumerate(sorted({r["family"] for r in cat}))}
    from jarvis_research.gate1_3_channel import canonical_record
    store = ObservationStore(
        [canonical_record(r, True, (-0.005, 0.005)) for r in cat])
    probes = []

    def ask(user_text, max_new=16):
        return gen_text(model, tok, user_text, max_new)

    if cond in ("R0R1", "R4", "R5", "R6"):
        items = lookup_items(cat, fam_code, rng, N_LOOKUP, N_AMBIG, N_MISS)
        for it in items:
            fam, lay = it["pair"]
            q = (f"Give the id number of the row with family tag "
                 f"{fam_code[fam]} and layer {lay}. If more than one "
                 f"row matches, reply with ONLY the word ambiguous. If "
                 f"no row matches, reply with ONLY the words not "
                 f"present. Reply with ONLY the number:")
            if cond == "R0R1":
                sub = cat
                ctx = encode_catalog_text(cat, fam_code, True, True,
                                          level, arm_mag=(-0.005, 0.005),
                                          arm_count=(1, 512))
            elif cond == "R4":
                sub = store.find(family=fam)
            elif cond == "R5":
                k = max(1, len(store.find(family=fam)))
                sub = rng.sample(cat, min(k, len(cat)))
            else:  # R6 oracle: target row(s) + distractors of same size
                k = max(1, len(store.find(family=fam)))
                target = [r for r in cat
                          if (r["family"], r["layer"]) == it["pair"]]
                pool = [r for r in cat if r not in target]
                sub = target + rng.sample(pool, max(0, k - len(target)))
                sub.sort(key=lambda r: r["param_id"])
            if cond != "R0R1":
                ctx = store.render_subset(
                    [canonical_record(r, True, (-0.005, 0.005))
                     for r in sub], fam_code)
            # Expected answer is defined by the RENDERED context — a
            # random subset may drop the target or split an ambiguous
            # pair, changing the truthfully correct response.
            hits = [r["param_id"] for r in sub
                    if (r["family"], r["layer"]) == it["pair"]]
            if len(hits) == 0:
                exp, kind = "not present", "missing"
            elif len(hits) > 1:
                exp, kind = "ambiguous", "ambiguous"
            else:
                exp = hits[0]
                kind = it["kind"] if it["kind"] == "unique" else \
                    "partial_" + it["kind"]
            resp, n_in, lat = ask(ctx + "\n\n" + q)
            probes.append({"cond": cond, "size": size, "kind": kind,
                           "orig_kind": it["kind"], "expected": exp,
                           "response": resp,
                           "verdict": classify(exp, resp),
                           "subset_n": len(sub),
                           "input_tokens": n_in, "latency_s": round(lat, 2)})

    elif cond == "R2":
        rows = rng.sample(cat, min(N_COPY, len(cat)))
        for r in rows:
            row_txt = store.render_subset(
                [canonical_record(r, True, (-0.005, 0.005))], fam_code)
            q = (f"Copy the leading id number of this catalog row and "
                 f"reply with ONLY the number:\n{row_txt}")
            resp, n_in, lat = ask(q)
            exp = r["param_id"]
            probes.append({"cond": "R2", "size": size, "kind": "copy",
                           "expected": exp, "response": resp,
                           "verdict": classify(exp, resp),
                           "input_tokens": n_in,
                           "latency_s": round(lat, 2)})

    elif cond == "R3":
        # Model names (family,layer) of the max-|lc| row; STORE resolves.
        cands = sorted(cat, key=lambda r: -abs(r["obs_log_contrast"]))
        truth = cands[0]
        # only ask when the argmax is unique (clean boundary)
        if abs(cands[0]["obs_log_contrast"]) - abs(
                cands[1]["obs_log_contrast"]) > 1e-9:
            for _ in range(N_DESCRIBE):
                ctx = encode_catalog_text(cat, fam_code, True, True,
                                          level, arm_mag=(-0.005, 0.005),
                                          arm_count=(1, 512))
                q = (f"Which (family tag, layer) identifies the single "
                     f"row with the largest |lc|? Reply with ONLY the "
                     f"pair like 'f3 L12'.")
                resp, n_in, lat = ask(q, max_new=16)
                m = None
                import re as _re
                mm = _re.search(r"(f\d+)\s+L(\d+)", resp)
                verdict, got = "malformed", None
                if mm:
                    inv = {c: f for f, c in fam_code.items()}
                    fam = inv.get(mm.group(1))
                    res = (store.resolve(fam, int(mm.group(2)))
                           if fam else {"status": "malformed_tag"})
                    got = res.get("param_id")
                    if res["status"] == "unique" and got == truth["param_id"]:
                        verdict = "correct"
                    elif res["status"] == "ambiguous":
                        verdict = "ambiguous_desc"
                    elif res["status"] == "missing":
                        verdict = "missing_desc"
                    else:
                        verdict = "wrong_desc"
                probes.append({"cond": "R3", "size": size, "kind": "describe",
                               "expected": truth["param_id"],
                               "response": resp, "resolved_id": got,
                               "verdict": verdict,
                               "input_tokens": n_in,
                               "latency_s": round(lat, 2)})
    return probes


def main():
    report = {"model": MODEL_ID, "level": LEVEL, "cells": {}}
    if OUT.exists():
        report = json.loads(OUT.read_text())
        print(f"[resume] {len(report['cells'])} cells", flush=True)

    provenance = local_provenance(MODEL_ID)
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=provenance["revision"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=provenance["revision"], dtype=DTYPE,
        device_map={"": "cuda"}, local_files_only=True)
    model.eval()
    report["revision"] = provenance["revision"]

    cats = {s: synthetic_catalog(248, seed=s) for s in SEEDS}
    conds = ["R0R1", "R2", "R3", "R4", "R5", "R6"]
    for s in SEEDS:
        for size in SIZES:
            for cond in conds:
                key = f"{s}:{size}:{cond}"
                if key in report["cells"]:
                    print(f"[skip] {key}", flush=True)
                    continue
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                probes = run_cell(model, tok,
                                  {"seed": s, "size": size, "cond": cond},
                                  cats[s])
                report["cells"][key] = {
                    "probes": probes,
                    "peak_vram_mb": int(
                        torch.cuda.max_memory_allocated() / 1e6),
                    "seconds": round(time.perf_counter() - t0, 1)}
                OUT.write_text(json.dumps(report, indent=1))
                ok = sum(1 for p in probes if p["verdict"].startswith(
                    "correct") or p["verdict"] in ("refused",))
                print(f"[{key}] {ok}/{len(probes)} "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
    print("done -> results/gate1_3_grounding.json")


if __name__ == "__main__":
    sys.exit(main())
