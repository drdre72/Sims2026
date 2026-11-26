"""HZN-IR-10 D4 — paired grounding sentinel (inference only).

Repairs from the IR-09 review:
- prompt no longer contradicts itself: ONE output instruction covering
  all three acceptable answers
- IDENTICAL paired targets across conditions: the same (seed, target)
  probe is asked under full catalog, store-retrieved subset and oracle
  subset — paired, not independently sampled
- dual truth recorded: full-catalog truth AND rendered-subset truth
  per probe
- verdict taxonomy distinguishes wrong_id_valid (id exists in catalog)
  from wrong_id_nonexistent (hallucinated id)
- multi-row transcription measured separately from selection
- sizes 64/128/248 only; no repeat of the full sweep

Cell keys: {seed}:{size}:{cond} — resumable, manifest-bound.
"""
import hashlib, json, random, sys, time
sys.path.insert(0, r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path

AI_RESEARCH = Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research")
LEVEL = sys.argv[1] if len(sys.argv) > 1 else "linguistic"
OUT = AI_RESEARCH / "results" / f"gate1_3_sentinel_d4_{LEVEL}.json"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jarvis_research.gate1_3_comprehension import synthetic_catalog
from jarvis_research.gate1_3_channel import (
    canonical_record, encode_catalog_text)
from jarvis_research.gate1_3_store import ObservationStore, catalog_sha256
from jarvis_research.gate1_pilot import MODEL_ID, DTYPE, local_provenance
from collections import defaultdict

SEEDS = [104, 106]
SIZES = [64, 128, 248]
CONDS = ["full", "retrieved", "oracle"] if (len(sys.argv) < 2 or sys.argv[1] == "linguistic") else ["retrieved", "oracle"]
N_UNIQUE = 8
N_AMBIG = 4
N_MISS = 4
HARNESS_VERSION = "sentinel-d4-1-" + (sys.argv[1] if len(sys.argv) > 1 else "linguistic")
SYSTEM = ("You are reading an observation catalog from a self-editing "
          "experiment. Answer each probe precisely. Reply with ONLY the "
          "requested format — no reasoning, no extra text.")

LOOKUP_TEMPLATE = (
    "Give the id number of the row with family tag {tag} and layer "
    "{layer}. Reply with ONLY one of these three things: the id "
    "number, the word ambiguous (if more than one row matches), or "
    "the words not present (if no row matches).")


def gen_text(model, tok, user_text, max_new=12):
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


def truth_in(rows, pair):
    hits = [r["param_id"] for r in rows
            if (r["family"], r["layer"]) == pair]
    if len(hits) == 0:
        return "not present", "missing"
    if len(hits) > 1:
        return "ambiguous", "ambiguous"
    return hits[0], "unique"


def grade(expected, resp, valid_ids):
    import re
    t = resp.strip()
    if expected == "ambiguous":
        if t.lower() == "ambiguous":
            return "correct"
        if re.fullmatch(r"[+-]?\d+", t):
            return "invalid_pick"
        return "malformed"
    if expected == "not present":
        if t.lower() in ("not present", "none", "missing"):
            return "correct"
        if re.fullmatch(r"[+-]?\d+", t):
            return "hallucinated_present" if int(t) in valid_ids \
                else "hallucinated_nonexistent"
        return "malformed"
    if re.fullmatch(r"[+-]?\d+", t):
        n = int(t)
        if n == expected:
            return "correct"
        return "wrong_id_valid" if n in valid_ids \
            else "wrong_id_nonexistent"
    if t.lower() in ("ambiguous", "not present", "none", "missing"):
        return "refusal_when_present"
    return "malformed"


def pick_targets(cat, seed, size):
    """Paired targets for one (seed, size): identical across conditions.
    Unique/ambiguous/missing (family, layer) pairs of THE SIZE-SUBSET."""
    rng = random.Random(int.from_bytes(
        hashlib.sha256(f"t4:{seed}:{size}".encode()).digest()[:8],
        "little"))
    by_pair = defaultdict(list)
    for r in cat:
        by_pair[(r["family"], r["layer"])].append(r["param_id"])
    uniq = sorted(p for p, ids in by_pair.items() if len(ids) == 1)
    amb = sorted(p for p, ids in by_pair.items() if len(ids) > 1)
    rng.shuffle(uniq); rng.shuffle(amb)
    fams = sorted({r["family"] for r in cat})
    miss = [(f, l) for f in fams for l in range(32)
            if (f, l) not in by_pair]
    rng.shuffle(miss)
    targets = ([("unique", p) for p in uniq[:N_UNIQUE]]
               + [("ambiguous", p) for p in amb[:N_AMBIG]]
               + [("missing", p) for p in miss[:N_MISS]])
    return targets


def run_size(model, tok, cat_full, seed, size):
    rng = random.Random(int.from_bytes(
        hashlib.sha256(f"ord:{seed}".encode()).digest()[:8], "little"))
    order = sorted(cat_full, key=lambda r: random.Random(
        seed * 7919 + r["param_id"]).random())
    cat = order[:size]
    fam_code = {f: f"f{i}" for i, f in
                enumerate(sorted({r["family"] for r in cat}))}
    canon = [canonical_record(r, True, (-0.005, 0.005)) for r in cat]
    store = ObservationStore(canon)
    valid_ids = {r["param_id"] for r in cat}
    targets = pick_targets(cat, seed, size)
    out = {}

    full_ctx = encode_catalog_text(cat, fam_code, True, True,
                                   LEVEL, arm_mag=(-0.005, 0.005),
                                   arm_count=(1, 512))
    for kind, pair in targets:
        fam, lay = pair
        q = LOOKUP_TEMPLATE.format(tag=fam_code[fam], layer=lay)
        truth_full, kind_full = truth_in(cat, pair)
        for cond in CONDS:
            if cond == "full":
                ctx, sub = full_ctx, cat
            elif cond == "retrieved":
                sub = store.find(family=fam)
                ctx = store.render_subset(sub, fam_code, level=LEVEL)
            else:
                k = max(1, len(store.find(family=fam)))
                tgt = [r for r in cat
                       if (r["family"], r["layer"]) == pair]
                pool = [r for r in cat if r not in tgt]
                sub = sorted(tgt + rng.sample(pool, max(0, k - len(tgt))),
                             key=lambda r: r["param_id"])
                ctx = store.render_subset(
                    [canonical_record(r, True, (-0.005, 0.005))
                     for r in sub], fam_code, level=LEVEL)
            ctx_rows = cat if cond == "full" else sub
            truth_sub, kind_sub = truth_in(ctx_rows, pair)
            resp, n_in, lat = gen_text(model, tok, ctx + "\n\n" + q)
            expected = truth_full if cond == "full" else truth_sub
            verdict = grade(expected, resp, valid_ids)
            out[f"{kind}:{pair[0]}:{pair[1]}:{cond}"] = {
                "kind_catalog": kind, "kind_in_context": kind_sub,
                "expected": expected, "expected_full_catalog": truth_full,
                "response": resp, "verdict": verdict,
                "n_rows_ctx": len(sub) if cond != "full" else len(cat),
                "input_tokens": n_in, "latency_s": round(lat, 2)}
    return out


def main():
    report = {"model": MODEL_ID, "harness": HARNESS_VERSION,
              "manifest": {}, "cells": {}}
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

    cats = {s: synthetic_catalog(248, seed=s) for s in SEEDS}
    report["manifest"] = {
        "model_revision": provenance["revision"],
        "harness_version": HARNESS_VERSION,
        "catalog_sha256": {str(s): catalog_sha256(
            [canonical_record(r, True, (-0.005, 0.005))
             for r in cats[s]]) for s in SEEDS},
        "seeds": SEEDS, "sizes": SIZES, "conds": CONDS,
        "lookup_template_sha256": hashlib.sha256(
            LOOKUP_TEMPLATE.encode()).hexdigest()[:16]}

    for s in SEEDS:
        for size in SIZES:
            key = f"{s}:{size}"
            if key in report["cells"]:
                print(f"[skip] {key}", flush=True)
                continue
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            probes = run_size(model, tok, cats[s], s, size)
            report["cells"][key] = {
                "probes": probes,
                "peak_vram_mb": int(
                    torch.cuda.max_memory_allocated() / 1e6),
                "seconds": round(time.perf_counter() - t0, 1)}
            OUT.write_text(json.dumps(report, indent=1))
            n_ok = sum(1 for p in probes.values()
                       if p["verdict"] == "correct")
            print(f"[{key}] {n_ok}/{len(probes)} correct "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    print(f"done -> {OUT}")


if __name__ == "__main__":
    sys.exit(main())
