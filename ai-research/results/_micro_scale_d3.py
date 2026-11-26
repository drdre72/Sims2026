"""HZN-IR-10 D3 — semantic arithmetic isolation at micro scale.

Direct-select showed semantic selection failing even at 32 rows. D3
asks where the floor is: 2, 4, 8 rows (expanding only where the
preceding scale succeeds), with controlled constructions:
  - negative extrema (the winner is negative — tests sign handling)
  - positive extrema
  - rounded ties (detected, excluded — never scored)
  - distinct prompt instances throughout (no repeated greedy probes)

Question types per directive: largest lc, smallest lc, largest |lc|,
largest alc (explicit column), largest ar, smallest ar.

Instances are PURPOSE-BUILT tiny catalogs: values are drawn so the
correct answer is unambiguous at rendered precision. The signed/abs
contrast is the core measurement: same magnitudes, winner sign flipped.

CPU-cheap prompts; GPU inference. Resumable; manifest-bound.
"""
import hashlib, json, random, sys, time
sys.path.insert(0, r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path

AI_RESEARCH = Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research")
OUT = AI_RESEARCH / "results" / "gate1_3_micro_d3.json"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jarvis_research.gate1_pilot import MODEL_ID, DTYPE, local_provenance
from jarvis_research.run_manifest import validate_resume_manifest

SIZES = [2, 4, 8]
N_INSTANCES = 12        # distinct instances per (size, construction)
HARNESS_VERSION = "micro-d3-1"
SYSTEM = ("You are reading an observation catalog from a self-editing "
          "experiment. Answer each probe precisely. Reply with ONLY "
          "the requested format — no reasoning, no extra text.")

QTYPES = [
    ("largest_lc", "largest lc"),
    ("smallest_lc", "smallest lc"),
    ("largest_abs_lc", "largest |lc|"),
    ("largest_alc_col", "largest alc"),
    ("largest_ar", "largest ar"),
    ("smallest_ar", "smallest ar"),
]
FAMS = ["attn.q_proj", "mlp.down_proj", "expert.0.w"]


def make_instance(rng, size, construction):
    """Purpose-built tiny catalog. construction:
      'neg_extreme': the |lc| winner is strongly negative
      'pos_extreme': the |lc| winner is positive
      'mixed':       winner sign randomized
    Returns rows + per-qtype answers (or None when tied)."""
    rows = []
    used_pairs = set()
    for i in range(size):
        fam = FAMS[i % len(FAMS)]
        layer = rng.randint(0, 31)
        while (fam, layer) in used_pairs:
            layer = rng.randint(0, 31)
        used_pairs.add((fam, layer))
        rows.append({"param_id": i, "layer": layer, "family": fam,
                     "numel": rng.choice([32768, 147456, 589824]),
                     "obs_available": True})
    # distinct magnitudes at rendered precision, no ties by
    # construction; losers get mixed signs in every construction so
    # sign handling is always exercised
    mags = rng.sample([x / 100 for x in range(5, 200)], size)
    for i, r in enumerate(rows):
        r["obs_log_contrast"] = round(
            rng.choice([-1, 1]) * mags[i], 2)
        r["obs_act_rank"] = round(rng.random(), 2)
    if construction in ("neg_extreme", "pos_extreme"):
        # row 0 is the strict |lc| winner with the construction's sign
        top = max(abs(r["obs_log_contrast"]) for r in rows)
        rows[0]["obs_log_contrast"] = round(
            (-1 if construction == "neg_extreme" else 1) * (top + 0.5), 2)
    # distinct ar values too (avoid ties on ar questions)
    ars = rng.sample([x / 100 for x in range(5, 99)], size)
    for r, a in zip(rows, ars):
        r["obs_act_rank"] = round(a, 2)
    return rows


def answers_for(rows):
    """Correct answer per qtype — exact at rendered precision. Ties
    cannot occur by construction but verified anyway."""
    out = {}
    lcs = [r["obs_log_contrast"] for r in rows]
    ars = [r["obs_act_rank"] for r in rows]
    def uniq_win(vals, pick):
        t = pick(vals)
        ws = [i for i, v in enumerate(vals) if v == t]
        return rows[ws[0]]["param_id"] if len(ws) == 1 else None
    out["largest_lc"] = uniq_win(lcs, max)
    out["smallest_lc"] = uniq_win(lcs, min)
    out["largest_abs_lc"] = uniq_win([abs(x) for x in lcs], max)
    out["largest_alc_col"] = out["largest_abs_lc"]
    out["largest_ar"] = uniq_win(ars, max)
    out["smallest_ar"] = uniq_win(ars, min)
    return out


def render(rows, alc=False):
    lines = []
    for r in rows:
        lc = round(r["obs_log_contrast"], 2)
        ar = round(r["obs_act_rank"], 2)
        base = (f"{r['param_id']}|L{r['layer']}|{r['family'][:6]}"
                f"|ar={ar:.2f}|lc={lc:.2f}")
        if alc:
            base += f"|alc={abs(lc):.2f}"
        lines.append(base)
    return "\n".join(lines)


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


def grade(expected, resp):
    import re
    t = resp.strip()
    if not re.fullmatch(r"[+-]?\d+", t):
        return "malformed"
    return "correct" if int(t) == expected else "wrong"


def main():
    report = {"model": MODEL_ID, "harness": HARNESS_VERSION,
              "manifest": {"harness_version": HARNESS_VERSION,
                           "sizes": SIZES, "n_instances": N_INSTANCES},
              "cells": {}}
    stored_manifest = {}
    if OUT.exists():
        report = json.loads(OUT.read_text())
        stored_manifest = dict(report.get("manifest", {}))
        print(f"[resume] {len(report['cells'])} cells", flush=True)

    provenance = local_provenance(MODEL_ID)
    report["manifest"]["model_revision"] = provenance["revision"]

    if stored_manifest:
        errs = validate_resume_manifest(stored_manifest, report["manifest"])
        if errs:
            print("REFUSING resume - manifest mismatch:", flush=True)
            for e in errs:
                print("  - " + e, flush=True)
            print("delete the stale results file or restore the matching configuration", flush=True)
            raise SystemExit(1)
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=provenance["revision"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=provenance["revision"], dtype=DTYPE,
        device_map={"": "cuda"}, local_files_only=True)
    model.eval()

    for size in SIZES:
        for construction in ("neg_extreme", "pos_extreme", "mixed"):
            for inst in range(N_INSTANCES):
                rng = random.Random(int.from_bytes(hashlib.sha256(
                    f"m3:{size}:{construction}:{inst}".encode()
                ).digest()[:8], "little"))
                rows = make_instance(rng, size, construction)
                ans = answers_for(rows)
                for qtype, phrasing in QTYPES:
                    if qtype == "largest_alc_col":
                        continue  # same answer as abs; separate alc run below
                    key = f"{size}:{construction}:{inst}:{qtype}"
                    if key in report["cells"] and "verdict" in report["cells"][key]:
                        continue
                    exp = ans[qtype]
                    if exp is None:
                        report["cells"][key] = {"skipped": "tie"}
                        continue
                    txt = render(rows)
                    q = (f"Which row has the {phrasing}? Reply with "
                         f"ONLY the id number.")
                    resp, n_in, lat = gen_text(model, tok,
                                               txt + "\n\n" + q)
                    report["cells"][key] = {
                        "size": size, "construction": construction,
                        "instance": inst, "qtype": qtype,
                        "expected": exp, "response": resp,
                        "verdict": grade(exp, resp),
                        "input_tokens": n_in,
                        "latency_s": round(lat, 2),
                        "rows_lc": [r["obs_log_contrast"] for r in rows]}
                    if len(report["cells"]) % 25 == 0:
                        OUT.write_text(json.dumps(report, indent=1))
                # alc column variant: one probe per instance
                key = f"{size}:{construction}:{inst}:alc_col"
                if key not in report["cells"]:
                    exp = ans["largest_abs_lc"]
                    if exp is None:
                        report["cells"][key] = {"skipped": "tie"}
                    else:
                        txt = render(rows, alc=True)
                        q = ("Which row has the largest alc? Reply "
                             "with ONLY the id number.")
                        resp, n_in, lat = gen_text(model, tok,
                                                   txt + "\n\n" + q)
                        report["cells"][key] = {
                            "size": size, "construction": construction,
                            "instance": inst, "qtype": "largest_alc_col",
                            "expected": exp, "response": resp,
                            "verdict": grade(exp, resp),
                            "input_tokens": n_in,
                            "latency_s": round(lat, 2),
                            "rows_lc": [r["obs_log_contrast"]
                                        for r in rows]}
                OUT.write_text(json.dumps(report, indent=1))
            done = [c for c in report["cells"].values() if "verdict" in c]
            ok = sum(1 for c in done if c["verdict"] == "correct")
            print(f"[{size}:{construction}] cumulative {ok}/{len(done)}",
                  flush=True)
    print("done -> results/gate1_3_micro_d3.json")


if __name__ == "__main__":
    sys.exit(main())
