"""HZN-IR-11 D3 — minimal arithmetic control (paired sentinel).

Distinguishes numerical-comparison failure from competing-field
interference, using IDENTICAL numerical instances under three
presentations:
  V1 signed-only:       rows carry lc only; question asks largest |lc|
  V2 signed+derived:    rows carry lc AND derived alc column
  V3 abs-only:          rows carry ONLY alc (signed lc excluded)

If V3 >> V1/V2, the failure is competing-field interference or
signed-max capture, not magnitude comparison itself. If V3 also
fails, multi-item numerical comparison is broken regardless of
presentation. 10 instances per variant per size (4 and 8 rows),
randomized identifier assignment. Small by directive — not a
comprehensive arithmetic campaign.
"""
import hashlib, json, random, sys, time
sys.path.insert(0, r"C:\Users\Shadow\Documents\Sims2026\ai-research\src")
from pathlib import Path

AI_RESEARCH = Path(r"C:\Users\Shadow\Documents\Sims2026\ai-research")
OUT = AI_RESEARCH / "results" / "gate1_4_arith_control.json"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jarvis_research.gate1_pilot import MODEL_ID, DTYPE, local_provenance
from jarvis_research.run_manifest import validate_resume_manifest

SIZES = [4, 8]
N_INSTANCES = 10
HARNESS_VERSION = "arith-control-1"
SYSTEM = ("You are reading an observation catalog from a self-editing "
          "experiment. Answer each probe precisely. Reply with ONLY "
          "the requested format — no reasoning, no extra text.")


def make_instance(rng, size):
    ids = rng.sample(range(200), size)  # randomized identifier assignment
    mags = rng.sample([x / 100 for x in range(5, 200)], size)
    rows = []
    for i in range(size):
        rows.append({"id": ids[i],
                     "lc": round(rng.choice([-1, 1]) * mags[i], 2)})
    # strict unique |lc| winner by construction (mags distinct)
    winner = max(rows, key=lambda r: abs(r["lc"]))
    return rows, winner["id"]


def render(rows, variant):
    lines = []
    for r in rows:
        if variant == "V1":
            lines.append(f"{r['id']}|lc={r['lc']:.2f}")
        elif variant == "V2":
            lines.append(f"{r['id']}|lc={r['lc']:.2f}|alc={abs(r['lc']):.2f}")
        else:
            lines.append(f"{r['id']}|alc={abs(r['lc']):.2f}")
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
                          "sizes": SIZES, "n_instances": N_INSTANCES}
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

    for size in SIZES:
        for inst in range(N_INSTANCES):
            rng = random.Random(int.from_bytes(hashlib.sha256(
                f"ac:{size}:{inst}".encode()).digest()[:8], "little"))
            rows, winner = make_instance(rng, size)
            for variant in ("V1", "V2", "V3"):
                key = f"{size}:{inst}:{variant}"
                if key in report["cells"] and "verdict" in report["cells"][key]:
                    continue
                txt = render(rows, variant)
                col = "|lc|" if variant == "V1" else "alc"
                q = (f"Which row has the largest {col}? Reply with "
                     f"ONLY the id number.")
                resp, n_in, lat = gen_text(model, tok, txt + "\n\n" + q)
                import re
                ok = bool(re.fullmatch(r"[+-]?\d+", resp.strip())) and \
                    int(resp.strip()) == winner
                report["cells"][key] = {
                    "size": size, "instance": inst, "variant": variant,
                    "expected": winner, "response": resp,
                    "verdict": "correct" if ok else "wrong",
                    "lcs": [r["lc"] for r in rows],
                    "ids": [r["id"] for r in rows],
                    "input_tokens": n_in, "latency_s": round(lat, 2)}
                OUT.write_text(json.dumps(report, indent=1))
        done = [c for c in report["cells"].values() if "verdict" in c]
        ok_n = sum(1 for c in done if c["verdict"] == "correct")
        print(f"[size {size}] cumulative {ok_n}/{len(done)}", flush=True)
    print("done -> results/gate1_4_arith_control.json")


if __name__ == "__main__":
    sys.exit(main())
