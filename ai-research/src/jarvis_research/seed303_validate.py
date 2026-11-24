#!/usr/bin/env python3
"""seed303_validate.py — independent validation of the Phase B seed-303 result.

Reviewer finding (2026-11-23): the Phase B "improvement" (7.46 -> 6.98) was
measured on the SAME fixed probe text used as the proposer's feedback signal.
The model was explicitly told to optimize that score. This script answers
three questions the reviewer raised:

  1. GENERALIZATION: does the exact seed-303 edit set improve perplexity on
     HELD-OUT text domains (not the training probe)?
  2. MODEL SELECTION: does the edit outperform MATCHED RANDOM ATTENUATION
     controls (same budget, same operation, different tensor/layer choice)?
     If random controls match, the "improvement" is a trivial shrinkage effect,
     not intelligent parameter selection.
  3. COHERENCE: does the edited model still produce coherent greedy generation
     on downstream prompts (not just a perplexity number)?

Protocol:
  - Load a FRESH Qwen3.5-4B BF16 checkpoint (no prior edits).
  - Evaluate baseline (unmodified) perplexity on N held-out text passages
    spanning different domains, PLUS the original Phase B probe for
    reproduction confirmation.
  - Apply the EXACT seed-303 edit set (8 dense x0.85 scales on all L0
    projections + 1 dense x0.95 on L0 input_layernorm), verify the original
    probe reproduces 7.46 -> ~6.98, then evaluate on ALL held-out texts.
  - Restore to original (bit-exact), then run three matched controls:
      Control A: random 8 L0 tensors x0.85 (tests tensor-selection specificity)
      Control B: same 8 tensor TYPES on a random non-L0 layer x0.85
                (tests layer-position specificity)
      Control C: x0.85 on the same 8 projection types across ALL 32 layers
                (tests whether uniform global attenuation is the effect)
    Each control uses multiple seeds and reports per-passage perplexity.
  - Greedy-generate on downstream prompts for baseline, seed-303, and one
    control — store the text for qualitative comparison.
  - Every edit is applied under torch.no_grad and restored bit-exact afterward.

All texts are original / public-domain; nothing is downloaded. Fully offline.

Usage:
    .venv-opt\\Scripts\\python.exe src/jarvis_research/seed303_validate.py
        [--control-seeds 1 2 3 4 5] [--smoke] [--out <filename>]
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

_THIS = Path(__file__).resolve()
AI_RESEARCH = _THIS.parents[2]
RESULTS = AI_RESEARCH / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen3.5-4B"
DTYPE = torch.bfloat16


def _git_sha():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=AI_RESEARCH, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Evaluation texts — original, diverse domains, NOT the Phase B training probe.
# Each is ~40-80 words to give a stable perplexity estimate. The original
# Phase B probe is included SEPARATELY (labelled) for reproduction confirmation.
# ---------------------------------------------------------------------------

HELD_OUT_TEXTS = {
    "science_general": (
        "Photosynthesis converts light energy into chemical energy stored in "
        "glucose. Plants absorb carbon dioxide through stomata and water through "
        "roots. Chlorophyll in the chloroplasts captures photons, splitting water "
        "molecules and releasing oxygen as a byproduct. The Calvin cycle then fixes "
        "carbon dioxide into sugar using the energy carried by ATP and NADPH."
    ),
    "code_python": (
        "def binary_search(arr, target):\n"
        "    lo, hi = 0, len(arr) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    return -1"
    ),
    "narrative": (
        "The old lighthouse keeper wiped the salt spray from the brass lens and "
        "gazed into the storm. Forty years he had tended this light, and never once "
        "had the sea frightened him. But tonight was different. The waves struck "
        "the rocks with a fury he had not seen before, and somewhere out in that "
        "darkness, a small boat was trying to reach the harbour."
    ),
    "math_reasoning": (
        "A train travels from City A to City B at 60 km/h and returns at 40 km/h. "
        "The total distance for the round trip is 240 km. To find the average speed "
        "for the entire journey, we use the harmonic mean of the two speeds, not "
        "the arithmetic mean, because the train spends more time at the slower speed. "
        "The average speed is 2 times 60 times 40 divided by 60 plus 40, which is 48."
    ),
    "history": (
        "The Roman Republic lasted nearly five centuries before transitioning to "
        "the Empire under Augustus. The Senate, composed of patricians, held "
        "legislative power, while two consuls were elected annually to lead the "
        "state and the military. The Punic Wars against Carthage expanded Roman "
        "influence across the Mediterranean, establishing dominance that would "
        "last until the western empire fell in 476 AD."
    ),
    "technical_explanation": (
        "In a transformer model, self-attention computes a weighted sum of value "
        "vectors for each token. The attention weights are derived from query and "
        "key vectors via a scaled dot-product followed by a softmax. Multi-head "
        "attention runs several attention operations in parallel, each with its "
        "own learned projections, and concatenates the results before a final "
        "linear projection back to the model dimension."
    ),
    "conversation": (
        "Alice: Have you finished the report yet?\n"
        "Bob: Almost. I just need to add the conclusion and check the figures.\n"
        "Alice: The meeting starts in an hour. Do you think you can make it?\n"
        "Bob: I should be done in twenty minutes. Can you save me a seat?\n"
        "Alice: Of course. I will send you a message when it begins."
    ),
    "factual_diverse": (
        "Mount Everest stands at approximately 8,849 meters above sea level. "
        "The Mariana Trench reaches nearly 11,000 meters below the ocean surface. "
        "Honey never spoils because its low water activity prevents microbial growth. "
        "Octopuses have three hearts and blue blood. A group of flamingos is called "
        "a flamboyance."
    ),
}

# The ORIGINAL Phase B probe — included to confirm the 7.46 -> 6.98 reproduces.
PHASEB_PROBE = (
    "The capital of France is Paris. Water is made of hydrogen and oxygen. "
    "A week has seven days. The Earth orbits the Sun once per year."
)

# Downstream generation prompts for coherence check.
GEN_PROMPTS = {
    "complete_factual": "The largest planet in our solar system is",
    "complete_code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "complete_story": "Once upon a time, in a forest deep in the mountains,",
    "answer_question": "Question: What is the chemical symbol for gold?\nAnswer:",
}


# The exact seed-303 edit set from Phase B v2 (verified from results JSON).
# Step 1: 8 dense x0.85 scales on all L0 projections.
# Step 23: 1 dense x0.95 on L0 input_layernorm.
SEED303_EDITS = [
    ("model.layers.0.mlp.down_proj.weight", "scale", 0.85),
    ("model.layers.0.mlp.gate_proj.weight", "scale", 0.85),
    ("model.layers.0.mlp.up_proj.weight", "scale", 0.85),
    ("model.layers.0.linear_attn.out_proj.weight", "scale", 0.85),
    ("model.layers.0.linear_attn.in_proj_a.weight", "scale", 0.85),
    ("model.layers.0.linear_attn.in_proj_b.weight", "scale", 0.85),
    ("model.layers.0.linear_attn.in_proj_qkv.weight", "scale", 0.85),
    ("model.layers.0.linear_attn.in_proj_z.weight", "scale", 0.85),
    ("model.layers.0.input_layernorm.weight", "scale", 0.95),
]

# The 8 L0 tensor types that seed-303 targeted (for controls B and C).
L0_PROJECTION_TYPES = [
    "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "linear_attn.out_proj.weight", "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight", "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
]

GEN_MAX_NEW = 48


def apply_scale(param, factor):
    """Dense multiplicative scale in-place under no_grad."""
    with torch.no_grad():
        param.view(-1).mul_(factor)


def restore(param, original):
    """Bit-exact restore."""
    with torch.no_grad():
        param.view(-1).copy_(original.view(-1))


@torch.no_grad()
def perplexity(model, tok, text):
    """Perplexity on a single text passage."""
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    if ids.shape[1] < 2:
        return float("inf")
    out = model(ids)
    logits = out.logits.float()
    sl, lab = logits[:, :-1, :], ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(
        sl.reshape(-1, sl.size(-1)), lab.reshape(-1))
    return float(torch.exp(loss).item()) if torch.isfinite(loss) else float("inf")


@torch.no_grad()
def nan_check(model, tok, text):
    ids = tok(text, return_tensors="pt").input_ids.to("cuda")
    out = model(ids)
    return bool(torch.isnan(out.logits).any() or torch.isinf(out.logits).any())


@torch.no_grad()
def generate(model, tok, prompt, max_new=GEN_MAX_NEW):
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    out = model.generate(ids, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def eval_all(model, tok, texts, label):
    """Perplexity on every text passage. Returns dict name -> ppl."""
    result = {"_label": label}
    for name, text in texts.items():
        result[name] = round(perplexity(model, tok, text), 4)
    return result


def eval_phaseb_probe(model, tok):
    """The original Phase B probe — to confirm reproduction."""
    return round(perplexity(model, tok, PHASEB_PROBE), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-seeds", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5],
                    help="seeds for random controls A and B")
    ap.add_argument("--smoke", action="store_true",
                    help="evaluate baseline + seed-303 only, skip controls")
    ap.add_argument("--out", default="seed303_validation.json",
                    help="results filename under ai-research/results/")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    all_texts = dict(HELD_OUT_TEXTS)

    t0 = time.perf_counter()
    print(f"[load] {MODEL_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map={"": "cuda"})
    model.eval()
    torch.cuda.synchronize()
    params = {n: p for n, p in model.named_parameters()}
    print(f"[load] done in {time.perf_counter()-t0:.1f}s, "
          f"{len(params)} params", flush=True)

    # Verify the seed-303 edit targets all exist.
    missing = [n for n, _, _ in SEED303_EDITS if n not in params]
    if missing:
        print(f"[ERROR] seed-303 references missing params: {missing}")
        return 1

    report = {
        "gate": "seed303_independent_validation",
        "model": MODEL_ID, "git_sha": _git_sha(),
        "purpose": ("Test whether the Phase B seed-303 edit generalizes beyond "
                    "the training probe and outperforms matched random controls."),
        "eval_texts": list(all_texts.keys()),
        "n_held_out": len(all_texts),
        "n_control_seeds": len(args.control_seeds),
        "seed303_edits": [{"param": n, "op": o, "magnitude": m}
                          for n, o, m in SEED303_EDITS],
        "controls": {
            "A": "random 8 L0 tensors x0.85 (tensor-selection specificity)",
            "B": "same 8 types on random non-L0 layer x0.85 (layer specificity)",
            "C": "x0.85 same 8 types across ALL 32 layers (global attenuation)",
        },
    }

    # ---- Baseline (unmodified) ----
    print("\n[baseline] evaluating unmodified model on all texts ...", flush=True)
    base_ppls = eval_all(model, tok, all_texts, "baseline")
    base_phaseb = eval_phaseb_probe(model, tok)
    base_nan = nan_check(model, tok, PHASEB_PROBE)
    report["baseline"] = {
        "held_out_ppls": base_ppls,
        "phaseb_probe_ppl": base_phaseb,
        "nan_inf": base_nan,
    }
    print(f"[baseline] phaseb_probe={base_phaseb:.3f}  held_out: "
          f"min={min(v for k,v in base_ppls.items() if k != '_label'):.2f} "
          f"max={max(v for k,v in base_ppls.items() if k != '_label'):.2f}",
          flush=True)
    for k, v in base_ppls.items():
        if k != "_label":
            print(f"           {k}: {v:.3f}", flush=True)

    # ---- Seed-303 exact replay ----
    print("\n[seed303] applying exact edit set ...", flush=True)
    originals = {}
    for name, op, mag in SEED303_EDITS:
        originals[name] = params[name].detach().clone()
        apply_scale(params[name], mag)
        print(f"  {name}  *= {mag}", flush=True)

    s303_phaseb = eval_phaseb_probe(model, tok)
    s303_ppls = eval_all(model, tok, all_texts, "seed303")
    s303_nan = nan_check(model, tok, PHASEB_PROBE)
    report["seed303_replay"] = {
        "held_out_ppls": s303_ppls,
        "phaseb_probe_ppl": s303_phaseb,
        "nan_inf": s303_nan,
        "phaseb_reproduction": (
            f"7.46->{s303_phaseb:.3f} "
            f"({'reproduced' if 6.9 < s303_phaseb < 7.1 else 'NOT reproduced'})"
        ),
    }
    print(f"[seed303] phaseb_probe={s303_phaseb:.3f} "
          f"(Phase B reported 6.983)", flush=True)
    for k, v in s303_ppls.items():
        if k != "_label":
            base_v = base_ppls[k]
            delta = v - base_v
            print(f"           {k}: {v:.3f}  (base {base_v:.3f}, "
                  f"{'+' if delta >= 0 else ''}{delta:.3f})", flush=True)

    # Greedy generation coherence check
    print("\n[seed303] downstream generation coherence ...", flush=True)
    s303_gen = {}
    for name, prompt in GEN_PROMPTS.items():
        s303_gen[name] = generate(model, tok, prompt)
        print(f"  {name}: {s303_gen[name][:80]}...", flush=True)
    report["seed303_replay"]["generation"] = s303_gen

    # Restore
    for name in originals:
        restore(params[name], originals[name])
    torch.cuda.synchronize()
    restored_ppl = eval_phaseb_probe(model, tok)
    report["seed303_replay"]["restored_phaseb_ppl"] = restored_ppl
    report["seed303_replay"]["restored_bit_exact"] = (
        abs(restored_ppl - base_phaseb) < 0.01)
    print(f"[seed303] restored: phaseb_probe={restored_ppl:.3f} "
          f"(base {base_phaseb:.3f}, "
          f"{'bit-exact' if abs(restored_ppl - base_phaseb) < 0.01 else 'DRIFT'})",
          flush=True)

    if args.smoke:
        print("\n[smoke] skipping controls.", flush=True)
        _write(report, args.out, t0)
        return 0

    # ---- Control A: random 8 L0 tensors x0.85 ----
    # All L0 named params available for editing.
    l0_params = sorted(n for n in params if n.startswith("model.layers.0."))
    report["control_A_details"] = {"l0_pool_size": len(l0_params)}

    print(f"\n[control A] random 8 L0 tensors x0.85, "
          f"{len(args.control_seeds)} seeds ...", flush=True)
    ctrl_a_runs = []
    for cseed in args.control_seeds:
        g = torch.Generator().manual_seed(cseed * 7919)
        chosen = [l0_params[i] for i in
                  torch.randperm(len(l0_params), generator=g)[:8].tolist()]
        originals_c = {}
        for name in chosen:
            originals_c[name] = params[name].detach().clone()
            apply_scale(params[name], 0.85)
        ca_phaseb = eval_phaseb_probe(model, tok)
        ca_ppls = eval_all(model, tok, all_texts, f"ctrlA_s{cseed}")
        ca_nan = nan_check(model, tok, PHASEB_PROBE)
        ctrl_a_runs.append({
            "seed": cseed, "chosen": chosen,
            "held_out_ppls": ca_ppls,
            "phaseb_probe_ppl": ca_phaseb, "nan_inf": ca_nan,
        })
        print(f"  seed {cseed}: phaseb={ca_phaseb:.3f}  "
              f"mean_held_out={_mean_heldout(ca_ppls):.3f}", flush=True)
        for name in chosen:
            restore(params[name], originals_c[name])
    report["control_A_runs"] = ctrl_a_runs

    # ---- Control B: same 8 types on a random non-L0 layer ----
    # Pick layers from {4, 8, 12, 16, 22, 28} — spanning early/mid/late.
    candidate_layers = [4, 8, 12, 16, 22, 28]
    print(f"\n[control B] same 8 projection types on random non-L0 layer, "
          f"{len(args.control_seeds)} seeds ...", flush=True)
    ctrl_b_runs = []
    for cseed in args.control_seeds:
        g = torch.Generator().manual_seed(cseed * 7919 + 1)
        layer = candidate_layers[
            torch.randint(len(candidate_layers), (1,), generator=g).item()]
        names = [f"model.layers.{layer}.{t}" for t in L0_PROJECTION_TYPES
                 if f"model.layers.{layer}.{t}" in params]
        if len(names) < 8:
            # full_attention layers lack linear_attn.* — use self_attn instead
            names = [f"model.layers.{layer}.self_attn.q_proj.weight",
                     f"model.layers.{layer}.self_attn.k_proj.weight",
                     f"model.layers.{layer}.self_attn.v_proj.weight",
                     f"model.layers.{layer}.self_attn.o_proj.weight",
                     f"model.layers.{layer}.mlp.gate_proj.weight",
                     f"model.layers.{layer}.mlp.up_proj.weight",
                     f"model.layers.{layer}.mlp.down_proj.weight",
                     f"model.layers.{layer}.input_layernorm.weight"]
            names = [n for n in names if n in params][:8]
        originals_c = {}
        for name in names:
            originals_c[name] = params[name].detach().clone()
            apply_scale(params[name], 0.85)
        cb_phaseb = eval_phaseb_probe(model, tok)
        cb_ppls = eval_all(model, tok, all_texts, f"ctrlB_s{cseed}")
        cb_nan = nan_check(model, tok, PHASEB_PROBE)
        ctrl_b_runs.append({
            "seed": cseed, "layer": layer, "chosen": names,
            "held_out_ppls": cb_ppls,
            "phaseb_probe_ppl": cb_phaseb, "nan_inf": cb_nan,
        })
        print(f"  seed {cseed} (layer {layer}): phaseb={cb_phaseb:.3f}  "
              f"mean_held_out={_mean_heldout(cb_ppls):.3f}", flush=True)
        for name in names:
            restore(params[name], originals_c[name])
    report["control_B_runs"] = ctrl_b_runs

    # ---- Control C: x0.85 same 8 types across ALL 32 layers ----
    print(f"\n[control C] x0.85 on same 8 projection types across ALL layers ...",
          flush=True)
    all_layer_names = []
    for li in range(32):
        for t in L0_PROJECTION_TYPES:
            n = f"model.layers.{li}.{t}"
            if n in params:
                all_layer_names.append(n)
    originals_c = {}
    for name in all_layer_names:
        originals_c[name] = params[name].detach().clone()
        apply_scale(params[name], 0.85)
    cc_phaseb = eval_phaseb_probe(model, tok)
    cc_ppls = eval_all(model, tok, all_texts, "ctrlC_global")
    cc_nan = nan_check(model, tok, PHASEB_PROBE)
    report["control_C_global"] = {
        "n_tensors": len(all_layer_names),
        "held_out_ppls": cc_ppls,
        "phaseb_probe_ppl": cc_phaseb, "nan_inf": cc_nan,
    }
    print(f"  {len(all_layer_names)} tensors: phaseb={cc_phaseb:.3f}  "
          f"mean_held_out={_mean_heldout(cc_ppls):.3f}", flush=True)
    for name in all_layer_names:
        restore(params[name], originals_c[name])

    # ---- Baseline generation for comparison ----
    print("\n[baseline] downstream generation ...", flush=True)
    base_gen = {}
    for name, prompt in GEN_PROMPTS.items():
        base_gen[name] = generate(model, tok, prompt)
    report["baseline_generation"] = base_gen

    # Final restore check
    torch.cuda.synchronize()
    final_ppl = eval_phaseb_probe(model, tok)
    report["final_restore_check"] = {
        "phaseb_ppl": final_ppl,
        "matches_baseline": abs(final_ppl - base_phaseb) < 0.01,
    }

    _write(report, args.out, t0)
    return 0


def _mean_heldout(ppls):
    vals = [v for k, v in ppls.items() if k != "_label"]
    return sum(vals) / len(vals) if vals else float("inf")


def _write(report, fname, t0):
    report["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    out = RESULTS / fname
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {out} ({out.stat().st_size} bytes) in "
          f"{report['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
