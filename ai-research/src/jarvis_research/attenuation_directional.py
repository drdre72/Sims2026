#!/usr/bin/env python
"""attenuation_directional.py — Phase A extension: bounded downward scaling.

Reviewer finding 2 (2026-11-23): Phase A's failure_boundary.py tested `scale`
as `p *= (1 + k)` — only UPWARD (growth). Phase B discovered that dense
DOWNWARD scaling (attenuation, p *= 0.85) survives on Layer 0, where coherent
additive edits collapse at k=0.25. This is a coverage gap: the safe surface
for attenuation was never mapped.

This script fills that gap. It tests dense multiplicative attenuation
`p *= factor` (factor < 1.0) on the same depth-gradient params Phase A used
(L0, L1, L12, L22) plus the exact L0 projection block seed-303 edited, to
answer:

  1. At what attenuation factor does each layer/param collapse?
  2. Is the attenuation safe surface wider than the additive surface?
  3. Does the seed-303 ×0.85 sit safely inside or near the boundary?
  4. Does multi-layer block attenuation (scaling L0+L1 together, etc.)
     behave differently from single-layer?

Collapse criteria identical to Phase A: NaN/Inf, degenerate greedy output,
perplexity ratio >5x baseline or >200 absolute.

All probes roll back bit-exact between strengths.

    .venv-opt\\Scripts\\python.exe src/jarvis_research/attenuation_directional.py
        [--smoke] [--out <filename>]
"""

from __future__ import annotations

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
SEED = 1234

PROBE_TEXT = (
    "The capital of France is Paris. Water is made of hydrogen and oxygen. "
    "A week has seven days. The Earth orbits the Sun once per year."
)
GEN_CHECK_TOKENS = 16
PPL_RATIO_COLLAPSE = 5.0
PPL_ABS_COLLAPSE = 200.0

# Attenuation factors (downward only: factor < 1.0).
# 0.99 = gentle shrink; 0.0 = full zeroing (the Phase A "zero" reference point).
FACTORS = [0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.30, 0.10, 0.0]

# Depth-gradient layers (matching Phase A's gradient map).
DEPTH_LAYERS = [0, 1, 3, 12, 22, 28]

# The 8 projection types seed-303 hit at L0 — test these at every depth layer.
SEED303_PROJ_TYPES = [
    "mlp.down_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
    "linear_attn.out_proj.weight", "linear_attn.in_proj_a.weight",
    "linear_attn.in_proj_b.weight", "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
]

FAMILY_RULES = [
    ("linear_attn.dt_bias", "recurrence"),
    ("linear_attn.A_log", "recurrence"),
    ("linear_attn.conv1d", "deltanet_proj"),
    ("linear_attn.in_proj", "deltanet_proj"),
    ("linear_attn.out_proj", "deltanet_proj"),
    ("linear_attn.norm", "norm"),
    ("self_attn", "attention"),
    ("mlp.gate_proj", "mlp"),
    ("mlp.up_proj", "mlp"),
    ("mlp.down_proj", "mlp"),
    ("layernorm", "norm"),
    ("embed_tokens", "embedding"),
    ("lm_head", "embedding"),
    ("model.norm", "norm"),
]


def classify(name):
    for sub, fam in FAMILY_RULES:
        if sub in name:
            return fam
    return "other"


def layer_idx(name):
    p = name.split(".")
    if "layers" in p:
        i = p.index("layers")
        if i + 1 < len(p):
            return int(p[i + 1])
    return None


def _git_sha():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=AI_RESEARCH, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse_helper()
    args = ap.parse_args()

    t0 = time.perf_counter()
    print(f"[load] {MODEL_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map={"": "cuda"})
    model.eval()
    torch.cuda.synchronize()
    params = {n: p for n, p in model.named_parameters()}
    print(f"[load] done in {time.perf_counter()-t0:.1f}s, {len(params)} params",
          flush=True)

    probe_ids = tok(PROBE_TEXT, return_tensors="pt").input_ids.to("cuda")

    @torch.no_grad()
    def ppl_nan():
        out = model(probe_ids)
        logits = out.logits.float()
        nan = bool(torch.isnan(logits).any() or torch.isinf(logits).any())
        sl, lab = logits[:, :-1, :], probe_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            sl.reshape(-1, sl.size(-1)), lab.reshape(-1))
        ppl = float(torch.exp(loss).item()) if torch.isfinite(loss) else float("inf")
        return ppl, nan

    @torch.no_grad()
    def degenerate():
        ids = tok("The capital of France is",
                  return_tensors="pt").input_ids.to("cuda")
        out = model.generate(ids, max_new_tokens=GEN_CHECK_TOKENS, do_sample=False)
        toks = out[0, ids.shape[1]:].tolist()
        return (not toks) or len(set(toks)) <= 1

    base_ppl, base_nan = ppl_nan()
    print(f"[baseline] probe ppl={base_ppl:.3f} nan={base_nan}", flush=True)

    report = {
        "gate": "attenuation_directional_sweep",
        "model": MODEL_ID, "seed": SEED, "git_sha": _git_sha(),
        "purpose": ("Map the downward-scaling (attenuation) collapse boundary "
                    "to close the Phase A coverage gap (only tested scale up)."),
        "factors": FACTORS,
        "depth_layers": DEPTH_LAYERS,
        "collapse_criteria": {
            "nan_inf": True, "ppl_ratio": PPL_RATIO_COLLAPSE,
            "ppl_abs": PPL_ABS_COLLAPSE, "degenerate_tokens": GEN_CHECK_TOKENS,
        },
        "baseline_probe_ppl": round(base_ppl, 3),
        "probes": [],
    }

    def evaluate(label, pnames, factor, extra=None):
        """Apply factor to all pnames, evaluate, restore."""
        origs = {n: params[n].detach().clone() for n in pnames}
        with torch.no_grad():
            for n in pnames:
                params[n].view(-1).mul_(factor)
        torch.cuda.synchronize()
        ppl, nan = ppl_nan()
        deg = degenerate()
        ratio = (ppl / base_ppl) if math.isfinite(ppl) and base_ppl > 0 else float("inf")
        collapse = (nan or deg or ratio > PPL_RATIO_COLLAPSE
                    or ppl > PPL_ABS_COLLAPSE)
        mech = [m for m, c in
                (("nan_inf", nan), ("degenerate", deg),
                 ("ppl_explosion", ratio > PPL_RATIO_COLLAPSE
                  or ppl > PPL_ABS_COLLAPSE)) if c]
        rec = {
            "label": label, "params": pnames, "factor": factor,
            "n_tensors": len(pnames),
            "ppl_after": round(ppl, 3) if math.isfinite(ppl) else "inf",
            "ppl_ratio": round(ratio, 3) if math.isfinite(ratio) else "inf",
            "nan_inf": nan, "degenerate": deg,
            "collapsed": collapse, "mechanisms": mech,
        }
        if extra:
            rec.update(extra)
        report["probes"].append(rec)
        tag = "COLLAPSE" if collapse else "ok"
        print(f"[{tag:8s}] {label:40s} x{factor:.2f} ppl->{rec['ppl_after']} "
              f"nan={nan} deg={deg}", flush=True)
        with torch.no_grad():
            for n in pnames:
                params[n].view(-1).copy_(origs[n].view(-1))
        return collapse

    # ---- Single-layer single-param: each seed-303 projection type at each depth ----
    print("\n=== Single-tensor attenuation: seed-303 projection types x depth ===",
          flush=True)
    for li in DEPTH_LAYERS:
        for pt in SEED303_PROJ_TYPES:
            pname = f"model.layers.{li}.{pt}"
            if pname not in params:
                continue
            fam = classify(pname)
            for factor in FACTORS:
                if evaluate(f"{fam} L{li} {pt.split('.')[0]}.{pt.split('.')[1]}",
                            [pname], factor,
                            {"family": fam, "layer": li,
                             "param": pname, "mode": "single_tensor"}):
                    break  # stop at first collapse

    # ---- Whole L0 projection block (what seed-303 did) at each depth ----
    print("\n=== Block attenuation: all 8 projections x depth (seed-303 pattern) ===",
          flush=True)
    for li in DEPTH_LAYERS:
        names = [f"model.layers.{li}.{pt}" for pt in SEED303_PROJ_TYPES
                 if f"model.layers.{li}.{pt}" in params]
        if not names:
            continue
        for factor in FACTORS:
            if evaluate(f"block L{li} (8 projections)", names, factor,
                        {"mode": "block_8proj", "layer": li,
                         "n_tensors": len(names)}):
                break

    # ---- Multi-layer block: L0+L1, L0+L1+L2, etc. (cross-layer attenuation) ----
    print("\n=== Multi-layer block attenuation: L0..Ln x 0.85 ===", flush=True)
    for n_layers in [1, 2, 3, 4, 8, 16, 32]:
        names = []
        for li in range(n_layers):
            for pt in SEED303_PROJ_TYPES:
                n = f"model.layers.{li}.{pt}"
                if n in params:
                    names.append(n)
        if not names:
            continue
        for factor in [0.99, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50, 0.0]:
            if evaluate(f"block L0..L{n_layers-1} x{factor} ({len(names)} tensors)",
                         names, factor,
                         {"mode": "multilayer_block",
                          "n_layers": n_layers, "n_tensors": len(names)}):
                break

    # ---- Full-attention layers at depth (they have self_attn, not linear_attn) ----
    print("\n=== Full-attention layer projection attenuation x depth ===", flush=True)
    fa_types = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                "mlp.gate_proj.weight", "mlp.up_proj.weight",
                "mlp.down_proj.weight"]
    # Find which layers are full_attention
    fa_layers = []
    for li in DEPTH_LAYERS:
        if f"model.layers.{li}.self_attn.q_proj.weight" in params:
            fa_layers.append(li)
    for li in fa_layers:
        names = [f"model.layers.{li}.{t}" for t in fa_types
                 if f"model.layers.{li}.{t}" in params]
        for factor in FACTORS:
            if evaluate(f"fa_block L{li} (7 projections)", names, factor,
                        {"mode": "fa_block", "layer": li,
                         "n_tensors": len(names)}):
                break

    report["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
    out = RESULTS / args.out
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[done] wrote {out} ({out.stat().st_size} bytes) in "
          f"{report['elapsed_seconds']}s", flush=True)
    print(f"[summary] {len(report['probes'])} probes, "
          f"{sum(1 for p in report['probes'] if p['collapsed'])} collapses",
          flush=True)
    return 0


def argparse_helper():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run a reduced grid (L0 only)")
    ap.add_argument("--out", default="attenuation_directional.json",
                    help="results filename under ai-research/results/")
    return ap


if __name__ == "__main__":
    raise SystemExit(main())
