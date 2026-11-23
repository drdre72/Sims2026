"""Gate 0.5 Phase A.2 — collapse-seeking escalation sweep.

The Phase A sparse-scalar sweep (gate0.5_failure_taxonomy.json) found ZERO
collapses: a few-hundred scattered scalars are too small an L2 perturbation
for a multi-million-element tensor. The paper's "4B broke itself" must come
from a different edit model. This script finds the actual lethal boundary.

Edit modes probed (escalating strength until collapse):
  * add_dense   : param += d  (all elements, same direction)     coherent add
  * scale       : param *= (1 + d)                             coherent scale
  * noise_dense : param += d * std * randn                     dense noise
  * zero        : param *= 0                                   destructive
  * compound    : K sequential sparse-scalar edits, no rollback  accumulation

Collapse = NaN/Inf in logits, or degenerate greedy output, or perplexity
explosion (>5x baseline or >200 absolute). Each probe rolls back bit-exactly
except `compound`, which intentionally accumulates.

Emits ai-research/results/gate0.5_failure_boundary.json (gitignored).

    .venv-opt\\Scripts\\python.exe src/jarvis_research/failure_boundary.py
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

# Escalation ladder for coherent/dense strengths (unitless multipliers of std,
# or absolute for the coherence test). We climb until collapse or cap.
STRENGTHS = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0]
COMPOUND_STEPS = [2, 4, 8, 16, 32, 64, 128]

# Pick one representative param per family (early + late layer) to escalate.
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


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map={"": "cuda"}
    )
    model.eval()
    torch.cuda.synchronize()
    params = {n: p for n, p in model.named_parameters()}

    report = {
        "gate": "gate_0_5_phase_a2_failure_boundary",
        "model": MODEL_ID, "seed": SEED, "git_sha": _git_sha(),
        "collapse_criteria": {
            "nan_inf": True, "ppl_ratio": PPL_RATIO_COLLAPSE,
            "ppl_abs": PPL_ABS_COLLAPSE, "degenerate_tokens": GEN_CHECK_TOKENS,
        },
        "probes": [],
    }

    probe_ids = tok(PROBE_TEXT, return_tensors="pt").input_ids.to("cuda")

    @torch.no_grad()
    def ppl_nan():
        out = model(probe_ids)
        logits = out.logits.float()
        nan = bool(torch.isnan(logits).any() or torch.isinf(logits).any())
        sl = logits[:, :-1, :]
        lab = probe_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            sl.reshape(-1, sl.size(-1)), lab.reshape(-1))
        ppl = float(torch.exp(loss).item()) if torch.isfinite(loss) else float("inf")
        return ppl, nan

    @torch.no_grad()
    def degenerate():
        ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
        out = model.generate(ids, max_new_tokens=GEN_CHECK_TOKENS, do_sample=False)
        toks = out[0, ids.shape[1]:].tolist()
        return (not toks) or len(set(toks)) <= 1

    base_ppl, base_nan = ppl_nan()
    report["baseline"] = {"probe_ppl": round(base_ppl, 3), "nan_inf": base_nan}

    def evaluate(fam, pname, mode, strength, extra=None):
        ppl, nan = ppl_nan()
        deg = degenerate()
        ratio = (ppl / base_ppl) if math.isfinite(ppl) and base_ppl > 0 else float("inf")
        collapse = nan or deg or ratio > PPL_RATIO_COLLAPSE or ppl > PPL_ABS_COLLAPSE
        mech = [m for m, c in
                (("nan_inf", nan), ("degenerate", deg),
                 ("ppl_explosion", ratio > PPL_RATIO_COLLAPSE or ppl > PPL_ABS_COLLAPSE))
                if c]
        rec = {
            "family": fam, "param": pname, "layer": layer_idx(pname),
            "mode": mode, "strength": strength,
            "ppl_after": round(ppl, 3) if math.isfinite(ppl) else "inf",
            "ppl_ratio": round(ratio, 3) if math.isfinite(ratio) else "inf",
            "nan_inf": nan, "degenerate": deg,
            "collapsed": collapse, "mechanisms": mech,
        }
        if extra:
            rec.update(extra)
        report["probes"].append(rec)
        tag = "COLLAPSE" if collapse else "ok"
        print(f"[{tag:8s}] {fam:14s} {mode:12s} k={strength} "
              f"ppl->{rec['ppl_after']} nan={nan} deg={deg}")
        return collapse

    # representative params: first occurrence of each family, early + late
    reps = {}
    for n in params:
        fam, li = classify(n), layer_idx(n)
        if li is None:
            reps.setdefault((fam, "global"), n)
        elif li <= 10:
            reps.setdefault((fam, "early"), n)
        elif li >= 22:
            reps.setdefault((fam, "late"), n)

    # --- Escalation over coherent/dense modes ------------------------------
    for (fam, reg), pname in sorted(reps.items()):
        p = params[pname]
        orig = p.detach().clone()
        std = float(p.std().item())
        for mode in ("add_dense", "scale", "noise_dense", "zero"):
            collapsed_at = None
            for k in STRENGTHS:
                with torch.no_grad():
                    if mode == "add_dense":
                        p += k * std                      # coherent additive
                    elif mode == "scale":
                        p *= (1.0 + k)                    # coherent scale
                    elif mode == "noise_dense":
                        g = torch.Generator(device="cpu").manual_seed(SEED)
                        p += k * std * torch.randn(p.shape, generator=g).to(p.device, p.dtype)
                    elif mode == "zero":
                        p *= 0.0                          # destructive
                torch.cuda.synchronize()
                if evaluate(fam, pname, mode, k):
                    collapsed_at = k
                    break
                with torch.no_grad():
                    p.copy_(orig)  # restore before next strength
            with torch.no_grad():
                p.copy_(orig)
            rec = report["probes"][-1]
            rec["collapsed_at_k"] = collapsed_at

    # --- Compounding: sequential sparse edits, NO rollback -----------------
    # For the most fragile families, apply K accumulated sparse-scalar edits
    # and find the step at which the model collapses.
    for fam in ("recurrence", "mlp", "attention"):
        cands = [n for n in params if classify(n) == fam]
        if not cands:
            continue
        pname = max(cands, key=lambda n: params[n].numel())
        p = params[pname]
        orig = p.detach().clone()
        g = torch.Generator(device="cpu").manual_seed(SEED)
        n = p.numel()
        perm = torch.randperm(n, generator=g)
        std = float(p.std().item())
        delta = 0.1 * std  # modest per-edit delta, accumulate
        with torch.no_grad():
            p.copy_(orig)
        prev = 0
        for k in COMPOUND_STEPS:
            with torch.no_grad():
                idx = perm[prev:k].to(p.device)
                if idx.numel() == 0:
                    break
                p.view(-1)[idx] += delta
            prev = k
            torch.cuda.synchronize()
            if evaluate(fam, pname, "compound", k, extra={"per_edit_delta": round(delta, 6)}):
                break
        with torch.no_grad():
            p.copy_(orig)
        torch.cuda.synchronize()

    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    out = RESULTS / "gate0.5_failure_boundary.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[failure_boundary] {len(report['probes'])} probes -> {out}")
    _summarize(report)
    return 0


def _summarize(report):
    from collections import defaultdict
    first = {}
    for pr in report["probes"]:
        key = (pr["family"], pr["mode"])
        if pr["collapsed"] and key not in first:
            first[key] = pr["strength"]
    print("\n=== First-collapse strength by family/mode ===")
    if not first:
        print("  (no collapses reached at probed strengths)")
    for (fam, mode), k in sorted(first.items()):
        print(f"  {fam:14s} {mode:12s} collapses at k={k}")


def _git_sha():
    try:
        import subprocess
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=AI_RESEARCH.parent,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
