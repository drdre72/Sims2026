"""Gate 0.5 Phase A — failure-surface sweep (failure-mode reverse-engineering).

Maps WHERE Qwen3.5-4B collapses under weight edits, independent of what a
proposer chooses. The output is the quantitative core of the failure-mode
taxonomy and directly yields the constrained edit surface for Gate 1 — the
constraints forbid exactly the failure modes observed here, not arbitrary
parameter families chosen for safety.

Protocol (unconstrained sweep, deterministic):
  For each (tensor family, layer region, delta magnitude):
    1. pick a representative parameter for that family in that region
    2. apply a sparse scalar edit: S scalars += delta
       (delta = magnitude x tensor-std for the relative sweep; a fixed value
        for the absolute anchors matching Gate 1's concrete bounds)
    3. evaluate collapse: NaN/Inf logits, perplexity ratio on a fixed probe
       text, and a short greedy degeneracy check
    4. roll the edit back bit-exactly
  Record every probe. Emit gate0.5_failure_taxonomy.json.

Why sparse scalar edits: Gate 1's constraint is expressed as a bounded count
of scalars with bounded deltas (<=4096 scalars, delta range). A sparse-scalar
edit is the same edit model the constraint governs; dense tensor noise is not.

Run under the FLA-accelerated env (Gate 0.1 default):

    .venv-opt\\Scripts\\python.exe src/jarvis_research/failure_diagnostic.py

Writes ai-research/results/gate0.5_failure_taxonomy.json (gitignored).
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
GEN_CHECK_TOKENS = 12

# Collapse thresholds
PPL_RATIO_COLLAPSE = 5.0
PPL_ABS_COLLAPSE = 200.0

# Sweep grid
REL_MAGNITUDES = [0.01, 0.05, 0.1, 0.5, 1.0]   # x tensor std
ABS_ANCHORS = [0.005, 0.05, 0.5]                # absolute deltas (Gate 1 bound range)
SCALAR_COUNT = 512                              # fixed edit budget for main sweep

REGIONS = {"early": (0, 10), "mid": (11, 21), "late": (22, 31)}

# Tensor-family classifier: param-name substring -> family bucket.
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


def classify(name: str) -> str:
    for sub, fam in FAMILY_RULES:
        if sub in name:
            return fam
    return "other"


def layer_idx(name: str):
    # "model.layers.<i>.rest" -> i, else None
    parts = name.split(".")
    if "layers" in parts:
        i = parts.index("layers")
        if i + 1 < len(parts):
            return int(parts[i + 1])
    return None


def region_of(idx):
    if idx is None:
        return "global"
    for r, (lo, hi) in REGIONS.items():
        if lo <= idx <= hi:
            return r
    return "other"


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

    # --- Build the family index -------------------------------------------
    families: dict[str, dict[str, list[str]]] = {}
    for n, p in params.items():
        fam = classify(n)
        li = layer_idx(n)
        reg = region_of(li)
        families.setdefault(fam, {}).setdefault(reg, []).append(n)

    report = {
        "gate": "gate_0_5_phase_a_failure_surface",
        "model": MODEL_ID,
        "seed": SEED,
        "git_sha": _git_sha(),
        "scalar_count": SCALAR_COUNT,
        "rel_magnitudes": REL_MAGNITUDES,
        "abs_anchors": ABS_ANCHORS,
        "collapse_criteria": {
            "nan_inf": True,
            "ppl_ratio": PPL_RATIO_COLLAPSE,
            "ppl_abs": PPL_ABS_COLLAPSE,
            "degenerate_check_tokens": GEN_CHECK_TOKENS,
        },
        "families_present": {f: sorted(regs) for f, regs in families.items()},
        "probes": [],
    }

    # --- Probe helpers -----------------------------------------------------
    probe_ids = tok(PROBE_TEXT, return_tensors="pt").input_ids.to("cuda")

    @torch.no_grad()
    def probe_metrics():
        """Forward the probe text; return (ppl, nan_inf, logits_ok)."""
        out = model(probe_ids)
        logits = out.logits.float()
        nan_inf = bool(
            torch.isnan(logits).any().item() or torch.isinf(logits).any().item()
        )
        # next-token cross-entropy -> perplexity
        shift_logits = logits[:, :-1, :]
        shift_labels = probe_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="mean",
        )
        ppl = float(torch.exp(loss).item()) if torch.isfinite(loss) else float("inf")
        return ppl, nan_inf

    @torch.no_grad()
    def degenerate():
        """Short greedy decode; True if output collapses to <=1 distinct token."""
        ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda")
        out = model.generate(ids, max_new_tokens=GEN_CHECK_TOKENS, do_sample=False)
        new = out[0, ids.shape[1]:]
        toks = new.tolist()
        if not toks:
            return True
        return len(set(toks)) <= 1

    # baseline (must be stable across every probe -> sanity check on rollback)
    base_ppl, base_nan = probe_metrics()
    report["baseline"] = {"probe_ppl": round(base_ppl, 3), "nan_inf": base_nan}

    def apply_edit(p, idx, delta):
        p.view(-1)[idx] += delta

    def run_probe(fam, reg, pname, magnitude, delta, mode):
        p = params[pname]
        n = p.numel()
        s = min(SCALAR_COUNT, n)
        g = torch.Generator(device="cpu").manual_seed(SEED)
        idx = torch.randperm(n, generator=g)[:s].to(p.device)
        orig = p.detach().clone()

        # capture baseline degeneracy once (cheap reuse)
        base_deg = None
        with torch.no_grad():
            apply_edit(p, idx, delta)
        torch.cuda.synchronize()

        ppl, nan_inf = probe_metrics()
        deg = degenerate()
        ppl_ratio = (ppl / base_ppl) if math.isfinite(ppl) and base_ppl > 0 else float("inf")
        collapse = nan_inf or deg or (ppl_ratio > PPL_RATIO_COLLAPSE) or (ppl > PPL_ABS_COLLAPSE)
        mech = []
        if nan_inf:
            mech.append("nan_inf")
        if deg:
            mech.append("degenerate")
        if ppl_ratio > PPL_RATIO_COLLAPSE or ppl > PPL_ABS_COLLAPSE:
            mech.append("ppl_explosion")

        # rollback
        with torch.no_grad():
            p.copy_(orig)
        torch.cuda.synchronize()
        restored = bool(torch.equal(p.detach(), orig))
        # sanity: ppl back to baseline
        rb_ppl, _ = probe_metrics()

        li = layer_idx(pname)
        rec = {
            "family": fam, "region": reg, "layer": li, "param": pname,
            "mode": mode, "magnitude": magnitude, "delta": round(float(delta), 6),
            "scalars": s, "tensor_std": round(float(p.std().item()), 6),
            "ppl_before": round(base_ppl, 3), "ppl_after": round(ppl, 3),
            "ppl_ratio": round(ppl_ratio, 3) if math.isfinite(ppl_ratio) else "inf",
            "nan_inf": nan_inf, "degenerate": deg,
            "collapsed": collapse, "mechanisms": mech,
            "restored_bit_exact": restored, "rollback_ppl": round(rb_ppl, 3),
        }
        report["probes"].append(rec)
        tag = "COLLAPSE" if collapse else "ok     "
        print(f"[{tag}] {fam:14s} {reg:5s} L{li} {mode} mag={magnitude} "
              f"ppl {base_ppl:.1f}->{ppl:.1f} nan={nan_inf} deg={deg}")
        return rec

    # --- Relative-magnitude sweep ------------------------------------------
    # one representative param per (family, region)
    for fam, regs in sorted(families.items()):
        for reg, names in sorted(regs.items()):
            # pick the param with the most elements in the region (canonical rep)
            pname = max(names, key=lambda n: params[n].numel())
            for mag in REL_MAGNITUDES:
                p = params[pname]
                delta = mag * float(p.std().item())
                run_probe(fam, reg, pname, mag, delta, mode="rel")

    # --- Absolute-magnitude anchors on the roadmap's Gate-1 bound range ------
    # Test the concrete bounds (0.005 / 0.05 / 0.5) on every family x region so
    # the constrained surface's absolute limits are evidence-based.
    for fam, regs in sorted(families.items()):
        for reg, names in sorted(regs.items()):
            pname = max(names, key=lambda n: params[n].numel())
            for mag in ABS_ANCHORS:
                run_probe(fam, reg, pname, mag, float(mag), mode="abs")

    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    out = RESULTS / "gate0.5_failure_taxonomy.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[failure_diagnostic] {len(report['probes'])} probes -> {out}")
    _summarize(report)
    return 0


def _summarize(report):
    from collections import defaultdict
    table = defaultdict(lambda: {"probes": 0, "collapses": 0})
    for pr in report["probes"]:
        t = table[pr["family"]]
        t["probes"] += 1
        t["collapses"] += int(pr["collapsed"])
    print("\n=== Collapse rate by family ===")
    for fam, t in sorted(table.items()):
        print(f"  {fam:16s} {t['collapses']}/{t['probes']} collapsed")


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
