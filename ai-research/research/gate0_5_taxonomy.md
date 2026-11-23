# Gate 0.5 — Failure-mode taxonomy (Phase A)

**Date:** 2026-11-23 · **Model:** Qwen/Qwen3.5-4B (BF16, FLA path) · **GPU:** RTX A4500
**Status:** Phase A complete. Phase B (self-edit reconstruction) not run this round.

Artifacts: `results/gate0.5_failure_taxonomy.json` (136 probes, sparse-scalar
sweep) and `results/gate0.5_failure_boundary.json` (384 probes, escalation
sweep). Both gitignored/local.

## What the source paper reported (to reproduce)

4B and 9B "broke themselves" under an unconstrained self-editing protocol
(proposer sees weights, proposes any edit to any tensor at any magnitude);
27B succeeded. The paper gives minimal failure analysis — no per-edit
diagnostics, no tensor-family breakdown, no magnitude/layer distribution.

## Headline finding — the failure axis is coherence × position, not magnitude

| Probe class | Result |
|---|---|
| **Sparse scalar edits** (≤512 scalars, any family/region, rel ≤1.0×std or abs ≤0.5) | **0 / 136 collapsed** |
| **Coherent/dense edits** (all-elements same-direction add/scale/noise/zero) | collapse at family-specific thresholds |

A scattered 512-scalar perturbation is ~0.008% of a 6.5M-element projection —
negligible L2. The paper's collapse cannot come from sparse scalar edits; it
requires **coherent** perturbation (a directed change to a tensor) **at the
input boundary**. The same total L2 that is harmless when scattered is lethal
when coherent.

## Fragility ranking (collapse threshold, coherent additive `param += k·std`)

| Family / param | Lowest collapse `k` | Region gradient | Robustness |
|---|---|---|---|
| `linear_attn.conv1d` (DeltaNet conv) | **0.25** | L0 lethal → L1 k≥2 → L12 k≥4 → L22+ never | most fragile *at early layers* |
| `linear_attn.out_proj` | **0.25** | same gradient | fragile at early layers |
| `embed_tokens` / `lm_head` | 1.0 (add) / 2.0 (scale) | global input boundary | fragile |
| `*.norm` / `*.layernorm` | scale 4.0; zero always | — | moderate (only scale/zero ops) |
| `self_attn.q_proj` (early) | 8.0 | early | robust |
| `mlp.*_proj` | 4.0–8.0 | L0 / L22 | robust |
| **`linear_attn.dt_bias` / `A_log` (recurrence)** | **never** | all layers | **most robust — survives even `zero` and 128-step accumulation** |

## Depth gradient (the load-bearing finding)

Coherent additive perturbation of `conv1d`/`out_proj`, collapse ppl (>5× base 11.4):

| Layer | k=0.25 | k=0.5 | k=1.0 | k=2.0 | k=4.0 |
|---|---|---|---|---|---|
| **L0** conv1d | 1315* | 151083* | 258597* | 92369* | 133289* |
| **L0** out_proj | 832* | 51686* | 30979* | 4560* | 6428* |
| L1 conv1d | ok | ok | ok | 128* | 309* |
| L12 out_proj | ok | ok | ok | ok | 63* |
| L22, L28 | all safe | | | | |

**Layer 0 is uniquely catastrophic** — a coherent +0.013 shift destroys the
model. Fragility decays steeply with depth. Errors at the input propagate
through all 31 downstream layers; mid/late edits are absorbed.

## Collapse mechanism — quality collapse, not numeric blowup

- Dominant mechanism: **perplexity explosion** (predictions go near-random),
  NOT NaN/Inf. Only `embed_tokens → 0` produced true `degenerate`
  (ppl = 248320 = vocab size = uniform-random output).
- Implication: a constrained surface needs a **quality guard** (perplexity /
  eval check) — a NaN-only guard would miss the real failure mode.

## Derived constrained surface for Gate 1 (evidence-based)

| Dimension | Evidence | Constraint |
|---|---|---|
| **Operation** | coherent dense edits lethal; sparse scalar safe | allow **sparse scalar additive edits only**; forbid scale/zero/set ops |
| **Edit budget** | 512-scalar edits never collapse | cap at **≤512 scalars** per proposal |
| **Magnitude** | sparse ≤1.0×std / ≤0.5 abs all safe | \|delta\| ≤ **0.5 abs** and ≤ **0.5×std** (whichever binds) |
| **Layer scope** | early layers lethal | **forbid layers 0–10**; allow mid/late 11–31 |
| **Family** | recurrence & mid/late projections safe; embedding/norm lethal | allow `dt_bias`, `A_log`, `mlp`, `linear_attn.*`, `self_attn.*` at allowed layers; **forbid embed_tokens, lm_head, *.norm** |
| **Guard** | collapse = ppl explosion | post-edit ppl + degenerate check, auto-rollback |

## The safe/effective tension (the honest scientific finding)

The provably-safe surface (sparse scalar, mid/late layers, recurrence params)
is also where edits are likely **low-leverage** — it produced zero collapse
*and* probably little signal. The leverage lives exactly where it's fragile:
early layers, coherent ops, projection tensors. **The safe edit surface and
the effective edit surface may be disjoint.** This is the real risk Gate 1
tests — the constraint may land in "safe null" territory (roadmap outcome b).
That outcome is informative, not a failure.

## What "4B broke itself" most plausibly was

Given the evidence, the paper's collapse was almost certainly a **coherent
edit to an early-layer / input-boundary tensor** (or an accumulated coherent
drift), not a scattered scalar edit — a self-editor optimizing for a goal can
select exactly the lethal direction the surface sweep exposes. This is the
hypothesis Phase B (self-edit reconstruction) would confirm.

## Limitations

- One representative param per (family, region) cell — not exhaustive.
- Single seed; FLA kernel path (numerics proven identical to fallback, G0.1).
- Collapse proxy = perplexity + degenerate check, not a full task eval.
- Phase B (self-edit loop, proposer scaffolding autopsy, batch behavior) not
  yet run — scaffolding choice "stats + prior feedback" recorded for later.
