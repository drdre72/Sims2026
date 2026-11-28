# HZN-FLOW-01 (PARKED SKELETON) — Transport dynamics over typed working state

**Status: PARKED — not an active experiment. No GPU commitment, no implementation.**
**Prerequisite:** the typed-state consumption question opened by HZN-EPI-05 (neural readout of `unknown`/`ambiguous`/`stale`/invalidated state) must be resolved or explicitly deferred first.
**Origin:** external research-branch proposal (Navier–Stokes cognitive analogue), stripped to its honest operator skeleton on 2026-11-27. This document exists so the design is not lost; it is not a preregistration commitment.

---

## 1. What was stripped, and why

Starting point: incompressible Navier–Stokes as a proposed cognitive-attention analogue.

| term | physical role | cognitive verdict |
|---|---|---|
| ∂u/∂t | time evolution | KEEP — state evolves between inference cycles |
| (u·∇)u | nonlinear self-advection (fluid transporting its own momentum; source of turbulence) | DROP — attention has no momentum; nothing transports itself. Dropping this is dropping Navier–Stokes itself, and that is accepted |
| −∇p/ρ | pressure gradient | REDEFINED — in incompressible flow pressure is the Lagrange multiplier enforcing conservation (∇·u=0), not a thermodynamic variable. Its honest cognitive analogue is the attention-budget constraint, not "unresolved questions" |
| ν∇²u | viscosity/diffusion | KEEP — graph-Laplacian smoothing; resistance to abrupt focus shifts |
| f | external body forces (gravity is merely one member) | KEEP MINUS GRAVITY — stimuli/observations are the only external forcing in this setting; gravity was never part of the problem, so removing it costs nothing |
| ∇·u=0 | incompressibility (volume conservation) | KEEP AS BUDGET — total activation conserved/renormalized per cycle |

**Result of stripping:** nothing Navier–Stokes-specific survives. What remains is the operator skeleton:

```
evolution = transport + diffusion + source − decay,  subject to budget conservation
```

i.e. advection-diffusion-reaction on a graph. This is a *feature* of the analysis: the metaphor debt is now zero, every surviving term has an assigned measurable meaning, and anything that cannot be operationalized gets cut rather than rebranded.

## 2. Mathematical specification (graph form)

- **Nodes:** cognitive-bus typed objects (evidence, comparisons, decisions, epistemic states).
- **Edges:** dependency/provenance links (already exist in `cognitive_bus.py`), temporal succession, semantic association. Edge weights w_ij.
- **Activation** A_i(t): scalar per node. Budget: Σ_i A_i = B (renormalized each cycle).
- **Pressure potential** P_i: computed **mechanically from typed state only** — contradiction flags (CONSISTENT failures), `unknown`/`ambiguous`/`invalid`/`stale` statuses, unresolved decisions, invalidation age. **No neural judgment in the pressure field.** This is the design's central bet: EPI-05 produced exactly these mechanical pressure sources.
- **Transport flux:** F_ij ∝ (P_j − P_i) · w_ij — activation flows toward pressure differentials along existing edges.
- **Diffusion:** D · (L A)_i where L is the graph Laplacian.
- **Source/sink:** S_i from new stimuli/observations entering the bus; decay λA_i.
- **Update:** A_i(t+1) = A_i + Δt·(transport + D·diffusion + S − λA), then renormalize to budget B.
- **Readout:** top-K activated nodes → retrieval/context selection for the next inference cycle.

All parameters (D, λ, w, Δt, B, K) are fixed before any task evaluation.

## 3. Arms

| arm | allocator |
|---|---|
| A | **Dumb baseline: priority queue** — recency + fixed priority heuristic. This arm is the point of the experiment |
| B | diffusion only (no transport) |
| C | transport + diffusion |
| D | C + outcome feedback (resolution of a query reduces source pressure) |
| E | D + adaptive edge weights |

Same task battery, same compute budget, same readout budget K for all arms.

## 4. Metrics

- task accuracy on resolution-dependent tasks (epistemic battery reuse)
- compute cost per correct resolution (neural tokens + wall time)
- interruption recovery: correctness after mid-task state serialization/resume
- false-alarm interventions (system attends to something that needed nothing)
- equilibrium-vs-truth failures: cases where the field settles but the answer is wrong — must be counted, not hidden

## 5. Falsification criteria (binding if activated)

- If arms B–E do not beat arm A by a preregistered margin on accuracy-per-compute, **the branch closes and the priority queue is kept**. No parameter re-tuning after seeing task results.
- If pressure cannot be computed mechanically for a task class, that class is excluded — pressure must never come from the neural model.

## 6. Honesty constraints

- No Navier–Stokes branding in code, metrics, or logs. The lineage is spreading activation (Collins & Loftus 1975; ACT-R) and graph diffusion, and the writeup must say so.
- Equilibrium is not truth. A settled field means nothing about correctness.
- Negative results are preserved and committed like every other Horizon branch.

## 7. Sequencing

1. Finish typed-state consumption (EPI-05 follow-up / possible HZN-EPI-06).
2. CPU-only prototype of arms A and C on a small fixed graph; validate bookkeeping.
3. Only then decide whether this graduates to a real preregistration.

## 8. Open architectural fork — dissipative vs oscillatory dynamics (recorded 2026-11-27)

The stripped skeleton above is **dissipative**: diffusion damps. The heat equation smooths and settles; it cannot sustain oscillation. A separate hypothesis exists — continuous dynamics in which **binding is temporal synchrony rather than typed pointer** (binding-by-synchrony in neuroscience; Kuramoto-style coupled-oscillator synchronization in mathematics). Under that hypothesis, "structural bonding" emerges when subsystems resonate at matched frequencies, not when a runtime writes a reference.

This is a genuinely different mathematics (wave/oscillator dynamics, second-order or reactive coupling — not the diffusion skeleton) with genuinely different properties:

- oscillatory modes propagate and persist; diffusion dissipates and settles — they solve different problems and may both be needed (allocation vs binding);
- synchrony-binding is **opaque to inspection** — a resonance either holds or it doesn't, and there is no readable provenance chain. Typed-pointer binding is inspectable, which is precisely what EPI-05 vindicated. Choosing oscillation trades away assayability — the touchstone property — unless a verification story is designed in from the start;
- therefore this fork is recorded, not adopted: any oscillatory variant must specify how its bindings are verified before it counts as a Horizon candidate.

Physical caution: resonance is real physics (efficient energy transfer at matched frequency), but molecular bonding is quantum-mechanical, not classical vibration; cymatics organizes existing matter into modes rather than creating bonds. The metaphor does not transfer as mechanism — only synchrony-as-binding does, and that transfer is a hypothesis to test, not an established fact.
