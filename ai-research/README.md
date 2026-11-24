# ai-research — JARVIS Recursive Intelligence Research

Isolated ML research environment for the JARVIS RIP v1.2 protocol
(`Horizon/ROADMAP_v1.2.md` in the JarvisProject repo). Separate from
Sims2026's hardware-simulation `.venv` — do not install GPU libraries
into the existing environment.

## Structure

```
ai-research/
  .venv/            # ignored; PyTorch 2.8.0+cu128, transformers 5.24.0
  .venv-opt/        # ignored; same base + triton-windows + fla-core (FLA path)
  locks/            # frozen package manifests
  research/         # ROADMAP, preregistration, literature, reviewer log
  schemas/          # proposal/observation/checkpoint JSON schemas
  src/jarvis_research/  # research code (preflight, model_adapter, etc.)
  tests/            # unit tests (fake tensors, mock models)
  configs/          # experiment configs
  data/             # ignored; practice only, local
  results/          # ignored; gate0_hardware.json etc.
```

## Gate 0 results (2026-11-23)

**Status: GO** — all critical checks passed.

| Check | Result |
|---|---|
| CUDA available | True (CUDA 12.8, RTX A4500, 21.46 GB VRAM) |
| BF16 supported | True |
| Model load | 10.4s, 8.41 GB VRAM (12 GB free) |
| Weight NaN/Inf | None |
| Baseline repeatability (5 runs, greedy) | All identical: True |
| Throughput (fallback kernels) | ~12 tok/s at all prompt lengths |
| BF16 edit/rollback bit-exact | True (dt_bias, 32 elements, delta 0.01) |
| Estimated 100 candidates | ~1.17h |
| Estimated 500 candidates | ~5.87h |

> **Caveat (post-Phase B review):** These estimates are an initial throughput
> approximation based on short generated answers and a 16-task evaluation
> assumption. They do **not** account for 2048-token self-edit proposals,
> repeated perplexity evaluations, long stats-table prompts, or diagnostic
> overhead. Actual proposer-step wall-clock in Phase B was ~50–90s/step
> (proposal decode + eval), not the preflight's per-candidate figure. Do not
> use these numbers to freeze a Gate 1 compute budget — instrument real
> GPU-seconds per step first.

**Fallback kernels:** flash-linear-attention and causal_conv1d are not
installed (native Windows / Triton limitation). The model runs correctly
with reference PyTorch implementations but slower. Gate 0 measured this
rather than assuming optimized kernels.

**Full report:** `results/gate0_hardware.json` (gitignored, local-only).

## Gate 0.1 results — kernel A/B (2026-11-23)

A/B compared the pure-PyTorch fallback (`.venv`) against a
FLA-accelerated path (`.venv-opt`) on identical weights, prompts,
greedy seeds.

**Working native-Windows accelerated combination (verified):**
`torch 2.8.0+cu128` + `transformers 5.24.0` + `triton-windows 3.9.0`
+ `fla-core 0.6.0`, Python 3.11, RTX A4500 (CC 8.6). triton-windows
JIT-compiles kernels on this GPU. This resolves the roadmap's
"do not assume native Windows support" concern with a concrete,
tested combination.

| Metric | Fallback | Accelerated (FLA) |
|---|---|---|
| fla delta-rule engaged | no | yes (chunk ×432, recurrent ×25704) |
| determinism (5-run identical) | true | true |
| edit/rollback bit-exact | true | true |
| recurrent-state isolation | true | true |
| peak VRAM | 8.296 GB | 8.288 GB |
| tok/s — 128 / 512 / 1024 | 12.23 / 11.95 / 11.36 | 14.13 / 14.06 / 13.82 |

**Speedup ~1.18x.** Below the 1.5x bar, but adopted anyway: the
speedup carries *zero numerical cost* (all correctness invariants
hold) and `.venv-opt` is already built — it is free throughput, not
a fast-but-divergent kernel.

**Decision: FLA is the measured default for Gate 0.5+.** The fallback
`.venv` stays frozen as a reference path for cross-checks.

**`causal_conv1d` — deferred.** Ships source-only (no wheels); needs
`nvcc` + MSVC `cl.exe`, neither installed. It runs every decode step
on all 24 DeltaNet layers, so it's the real residual bottleneck and
the lever worth ~2-3x if pursued. Deferred: requires installing CUDA
toolkit + VS Build Tools (~6-8 GB, invasive). Revisit if a study
proves throughput-bound.

**Full report:** `results/kernel_ab_{fallback,accelerated}.json`
(gitignored, local-only). Script: `src/jarvis_research/kernel_ab.py`.

## Gate 0.5 Phase A results — failure surface (2026-11-23)

Mapped where Qwen3.5-4B collapses under weight edits. **Key finding:
the failure axis is coherence × position, not magnitude.**

- **Sparse scalar edits never collapse** (0/136 probes): ≤512 scalars,
  any family/region, up to 1.0×std / 0.5 absolute — all safe.
- **Coherent/dense edits collapse** at family+layer-specific thresholds.
- **Layer 0 is uniquely lethal** — a coherent +0.013 shift to
  `linear_attn.conv1d`/`out_proj` explodes perplexity (7.5 → 1500+).
  Fragility decays steeply with depth; mid/late layers robust.
- **`dt_bias`/`A_log` (recurrence) are the MOST robust family** — never
  collapsed, even under full zeroing or 128-step accumulation.
- **Collapse = perplexity explosion**, not NaN — the failure is a
  quality collapse; needs a ppl/degeneracy guard, not just NaN checks.

**Derived Gate 1 surface:** sparse scalar additive edits only, ≤512
scalars, |delta| ≤0.5 abs/×std, layers 11–31 only, whitelist
{recurrence, mlp, deltanet_proj, attention}, forbid embedding/lm_head/
norms. See `research/gate0_5_taxonomy.md` + `schemas/proposal.schema.json`.

**Tension to note:** the provably-safe surface is likely low-leverage;
leverage lives where it's fragile (early/coherent). Gate 1 tests whether
the constraint lands "safe null" (outcome b).

Scripts: `failure_diagnostic.py` (sparse sweep), `failure_boundary.py`
(escalation). Artifacts: `results/gate0.5_*.json` (gitignored).

## Gate 0.5 Phase B results — self-edit reconstruction (2026-11-23)

Reconstructed the paper's unconstrained protocol: the 4B model proposes and
receives arbitrary edits to its own weights, cumulative, 3 seeds × 24 steps —
run twice (768-token and 2048-token proposal budgets). **The paper's "4B broke
itself" was NOT reproduced in either version: all six runs survived, 123
unconstrained edits applied, zero collapses.**

- **100% of applied edits were attenuating `scale` ops** (×0.85–0.997 dense) —
  it never applied add/noise/zero/set, and deliberately chose conservatism.
- **It did not avoid early layers**: 58/87 v2 edits hit layers 0–10 (including
  ×8 dense scales on the whole L0 projection block, seed 303) — survival came
  from gentle op choice, not region avoidance.
- **Bottleneck is proposal competence, not safety**: v1 lost ~50% of steps to
  768-token truncation; v2 lost steps to *non-terminating deliberation*
  (5–6k chars of reasoning, no JSON) and 5× over-8 batch over-proposal.
- **More editing ≠ better**: seed 202's 59 edits drifted ppl UP to 7.62; only
  seed 303's early L0-batch win held (7.46→6.98) — flagged for controlled
  recheck in Gate 1, not treated as a result.
- **New mechanism evidence**: dense multiplicative *shrink* is tolerated on
  L0 where coherent additive/growth edits collapse — Phase A only probed
  scale upward (coverage gap noted).
- Paper's claim treated as **unreplicated — cause undetermined**; candidate
  divergences (scaffolding richness, step budget, op semantics, token budget,
  decode, checkpoint revision) enumerated in `research/gate0_5_phaseb_selfedit.md`.

Script: `proposer_unconstrained.py` (`--max-new`, `--out` for protocol
variants). Artifacts: `results/gate0.5_selfedit_reconstruction{,_2048}.json`
(gitignored, full raw traces).

## Seed-303 independent validation (2026-11-24, v2 corrected)

Reviewer finding: Phase B's "improvement" (7.46 → 6.98) was measured on the
same fixed probe used as the proposer's feedback signal. This test checks
whether the edit generalizes to held-out text.

**Result: the seed-303 edit does NOT generalize.** With corrected methodology
(24 sealed holdout passages, token-weighted aggregate perplexity, `torch.equal()`
restore verification, matched 9-tensor controls), the edit improved the
training probe by 6.4% but worsened token-weighted sealed perplexity by 1.1%.

| Condition | Phase B probe | Sealed agg (24 passages) |
|---|---|---|
| Baseline (unmodified) | 7.460 | 4.780 |
| Seed-303 (exact replay) | **6.983** | 4.834 (+1.1%) |
| Control A: random L0 (5 seeds) | ~7.30 avg | ~4.862 avg (+1.7%) |
| Control B: non-L0 layer (5 seeds) | ~7.49 avg | ~4.825 avg (+0.9%) |
| Control C: all 32 layers | 9.495 | 6.937 (+45%) |

Key findings: (1) random L0 attenuation does NOT reproduce the probe
improvement — specific projection-type selection matters for the probe;
(2) ALL L0 attenuation worsens held-out text; (3) non-L0 attenuation is
slightly less harmful (confirming L0 fragility from Phase A); (4) all
tensors verified bit-exact by `torch.equal()`.

**V2 corrections applied:** expanded to 3 passages/domain (24 sealed);
token-weighted aggregate perplexity (not arithmetic mean); `torch.equal()`
restore (not perplexity-within-0.01); matched 9-tensor controls (8 proj
×0.85 + 1 norm ×0.95); fixed Control A degeneracy (L0 projection pool = 8
= seed-303 target set).

Full report: `research/seed303_validation_report.md` (v1 + v2 sections).
Script: `seed303_validate.py`. Artifacts: `results/seed303_validation{,_v2}.json`.

**Implication for Gate 1:** evaluation MUST use held-out text (not the
feedback probe); matched random controls are essential; original 8
diagnostic passages are now "examined" and must not be reused as sealed.

## Attenuation directional sweep (2026-11-24)

Reviewer finding 2: Phase A tested scale only upward (growth). Phase B
discovered downward scaling (attenuation) survives on L0. This sweep maps
the attenuation collapse boundary.

**594 probes, 11 collapses.** Key findings:

- **L0 block (8 projections) has a sweet spot at ×0.70** — probe ppl
  drops from 7.46 to 6.66 (11% "improvement" on the probe, but see
  seed-303 validation above — this does NOT generalize).
- **Directional asymmetry confirmed:** coherent additive to L0 out_proj
  collapses at +0.25×std (Phase A); attenuation collapses at ×0.10.
  The safe surface for downward is much wider than for upward.
- **L0 mlp projections survive complete zeroing** (×0.00 → 7.43). L0
  DeltaNet in_proj_a/b also survive. But out_proj, in_proj_qkv, and
  in_proj_z collapse under extreme attenuation.
- **Multi-layer attenuation narrows the safe surface:** 1 layer survives
  to ×0.00; 2-16 layers collapse at ×0.50; all 32 layers collapse at ×0.70.

Script: `attenuation_directional.py`.
Artifact: `results/attenuation_directional.json` (gitignored).

## Running the preflight

```powershell
PYTHONPATH=ai-research/src ai-research\.venv\Scripts\python.exe ai-research\src\jarvis_research\preflight.py
```

## Running the kernel A/B

Two isolated envs; the only variable is the interpreter.

```powershell
# A — pure-PyTorch fallback (committed baseline path)
ai-research\.venv\Scripts\python.exe ai-research\src\jarvis_research\kernel_ab.py --mode fallback
# B — FLA-accelerated (Gate 0.5+ default)
ai-research\.venv-opt\Scripts\python.exe ai-research\src\jarvis_research\kernel_ab.py --mode accelerated
```

Each writes `ai-research/results/kernel_ab_<mode>.json` (gitignored).
