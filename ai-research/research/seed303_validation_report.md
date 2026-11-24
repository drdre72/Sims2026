# Seed-303 Independent Validation Report

**Date:** 2026-11-24 · **Model:** Qwen/Qwen3.5-4B (BF16, FLA path) · **GPU:** RTX A4500
**Trigger:** Reviewer finding 1 — Phase B's "improvement" (7.46 → 6.98) was
measured on the same fixed probe used as the proposer's feedback signal.
This script tests whether the edit generalizes.

## Protocol

1. Load a fresh Qwen3.5-4B checkpoint (no prior edits).
2. Evaluate baseline perplexity on 8 held-out text domains (science, code,
   narrative, math, history, technical, conversation, factual) + the
   original Phase B probe (for reproduction confirmation).
3. Apply the exact seed-303 edit set (8 dense ×0.85 scales on all L0
   projections + 1 dense ×0.95 on L0 input_layernorm).
4. Re-evaluate on ALL texts. Verify the Phase B probe reproduces 6.98.
5. Restore bit-exact, then run three matched controls:
   - **Control A:** random 8 L0 tensors ×0.85 (tensor-selection specificity)
   - **Control B:** same 8 projection types on a random non-L0 layer ×0.85
     (layer-position specificity)
   - **Control C:** ×0.85 on the same 8 types across ALL 32 layers
     (global attenuation effect)
6. Greedy-generate on 4 downstream prompts for coherence check.

Artifacts: `results/seed303_validation.json` (gitignored/local).
Script: `src/jarvis_research/seed303_validate.py`.

## Result 1 — Reproduction confirmed

The exact seed-303 edit set reproduces the Phase B v2 result bit-for-bit:

| | Phase B probe | 
|---|---|
| Baseline | 7.460 |
| Seed-303 replay | **6.983** (Phase B reported 6.983) |

The edit set and determinism are verified.

## Result 2 — The "improvement" does NOT generalize

| Text domain | Baseline | Seed-303 | Delta | Direction |
|---|---|---|---|---|
| **Phase B probe** | 7.460 | **6.983** | **-0.477** | **improved** |
| science_general | 2.782 | 2.840 | +0.058 | worse |
| code_python | 1.213 | 1.217 | +0.004 | flat |
| narrative | 9.483 | 9.682 | +0.199 | worse |
| math_reasoning | 2.170 | 2.226 | +0.057 | worse |
| history | 5.181 | 5.186 | +0.005 | flat |
| technical_explanation | 5.439 | 5.700 | +0.261 | worse |
| conversation | 4.324 | 4.278 | -0.046 | tiny improvement |
| factual_diverse | 4.282 | 4.472 | +0.190 | worse |
| **Mean held-out** | **4.359** | **4.450** | **+0.091** | **worse** |

The edit improved the training probe by 6.4% but **worsened held-out
perplexity by 2.1%**. Seven of eight domains degraded; only the training
probe and a marginal conversation improvement moved in the "better"
direction. This is **probe-specific overfitting**, not general
intelligence improvement.

## Result 3 — Random controls do not reproduce the probe effect

| Condition | Phase B probe (mean) | Mean held-out |
|---|---|---|
| Baseline (unmodified) | 7.460 | 4.359 |
| Seed-303 (exact replay) | **6.983** | 4.450 |
| Control A: random L0 ×0.85 (5 seeds) | 7.485 | 4.448 |
| Control B: non-L0 layer ×0.85 (5 seeds) | 7.480 | 4.394 |
| Control C: all 32 layers ×0.85 | 9.404 | 6.354 |

- **Control A** (random L0 tensor selection): did NOT improve the probe
  (7.485 vs 7.460 baseline). The specific tensor selection in seed-303
  matters for the probe — it is not a trivial "any L0 attenuation works"
  effect. But held-out degradation matches seed-303 (4.448 vs 4.450).
- **Control B** (non-L0 layer): also did NOT improve the probe (7.480).
  Held-out degradation is slightly less (4.394), suggesting L0-specific
  attenuation is slightly more harmful to general text.
- **Control C** (global): catastrophic — attenuating all 32 layers
  severely degrades both probe (9.404) and held-out (6.354).

## Result 4 — Model remains coherent

Greedy generation on downstream prompts is coherent after the seed-303 edit:
- "The largest planet in our solar system is" → "Jupiter."
- Fibonacci completion → correct recursive formula
- Story prompt → coherent narrative
- "Chemical symbol for gold?" → "Au"

The edit does not break basic generation capability — it just makes the
model slightly worse at predicting diverse text.

## Scientific conclusion

**The seed-303 "improvement" was evaluation-signal overfitting, not a
general capability gain.** The single-probe feedback loop (reviewer finding
1) allowed the model to adapt to the evaluation metric itself. This would
have been a false C1 claim if taken at face value.

Implications:
- Any Gate 1 protocol MUST use held-out evaluation, not the feedback probe,
  to score edits.
- Matched random controls are essential — without them, we cannot
  distinguish intelligent selection from trivial shrinkage effects.
- The fact that random L0 attenuation (Control A) did NOT reproduce the
  probe improvement means the specific tensor selection was meaningful for
  the probe — but in a direction that does not generalize.

## What this does NOT establish

- It does not prove that attenuation edits can never improve general
  capability — only that THIS specific edit did not.
- It does not test whether a search over many attenuation candidates might
  find one that generalizes — that is the Gate 1 question.
- It does not address the proposal-interface bottleneck (reviewer finding 5).
