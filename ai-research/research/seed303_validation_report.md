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

---

# V2 — Corrected methodology (2026-11-24, reviewer v2 feedback)

The v1 report had four methodological issues identified by the reviewer:

1. **8 domains, 1 passage each** — too small for a generalization claim.
   **Fixed:** expanded to 3 passages/domain (24 sealed holdout). Original 8
   marked `DIAGNOSTIC` (previously examined, not sealed).
2. **Arithmetic mean of passage perplexities** — not a proper aggregate.
   **Fixed:** token-weighted aggregate (sum NLL / sum tokens, then exp).
3. **Restore verified by perplexity within 0.01** — indirect.
   **Fixed:** `torch.equal()` on every restored tensor.
4. **Controls not matched** — seed-303 has 9 tensors (8 proj + 1 norm);
   v1 controls selected 8, missing the norm and unmatched by family/count.
   **Fixed:** controls now match 8 proj ×0.85 + 1 norm ×0.95.
   **Additional fix:** Control A's L0 projection pool had exactly 8 tensors
   (= seed-303's target set), making it degenerate. Fixed to draw from all
   12 L0 non-norm params (including dt_bias, A_log, conv1d).

## V2 results — token-weighted aggregate perplexity

| Condition | Phase B probe | Sealed agg (24 passages) | Diagnostic agg (8) |
|---|---|---|---|
| **Baseline (unmodified)** | 7.460 | **4.780** | 3.536 |
| **Seed-303 (exact replay)** | **6.983** | **4.834** (+1.1%) | 3.601 (+1.8%) |
| Control A: random L0 (5 seeds) | ~7.30 avg | ~4.862 avg (+1.7%) | — |
| Control B: non-L0 layer (5 seeds) | ~7.49 avg | ~4.825 avg (+0.9%) | — |
| Control C: all 32 layers | 9.495 | 6.937 (+45%) | — |

All 9 seed-303 tensors verified bit-exact by `torch.equal()`.

## V2 findings (refined from v1)

1. **Negative result confirmed with rigorous methodology.** Seed-303
   worsens token-weighted sealed perplexity by +1.1% (24 passages, 3 per
   domain). The v1 result (+2.1% arithmetic mean) was directionally
   correct; the corrected metric shows slightly less degradation but the
   same conclusion: the edit does not generalize.

2. **Per-passage: 17/24 sealed passages worsened, 7 improved slightly.**
   The improvements were small (largest: -0.584 on sealed_narrative_2).
   The degradations were larger (worst: +0.409 on sealed_conversation_3).
   No domain uniformly improved.

3. **Random L0 attenuation does NOT reproduce the probe improvement.**
   Control A (random 8 from 12 L0 non-norm params + 1 L0 norm, 5 seeds):
   probe mean ~7.30 vs seed-303's 6.983. The specific PROJECTION-TYPE
   selection matters for the probe — attenuating dt_bias/A_log/conv1d
   instead of projections does not produce the probe benefit.

4. **ALL L0 attenuation worsens held-out text.** Control A sealed mean
   4.862 (+1.7%), seed-303 sealed 4.834 (+1.1%). L0 attenuation is
   generally harmful to general text regardless of which L0 tensors
   are selected.

5. **Non-L0 attenuation is slightly LESS harmful.** Control B sealed
   mean 4.825 (+0.9%). This confirms Phase A's finding that L0 is the
   most fragile region — attenuating it hurts general text more than
   attenuating mid/late layers.

6. **Global attenuation is catastrophic.** Control C: probe 9.495,
   sealed 6.937 (+45%). Attenuating all layers destroys the model.

## V2 scientific conclusion

The seed-303 edit is a **probe-specific optimization that does not
generalize**. The specific projection-type selection is meaningful for
the probe (not reproduced by random L0 attenuation), but L0 attenuation
in general is harmful to held-out text. The negative result is now
established with:
- 24 sealed holdout passages (3 per domain, not previously examined)
- Token-weighted aggregate perplexity (not arithmetic mean)
- `torch.equal()` bit-exact restore verification
- Matched controls (9 tensors, 8 proj ×0.85 + 1 norm ×0.95)
- Non-degenerate random L0 pool (12 params, not 8)

**The original 8 diagnostic passages are now marked DIAGNOSTIC and should
NOT be reused as a sealed holdout for future experiments.** The 24 sealed
passages in this script are also now "examined" — future confirmation runs
must use new passages.
