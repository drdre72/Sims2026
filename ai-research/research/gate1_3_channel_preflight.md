# HZN-OC-01 Preflight — Measured Channel Costs

**Component status (HZN-IR-05 d2; HZN-IR-08 corrections applied):** symbolic encoder ✅, tokenizer profiling ✅, observation audit layer ✅ (mandatory provenance, edit-chain binding), comprehension harness C1–C4 ✅ (fail-closed gate, strict grading, campaign-realistic catalogs). 47/47 tests. This document reports measured token costs over the exact rendered surfaces (families + bounds legends + rows) with canonical field-parity verified on every level.

## 1. Measured costs, real tokenizer, real catalogs

192 rows arm A, 248 rows arm B. Parity = canonical record equality including effective magnitude intervals and count defaults:

| Level | Arm A tokens | /row | ratio | parity | Arm B tokens | /row | ratio | parity |
|---|---|---|---|---|---|---|---|---|
| linguistic (baseline) | 8,801 | 45.8 | 1.00 | ✅ | 8,877 | 35.8 | 1.00 | ✅ |
| s1 (spaces, familiar keys) | 7,811 | 40.7 | 0.888 | ✅ | 7,637 | 30.8 | 0.860 | ✅ |
| s2 (`O:/F:/AR:/LC:` symbolic) | 8,488 | 44.2 | 0.964 | ✅ | 8,457 | 34.1 | 0.953 | ✅ |
| **s3 (headered positional)** | **7,386** | **38.5** | **0.839** | ✅ | **7,156** | **28.9** | **0.806** | ✅ |

Sample rows (arm B): linguistic `0|L0|f0|n=81920|ar=1.00|lc=+0.02` → s3 `0|0|f0|81920|1.00|+0.02` under header `cols: id|layer|fam|numel|ar|lc`.

**Two measured findings:**
1. **S-II loses to S-I** — symbolic vocabulary is shorter in characters but BPE-fragments; character compression ≠ token compression (reviewer prediction confirmed).
2. **S-III (headered) beats S-I** — removing *repeated field names* (structural redundancy), not inventing vocabulary, is the best measured scheme: **−16.1% arm A, −19.4% arm B** at full field precision and complete semantics. This supports the reviewer's "eliminate unnecessary representation" direction.

## 2. Token accounting — consistent denominators (HZN-IR-08 d9)

All figures below are **per-row tokens** on the arm-B row `1|L0|f1|n=81920|ar=0.04|lc=+0.06` (35.8 tok/row mean, linguistic):

| Field group | Rendering | Tokens | % of row |
|---|---|---|---|
| id + layer + fam | `1\|L0\|f1` | ~6 | ~17% |
| numel | `\|n=81920` | ~6 | ~17% |
| obs columns (ar+lc) | `\|ar=0.04\|lc=+0.06` | ~15 | ~42% |
| separators/syntax | | remainder | ~24% |

So, **of a row**: observation columns ≈ 34–42%, structural identity fields ≈ 34–45%, syntax ≈ 20–25%. **Of the whole prompt**: catalog rows ≈ 8.8k of ~9.5k serialized tokens ≈ **93%** (legends + instructions + arm context + history ≈ 600 tok). The earlier inconsistent figure ("structural fields 85–90% of catalog") conflated row-share with prompt-share; corrected accounting above: within the catalog, obs columns ≈ 40%, structure ≈ 60%; the catalog itself is ≈ 93% of the prompt.

## 3. Consequences and reviewer rulings applied

- Precision reductions (numel→`23M`, mag→2sf, numel removal) are **information-ablation experiments, not channel candidates** — they change what the proposer knows, so they cannot run inside the information-matched epoch (HZN-IR-08 d8, resolving my earlier §3 questions 1–3).
- **Eligibility is against the L baseline**, not the previous best alternative: s3 qualifies on tokens (−16/−19%), parity (canonical equality), and must still pass C1–C4 before trajectories (HZN-IR-08 d7 fix — the earlier "must beat S1" clause would have excluded S1 itself).
- Separator/field-name optimization was explicitly sanctioned with complete semantics retained — that is exactly what s3 is.
- Realistic ceiling at full precision: **≈ −20% catalog ≈ −18% of the serialized prompt.** The 10–30× figure remains a documented negative result.
- The addressable-store arm and KV-cache/static-prefix reuse (with post-edit invalidation) become the more promising structural directions — filed as the deferred candidates; the catalog's static-vs-dynamic split (identity/bounds vs ar/lc/freshness) is the formal hook.

## 4. Operational comprehension gate — measured result (HZN-IR-08 d6)

Ran the C1–C4 pretest against the real model, campaign-identical prompting (chat template, thinking disabled, greedy): 3 encodings × 3 independently generated campaign-realistic catalogs (248 rows, near-zero contrast mass, rounded ties, reserved-unique + repeated family/layer pairs). `results/gate1_3_comprehension.json`.

| Encoding | C1 decode (≥95%) | C2 compare (≥90%) | C3 compose (≥90%) | C4 ground (100%, ≥60) | Verdict |
|---|---|---|---|---|---|
| linguistic | 48/48 (100%) | 43/48 (90%) | 15/24 (62%) | 35/84 (42%) | **FAIL** |
| s1 | 48/48 (100%) | 43/48 (90%) | 12/24 (50%) | 38/84 (45%) | **FAIL** |
| s3 | 48/48 (100%) | 29/48 (60%) | 0/24 (0%) | 33/84 (39%) | **FAIL** |

**Reading:**

1. **Decoding is never the problem** — 100% on every encoding, including positional s3.
2. **Grounding is the bottleneck, encoding-invariant** — C4 ≈ 40–45% on ALL levels including the linguistic baseline. The failures are clean (valid integer or "ambiguous" — just wrong), with ~half of true-ambiguous probes answered with a confident id instead of refusal. Under the reviewer's own logic, an observation the model decodes but addresses to the wrong parameter is an intervention on the wrong component — and this channel shows that failure ~half the time at 248-row scale, *independent of representation*.
3. **s3 pays a real comprehension tax** — C2 drops to 60% and C3 to 0% (most responses empty `[]`). Positional rows shown without their `cols:` header make field-identification ambiguous in isolated-row probes; whether that under-tests s3 relative to the full catalog context is a fairness question flagged for the reviewer — but the C4 result holds even where decoding is perfect.
4. **The pretest is stricter than the campaign's actual grounding operation** — C4 probes test *find-the-row-matching-a-description*; the campaign's flow is *select-a-row-and-emit-its-param_id* (copy, not lookup). The strict form is what the spec registered; both framings are reported.

**Implication:** this gate did its job — it separated decodability (solved) from operational grounding (not solved, at catalog scale, on any encoding). It suggests the binding constraint on observation-driven editing is not channel cost at all but **catalog-scale addressing** — which strengthens, again, the addressable-store direction: the model cannot reliably ground over 248 rows anyway, so retrieving fewer relevant rows loses little and may help. Alternatively a small-catalog arm (e.g., ~32 rows) would directly test whether C4 accuracy is scale-driven.

## 5. What remains before campaign authorization

- ~~Paired within-seed sensitivity analysis~~ — done (`gate1_3_sensitivity.json`): ε=δ=0.05 provisional; within-seed σ≈0.127 (CI [0.066, 0.795]); 80% correspondence power needs n=7 at Gate-1.2-magnitude effects; **non-inferiority at ε=0.05 is below the noise floor — margin decision returned to the reviewer.**
- ~~Budget accounting split~~ — done: `stage_seconds` (in-loop pool) vs `final_eval_seconds` (fixed allowance) vs `unattributed_wall_seconds`.
- ~~Provenance threading~~ — done: `collect_observations` emits the full `{base, edit-chain, context, config, epoch}` tuple; surfaced in the campaign report.
- ~~Comprehension operational run~~ — done (§4): **all encodings fail C3/C4 at campaign scale; grounding, not channel cost, is the measured bottleneck.**
- ε/δ finalization — reviewer's call informed by the sensitivity table.
- Final resource allocation + primary utility definition sign-off.
- Reviewer disposition on the C3/C4 finding before trajectories run.

## 5. Artifacts

- `results/gate1_3_channel_profile.json` — per-level profile + parity on the real catalogs
- `results/gate1_3_comprehension.json` — 9 cells × responses + scores (campaign prompting)
- `results/gate1_3_comprehension_rawprompt_invalid.json` — preserved method error: raw (non-chat-template) prompts produce 0% — the model continues the catalog rather than answering; demonstrates why the pretest must match the campaign's prompt surface
- `results/_profile_channels.py`, `results/_comprehension_run.py`, `results/_sensitivity_analysis.py`, `results/_grounding_isolation.py` — deterministic reproductions (committed via `-f`; `results/` is gitignored)
- `results/gate1_3_grounding.json` — 60 cells × probes + verdicts + tokens/latency/VRAM (HZN-G1.3-R)
- `src/jarvis_research/gate1_3_channel.py` (canonical parity, audit, s3), `gate1_3_comprehension.py` (fail-closed, strict grading, realistic catalogs, IR-09 F1–F7), `gate1_3_store.py` (deterministic read-only store), `tests/test_gate1_3_channel.py` (57 tests incl. the HZN-IR-08/09 regression sets)

## 7. HZN-G1.3-R grounding isolation — measured result (HZN-IR-09 §4)

Corrected harness (F1–F7 applied), linguistic encoding, nested catalogs 16/32/64/128/248 (seeded subsets), 2 seeds, inference only. `results/gate1_3_grounding.json`, runner `results/_grounding_isolation.py`, store `src/jarvis_research/gate1_3_store.py`.

**Unique-target descriptive lookup — the scale curve:**

| rows | R0R1 full catalog | R4 store-retrieved | R6 oracle |
|---|---|---|---|
| 16 | 18/20 | 20/20 | 20/20 |
| 32 | 18/20 | 20/20 | 20/20 |
| 64 | 17/20 | 19/20 | 20/20 |
| 128 | **9/20** | 18/20 | 20/20 |
| 248 | **9/20** | 18/20 | 16/20 |

**The grounding cliff is between 64 and 128 rows.** Deterministic family-filtered retrieval (~27–45 rows at 248) holds 90% everywhere and matches oracle within noise.

**All conditions, both seeds (unique+ambiguous+missing probes):**

| rows | R0R1 | R2 copy-id | R3 describe→resolve | R4 | R5 random | R6 |
|---|---|---|---|---|---|---|
| 16 | 23/29 | 16/16 | 5/16 | 27/29 | 28/29 | 28/29 |
| 32 | 26/32 | 16/16 | 5/16 | 27/32 | 31/32 | 28/32 |
| 64 | 23/36 | 16/16 | 3/16 | 27/36 | 34/36 | 28/36 |
| 128 | 12/36 | 16/16 | 3/16 | 26/36 | 30/36 | 27/36 |
| 248 | **9/36** | 16/16 | **0/16** | 25/36 | 24/36 | 22/36 |

**Failure taxonomy at 248 (R0R1):** 9 correct, 9 wrong-but-valid ids, 8 hallucinated ids on missing targets, 8 invalid picks on ambiguous targets, 2 refusals when present. The model answers a confident id ~always — ambiguity recognition is 0/8 at *every* size; missing recognition collapses 5/8→8/8→0/8 across the sweep.

**Attribution:**
- **R2 copy-id is 16/16 at every size** — the campaign's actual select-and-copy operation is NOT the bottleneck. Descriptive lookup and copying an existing identifier are different operations, as the reviewer predicted.
- **R3 describe→resolve fails at 248 (0/16) and is already imperfect at 16** — separating lookup from reasoning (store resolves the model's descriptor) doesn't rescue identification, because the model emits well-formed, resolvable, *wrong* descriptors. Failure is upstream of addressing: it picks the wrong semantic target. Notably `largest |lc|` is 0/30 at every size while `smallest ar` works at small scale — a likely signed-vs-absolute-value comprehension gap, not a lookup error.
- **R4 ≈ R6 ≈ R5 (with target present)** — retrieval quality adds nothing beyond context reduction itself. The store's value is bounding the context, which it does deterministically and with guaranteed-correct coverage of the probed (family, layer) space.

**Cost per probe at 248:** full catalog ≈8.5k input tokens / ~22s; store-retrieved subset ≈1.3k tokens / <1s — **6.4× fewer tokens, ~20× faster, and ~2× the unique-target accuracy.**

**Verdict on the IR-09 question:** grounding failure is dominated by *context scale* — it onsets sharply between 64 and 128 rows, while identifier binding (copy) is solved and semantic selection (R3) fails independently. Reliable internal addressing without weight modification is demonstrated: a deterministic store + narrow context restores ~90% grounding on the exact probes the full catalog fails. The residual ~10–30% subset-condition misses and the universal ambiguity-refusal failure are the open item for the success criterion's "no ambiguous or invalid address accepted" clause.
