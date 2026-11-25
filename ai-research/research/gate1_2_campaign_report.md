# Gate 1.2 Exploratory Campaign — Compute-Matched Results

**Evidence package for review (HZN-IR-06).** Exploratory designation — n=3 seeds per condition, small effects, not confirmatory. All numbers below are extracted from `results/gate1_2_campaign_compute.json` (gitignored; sha-pinned by manifest).

- **Implementation:** `1d234d3` on `research/gate-1-acceptance` (post HZN-IR-05 launch checks)
- **Manifest:** `--arms A B --conditions A0 A1 A2 B0 B1 --seeds 1 2 3 --budget-mode seconds --compute-budget-seconds 240 --max-steps 24 --data-dir gate1_eval_v4`
- **Runtime:** 8,930 s wall (~2.5 h), 30 candidates, incremental writes, `--resume`-capable
- **Integrity:** 0 collapses, 0 rollbacks needed beyond candidate cleanup, 0 containment refusals, 0 infrastructure failures, capability suite executed on all 30 candidates

## 1. Headline results

Selection delta = candidate selection ppl − baseline ppl (8.626). Positive is worse.

| Cond | Arm A deltas | mean | reg | Arm B deltas | mean | reg |
|---|---|---|---|---|---|---|
| A0 | +0.016, +0.004, +0.010 | **+0.010** | 3/3 | +0.306, +0.130, +0.202 | **+0.213** | **0/3** |
| A1 | −0.006, +0.008, +0.006 | +0.003 | 3/3 | −0.027, +0.064, −0.032 | **+0.002** | **3/3** |
| A2 | +0.003, +0.005, −0.011 | −0.001 | 3/3 | +0.175, +0.047, −0.072 | +0.050 | 2/3 |
| B0 | −0.005, +0.008, −0.003 | +0.000 | 3/3 | +0.086, +0.042, +0.077 | +0.068 | 2/3 |
| B1 | +0.014, −0.009, +0.014 | +0.007 | 3/3 | +0.123, −0.009, +0.075 | +0.063 | 3/3 |

**Arm A is a null field** — every condition within ±0.016, all regression-pass. **Arm B is where the comparison lives**: blind autonomous editing (A0) produced broad regression damage in all 3 seeds (mean +0.21 selection ppl); real observations (A1) kept all 3 seeds flat (+0.002) and clean.

**The ordinal pattern the experiment was designed to detect is present on arm B:** A1 > A2 > A0 on both selection delta and regression integrity. The placebo sits in between — the evidence channel is being *used* (§4.2), and correspondence determines whether that use lands on real contrast.

### Regression-domain transition ledger (±0.005 CE per domain, 24 domain-instances per condition)

| Cond | arm A (acq / ret / reg) | arm B (acq / ret / reg) |
|---|---|---|
| A0 | 1 / 21 / 2 | 2 / 2 / **20** |
| A1 | 1 / 22 / 1 | 7 / 10 / **7** |
| A2 | 1 / 23 / 0 | 5 / 3 / **16** |
| B0 | 0 / 24 / 0 | 7 / 4 / 13 |
| B1 | 0 / 23 / 1 | 9 / 9 / **6** |

The sensitive-threshold ledger sharpens the ordering: on arm B, **A1 ≈ B1 ≫ A2, B0 ≫ A0**. Real-observation autonomous editing and interpreter-guided adaptive search are the two healthy conditions; the placebo visibly damages domains the tolerance gate would miss.

A0's arm-B failures were *broad* — factual (×2), narrative (×3), technical, history, science — not a single-domain accident. A2's single hard failure was narrow (technical, ΔCE +0.054).

## 2. Reviewer §4 measurements

### 4.1 Parameter-selection distribution — does evidence change decisions?

Rebuilt the exact annotated catalogs post-hoc (deterministic pipeline, 192/248 rows, 263 groups — matches campaign). Picked params per candidate:

- **Pick overlap is low**: arm A J(A1,A2) = 0.30, arm B J(A1,A2) = 0.10; A0 overlaps either obs condition < 0.17. Same instructions, same catalog skeleton, largely different targets — the evidence *content* drives selection.
- **Family targeting**: A1 concentrates on `mlp.down_proj` (arm A 7/14 picks; arm B 5/16). A2 scatters into `linear_attn.out_proj`/`in_proj_a` on arm B — it chases shuffled signal into families where the true contrast doesn't live.

### 4.2 Real vs shuffled observation response — does correspondence matter?

Presented |lc| of picked vs unpicked params, and **true** |lc| of picks:

| | pres. \|lc\| picked | pres. \|lc\| unpicked | **true** \|lc\| picked |
|---|---|---|---|
| A1 (arm B) | 0.013–0.020 | 0.015 | **0.013–0.020** |
| A2 (arm B) | **0.034–0.045** | 0.015 | 0.010–0.022 |

The model reads the evidence channel in both conditions — A2 follows its (permuted) high-|lc| rows *more* strongly than A1 follows true ones (top-quartile share 4–6 of 6 vs 1–3 of 6). But only A1's picks land on parameters whose contrast is real. A2 chases phantom evidence to ordinary parameters — and on arm B that misdirection produced a regression failure and the intermediate ledger damage in §1.

This is the cleanest mechanistic finding of the campaign: **the channel is read, followed, and correspondence determines its value.**

### 4.3 Abstention frequency — does evidence support restraint?

| | applied | abstained | rejected |
|---|---|---|---|
| A-conditions | 5–6 (obs) / 14–16 (blind) per 7/17–18 steps | 0 | 0–3 |
| B-conditions | 6–19 per 22–24 steps | **10–18 of 24 steps** | 0 |

The B-conditions, given the unchanged checkpoint as a legal outcome (HZN-IR-05), abstain on **~50–75% of iterations** — the adaptive search's dominant behavior is declining to edit. Restraint is protective: B1's high abstention correlates with its clean ledger. A-conditions never abstained — the autonomous proposer always emits an edit.

### 4.4 Predicted vs actual improvement — calibration

Mean predicted `practice_delta_ce` vs measured, applied edits:

| Cond | mean predicted | mean actual | r |
|---|---|---|---|
| A0 (A / B) | −0.002 / −0.008 | +0.000 / **+0.001** | +0.01 / −0.01 |
| A1 (A / B) | −0.004 / −0.004 | −0.000 / +0.000 | +0.12 / −0.24 |
| A2 (A / B) | −0.006 / −0.016 | +0.000 / +0.000 | +0.08 / +0.31 |

**Every condition systematically over-predicts improvement by ~10–30×**; correlations are near zero. On arm B, A0 predicts −0.008 per edit while actually landing +0.001 — optimistic at the moment it was doing the most damage. The proposer cannot currently predict its own edit outcomes. Diagnostic for future work, not a blocking defect.

### 4.5 Improvement per compute second

No condition purchased selection improvement at any scale (9/30 candidates improved, all |Δ| < 0.03). The compute-matched comparison's discriminating output is entirely on the **damage-avoidance axis**: at equal 240 s, real observations cost ~7 proposal cycles (vs ~18 blind) but produced zero regression failures where blind produced three.

### 4.6 Observation staleness (first-step vs late-step)

No clean first-step advantage visible. Per-step practice deltas on arm B: A1 drifts mildly positive across the trajectory (steps 3–5: +0.0004…+0.0011), A2 more so (+0.0019…+0.0021). This is consistent with *cumulative damage* in misdirected conditions rather than stale evidence — A1's step-0 deltas are not better than its step-5. **Staleness hypothesis: unresolved; data does not isolate mechanism.**

### 4.7 The observation tax, quantified

Proposal-stage share of measured compute: **78–79% for A1/A2 vs 63–70% for A0/B-conditions**. Observation-bearing prompts (~9.4k tokens serialized, incl. history) cut achievable proposal cycles ~2.5× at fixed compute — the price of the evidence channel at this scale.

## 3. Budget integrity & caveats (honest report)

- **Overshoot**: all 30 candidates recorded compute_seconds_used 252–340 s vs 240 budget; mean overshoot ~55 s = final-eval tail (sel+reg+cap ≈ 48 s, charged post-loop) + the threshold-crossing step. Recorded per candidate, never hidden.
- **Cap-bound caveat**: 11/12 B-condition candidates hit `max_steps=24` before in-loop seconds reached 240 (their steps are cheap; in-loop totals ~215–245 s). B-conditions thus received ~5–10% *less* in-loop compute than A-conditions on cap-bound runs. Modest, recorded, and runs *against* the B-conditions — it cannot explain their clean outcomes, but it should be raised on the confirmatory design (`--max-steps` ≥ 30 or budget-proportional caps).
- **Capability suite fragility**: `retrieval` (n=1 task) and `structured` (n=2) flip on single outcomes. arm-B retrieval losses hit A0×3, **A1×2**, B0×0, A2×0 — noisy across conditions; do not read as a ranked signal at this n. A2's clean capability sheet despite one CE-regression failure is a reminder these are different instruments.
- **Capability acquisition observed**: arm-B B1 raised `structured` accuracy 0.5→1.0 on 2/3 seeds — the only repeated capability *gain* in the campaign.

## 4. Disposition per hypothesis

| | Status after campaign |
|---|---|
| H-OBS.1 (observations usable by proposer) | **Supported** — channel is read and followed (§4.2); pick distribution responds to content |
| H-OBS.2 (correspondence matters) | **Supported directionally** — A1 > A2 on selection delta, regression integrity, and domain ledger on arm B; n=3 |
| H-OBS.3 (advantage under compute-matching) | **Directional** — A1 protected selection+regression at equal seconds; no acquisition advantage observed for any condition |
| Calibration of self-prediction | **Refuted as implemented** — predictions over-optimistic ~10–30×, r ≈ 0 |

## 5. Assessment

The campaign produced a real, ordinal, internally-controlled result rather than noise: **on the arm where blind editing demonstrably damages the checkpoint, observation-grounded editing avoided that damage while placebo-grounded editing fell between.** The benefit manifests as *selectivity* (knowing where not to push), not acquisition — consistent with the observation that A1 applies few, targeted edits while A0 applies many.

Honest limits: n=3 seeds, single 4B model, single edit grammar, deltas small everywhere outside arm-B A0. The confirmatory protocol (frozen) is where this either becomes a fulcrum or regresses to noise.

Artifacts: `results/gate1_2_campaign_compute.json` (full record incl. per-edit proposals, timings, permutation provenance), `results/gate1_2_campaign_compute.log` (stdout), `results/_catalog_dump.json` (rebuilt annotated catalogs used for §4.1–4.2).
