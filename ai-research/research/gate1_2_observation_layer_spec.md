# Gate 1.2 — Observation-Layer Specification (DRAFT for review)

Status: **specification only — no implementation authorized.** This document
proposes the next interface revision so an independent reviewer can approve,
reject, or amend it BEFORE code is written. It changes no edit surfaces,
model families, endpoints, or confirmatory blockers.

## 1. Scientific motivation

Gate 1.1 showed the proposer reliably produces schema-valid edits (96/96
applied) but its selections do not beat matched controls. Diagnosis, per the
Frontier Synthesis and the independent review: the catalog is a *parts
inventory* — every row is structurally anonymous. Fan et al.'s successful
27B model intervened *after* identifying a contrast between internal
states associated with different behaviors. Our 4B model currently has no
channel through which functional evidence can reach it.

**Hypothesis H-obs:** adding forward-only, practice-derived activation
statistics to catalog rows changes proposer behavior measurably — either
selection CE vs. the unchanged catalog (if evidence is usable) or at
minimum proposal distribution (if the model attends to the new columns).

**Non-goals:** proving causal responsibility, mechanistic understanding,
or that the model "understands itself." Activation statistics are
*correlational descriptors* presented as diagnostic input only.

## 2. Proposed observation: high/low-loss activation contrast

Mimic the published diagnostic structure without gradients:

1. Compute per-passage loss on **practice** passages only.
2. Split into high-loss vs. low-loss subsets (median split or fixed
   threshold on the baseline checkpoint).
3. Register forward hooks capturing per-module activation statistics
   (output-tensor RMS and max |activation|) for each catalog parameter's
   owning module, separately on high- and low-loss subsets.
4. Emit contrast ratios: `hi/lo` activation ratio per catalog row.

Catalog row becomes:
`id|L|fam|n=..[|m=cap][|c<=N][|act=0.42|hilo=1.8]`

- `act` = mean output RMS on all practice passages (scale of use)
- `hilo` = ratio of mean activation on high-loss vs low-loss practice
  passages (association with failure conditions; >1 = more active under
  high loss — NOT proof of causation)

Prompt adds one sentence: *"act=typical activation scale; hilo=ratio of
activation on high- vs low-loss practice text (correlational, not causal)."*

## 3. Implementation sketch (if approved)

- `gate1_observation.py`: forward hooks on the parent module of each
  catalog parameter (match by parameter name prefix); two practice
  passes (sorted by baseline loss) with `torch.no_grad()`; per-catalog
  aggregation. Cost: ~2 extra practice-forward passes per run —
  negligible vs. evaluation budget.
- Statistics computed ONCE per run on the unmodified checkpoint and held
  fixed for all candidates/conditions (avoids leaking candidate state and
  keeps conditions identical).
- Catalog entries carry the raw values; only the proposer's TEXT changes.
- No gradients anywhere — consistent with the no-gradient rule.

## 4. Control fairness — requires a reviewer decision

Options, with implications:

| Option | Consequence |
|---|---|
| A. Controls keep blind catalog | Tests "model + evidence" vs blind search — measures the observation layer's value, confounds model-vs-search |
| B. Random/adaptive sample weighted by `hilo` | Tests whether the STATISTIC does the work vs. the model's judgment — the strongest comparison |
| C. Both reported | Maximum information, ~2x candidate count |

Recommendation: **C within a fixed candidate budget** — pilot sized so
total runtime stays under ~30 min.

## 5. Falsification criteria

- Primary: Δ selection CE of autonomous+obs vs. autonomous (same seeds).
- Secondary: does proposer disproportionately target high-`hilo` or
  low-`act` rows? (measures whether it uses the signal at all)
- If proposal distributions are unchanged AND selection CE unchanged →
  the observation layer is unused; hypothesis revised, not abandoned.

## 6. Explicit non-changes

- No new edit surfaces, magnitudes, or arm boundaries.
- No gradient-based attribution (prohibited by protocol).
- No selection/regression/holdout content in observations — practice
  passages only.
- Prompt isolation remains prompt isolation; no new access boundary.
- n=3 seeds remains exploratory; confirmatory protocol still frozen.
