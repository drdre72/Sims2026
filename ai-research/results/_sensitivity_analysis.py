"""HZN-OC-01 paired within-seed sensitivity analysis (HZN-IR-07 d4).

Sample size for the correspondence and non-inferiority hypotheses is
derived from the distribution of WITHIN-SEED DIFFERENCES, not pooled
unpaired means. For each plausible effect size, Monte-Carlo power of a
one-sided paired design (per-seed differences ~ N(mu, sigma), sign/
t-test analogue) is computed across the number of seeds. eps and delta
express what Horizon considers MEANINGFUL — they are set here as
protocol constants, independently of the Gate 1.2 observed gap (which
is used only to sanity-check the plausible range, per R2).

Outputs results/gate1_3_sensitivity.json + the printed preregistration
table for the spec.
"""
import json
import math
import random
import sys
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent

# --- Protocol constants (preregistered, fixed independent of obs) -----
# eps: maximum practically acceptable utility loss (selection-delta
# units) when replacing L with S. delta: minimum practically meaningful
# genuine-over-permuted advantage. Both are Horizon meaningfulness
# judgments, sized so that an effect smaller than half the measured
# arm-B genuine-vs-blind protection (+0.213) still counts.
EPS = 0.05      # S may cost up to 0.05 selection-delta and be non-inferior
DELTA = 0.05    # genuine must beat permuted by at least 0.05 to count
N_SIMS = 20000
RNG_SEED = 20261124


# One-sided t critical values, alpha=0.05, df 1..30 (beyond -> z).
_T05 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943,
        7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812, 11: 1.796, 12: 1.782,
        13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740,
        18: 1.734, 19: 1.729, 20: 1.725, 21: 1.721, 22: 1.717,
        23: 1.714, 24: 1.711, 25: 1.708, 26: 1.706, 27: 1.703,
        28: 1.701, 29: 1.699, 30: 1.697}


def _t_crit(df):
    return _T05.get(min(int(df), 30), NormalDist().inv_cdf(0.95))


def _paired_power(mu, sigma, n_seeds, margin, n_sims=N_SIMS, seed=RNG_SEED,
                  alpha=0.05):
    """One-sided paired t-test power: P((mean - margin)/SE > t(df=n-1))
    under diffs ~ N(mu, sigma), Monte Carlo on the estimated SE. For the
    non-inferiority test the null boundary is -eps; for correspondence
    it is +delta."""
    rng = random.Random(seed + hash((round(mu, 4), round(sigma, 4), n_seeds)))
    tcrit = _t_crit(n_seeds - 1)
    hits = 0
    for _ in range(n_sims):
        ds = [rng.gauss(mu, sigma) for _ in range(n_seeds)]
        m = sum(ds) / n_seeds
        if n_seeds > 1:
            var = sum((d - m) ** 2 for d in ds) / (n_seeds - 1)
        else:
            var = sigma * sigma
        se = math.sqrt(var / n_seeds) or 1e-9
        if (m - margin) / se > tcrit:
            hits += 1
    return hits / n_sims


def main():
    nd = NormalDist()
    # Gate 1.2 arm-B seed-level A1-vs-A2 diffs: +0.202, +0.017, -0.040.
    # Within-seed sd of the difference ~0.12; plausible effect sizes span
    # zero through the observed mean for honesty.
    diffs_g12 = [0.202, 0.017, -0.040]
    sd_w = math.sqrt(sum((d - sum(diffs_g12) / 3) ** 2
                       for d in diffs_g12) / 2)
    effect_sizes = [0.0, 0.02, DELTA, 0.08, 0.12, 0.20]
    sigmas = sorted({round(sd_w, 3), 0.08, 0.15, 0.20})
    seed_grid = [3, 5, 7, 10, 14, 20]

    table = []
    for sigma in sigmas:
        for mu in effect_sizes:
            row = {"sigma_within": sigma, "true_effect": mu, "cells": {}}
            for n in seed_grid:
                p_corr = _paired_power(mu, sigma, n, DELTA)
                # Non-inferiority at TRUE equality of means (mu_ni=0):
                # power = P(reject 'D <= -eps') under D~N(0, sigma).
                p_ni = _paired_power(0.0, sigma, n, -EPS)
                row["cells"][n] = {"power_correspondence": round(p_corr, 3),
                                   "power_noninferiority": round(p_ni, 3)}
            table.append(row)

    # Chi-square 95% CI on the within-seed sd itself (df=2 — the estimate
    # is genuinely uncertain and the seed count must be robust across it).
    import math as _m
    chi2_lo, chi2_hi = 0.0506, 7.378  # chi2 quantiles, df=2
    sd_lo = sd_w * _m.sqrt(2 / chi2_hi)
    sd_hi = sd_w * _m.sqrt(2 / chi2_lo)

    def min_seeds(mu, sigma, margin, target=0.80):
        for n in range(2, 41):
            if _paired_power(mu, sigma, n, margin) >= target:
                return n
        return None

    recs = {
        "correspondence": {f"mu={mu},sigma={s}": min_seeds(mu, s, DELTA)
                           for mu in (0.08, 0.12, 0.20)
                           for s in (round(sd_w, 3), 0.08, 0.20)},
        "noninferiority_at_equality": {f"sigma={s}":
                                       min_seeds(0.0, s, -EPS)
                                       for s in (round(sd_w, 3), 0.08,
                                                 0.20)},
    }

    out = {"eps": EPS, "delta": DELTA, "alpha": 0.05, "n_sims": N_SIMS,
           "g12_within_seed_sd": round(sd_w, 4),
           "g12_within_seed_sd_ci95": [round(sd_lo, 4), round(sd_hi, 4)],
           "g12_armB_A1_minus_A2": diffs_g12,
           "note": ("eps/delta are preregistered meaningfulness margins, "
                    "independent of Gate 1.2 observations (HZN-IR-07 R2); "
                    "the observed diffs only sanity-check the plausible "
                    "range. Paired within-seed analysis, per R5. The "
                    "within-seed sd is itself uncertain (df=2) — the "
                    "seed recommendation must hold across its CI, not "
                    "just its point estimate."),
           "min_seeds_for_80pct": recs,
           "table": table}
    (HERE / "gate1_3_sensitivity.json").write_text(json.dumps(out, indent=2))

    print(f"within-seed sd of Gate 1.2 A1-A2 diffs: {sd_w:.3f}")
    print(f"eps={EPS}  delta={DELTA}  alpha=0.05  sims={N_SIMS}")
    print("\ncorrespondence power  E[D]>delta   /  non-inferiority power  E[D]>-eps")
    hdr = "sigma   mu    | " + "  ".join(f"n={n:<3d}" for n in seed_grid)
    print(hdr); print("-" * len(hdr))
    for r in table:
        c = "  ".join(f"{r['cells'][n]['power_correspondence']:.2f}/"
                      f"{r['cells'][n]['power_noninferiority']:.2f}"
                      for n in seed_grid)
        print(f"{r['sigma_within']:.2f}  {r['true_effect']:+.2f}  | {c}")

    print(f"\nsd 95% CI: [{sd_lo:.3f}, {sd_hi:.3f}] (df=2 — very wide)")
    print("min seeds for >=80% power:")
    for k, v in recs["correspondence"].items():
        print(f"  correspondence {k:24s} -> n={v if v else '>40'}")
    for k, v in recs["noninferiority_at_equality"].items():
        print(f"  non-inferiority {k:16s} -> n={v if v else '>40'}")


if __name__ == "__main__":
    sys.exit(main())
