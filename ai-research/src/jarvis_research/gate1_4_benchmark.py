"""HZN-IR-11 D4 — candidate-conditioned reasoning benchmark.

Independently verifiable, read-only reasoning tasks in which the
correct answer CANNOT always be determined by selecting the largest
magnitude. Every task presents a decision rule plus a set of labeled
candidates; ground truth is the mechanical application of that rule —
never the retrieval heuristic, never magnitude ranking.

Design contract:
- candidates carry short labels (A, B, C...) — a correct selection
  never requires counting rows or transcribing long identifiers (D5)
- rules are explicit threshold logic stated in the prompt; the grader
  is the same deterministic procedure — agreement is by construction,
  and is unit-tested
- task families cover the required cases: misleading magnitude,
  conflicting observations, insufficient evidence, tied candidates,
  missing observations, multiple plausible explanations, abstention
- the fraction of tasks where the magnitude-max candidate coincides
  with the correct answer is CONTROLLED and reported — the T1
  ranking-only baseline's ceiling is a design parameter, not an
  accident
"""

import random

RULES = {
    "conj_active": {
        "text": ("Select the candidate that is frequently active "
                 "(ar >= {ar_thr}) AND shows strong contrast "
                 "(|lc| >= {lc_thr}). If several qualify, choose the "
                 "one with the highest ar. If none qualify, or if the "
                 "qualifying candidates tie exactly on ar, answer "
                 "none."),
        "ar_thr": 0.50, "lc_thr": 0.30,
    },
    "conj_strong": {
        "text": ("Select the candidate that is frequently active "
                 "(ar >= {ar_thr}) AND shows strong contrast "
                 "(|lc| >= {lc_thr}). If several qualify, choose the "
                 "one with the largest |lc|. If none qualify, or if "
                 "the qualifying candidates tie exactly on |lc|, "
                 "answer none."),
        "ar_thr": 0.40, "lc_thr": 0.50,
    },
    "avoid_negative": {
        "text": ("Select the candidate with the largest |lc| among "
                 "those whose lc is NOT negative (lc >= 0) and whose "
                 "ar is at least {ar_thr}. If none qualify, answer "
                 "none."),
        "ar_thr": 0.30, "lc_thr": 0.0,
    },
}

LABELS = "ABCDE"


def grade_task(rule_key, candidates):
    """Deterministic ground truth: apply the rule mechanically.
    candidates: list of dicts with keys label, ar, lc, obs_available.
    Returns {"answer": label-or-"none", "qualifying": [labels],
             "reason": str}. Missing-observation candidates can never
    qualify (their evidence cannot be verified)."""
    r = RULES[rule_key]
    ar_thr, lc_thr = r["ar_thr"], r["lc_thr"]
    qual = []
    for c in candidates:
        if not c["obs_available"]:
            continue
        if rule_key == "avoid_negative":
            if c["ar"] >= ar_thr and c["lc"] >= 0:
                qual.append(c)
        else:
            if c["ar"] >= ar_thr and abs(c["lc"]) >= lc_thr:
                qual.append(c)
    if not qual:
        return {"answer": "none", "qualifying": [], "reason": "none_qualify"}
    if rule_key == "avoid_negative":
        keyf = lambda c: abs(c["lc"])
    elif rule_key == "conj_active":
        keyf = lambda c: c["ar"]
    else:
        keyf = lambda c: abs(c["lc"])
    best = max(keyf(c) for c in qual)
    winners = [c for c in qual if keyf(c) == best]
    if len(winners) > 1:
        return {"answer": "none", "qualifying": sorted(c["label"] for c in qual),
                "reason": "tie"}
    return {"answer": winners[0]["label"],
            "qualifying": sorted(c["label"] for c in qual),
            "reason": "unique"}


def magnitude_answer(candidates):
    """The T1 deterministic-ranking baseline: pick the obs-available
    candidate with the largest |lc|; ties or all-missing -> none."""
    avail = [c for c in candidates if c["obs_available"]]
    if not avail:
        return "none"
    top = max(abs(c["lc"]) for c in avail)
    wins = [c for c in avail if abs(c["lc"]) == top]
    return wins[0]["label"] if len(wins) == 1 else "none"


def make_task(rng, family, n_cand=5):
    """Construct one task. family in:
      misleading_magnitude  unique answer that is NOT the magnitude max
      conflicting           highest-lc candidate fails the ar test
      insufficient          no candidate qualifies (answer none)
      tied                  two qualifiers tie exactly (answer none)
      missing_obs           some candidates unverifiable; unique answer
      multiple_plausible    several qualify but rule yields unique winner
    Returns dict with rule_key, candidates, plus grader inputs."""
    rule_key = rng.choice(["conj_active", "conj_strong", "avoid_negative"])
    cands = []
    for i in range(n_cand):
        cands.append({"label": LABELS[i],
                      "ar": round(rng.uniform(0.05, 0.95), 2),
                      "lc": round(rng.uniform(-1.8, 1.8), 2),
                      "obs_available": True})

    def big(c):
        return abs(c["lc"])

    if family in ("misleading_magnitude", "conflicting"):
        # magnitude king: huge |lc| but ar below threshold
        king = cands[0]
        king["lc"] = rng.choice([-1, 1]) * round(rng.uniform(1.2, 1.8), 2)
        king["ar"] = round(rng.uniform(0.05, 0.35), 2)
        # rightful: passes both criteria, moderate values
        rightful = cands[1]
        rightful["ar"] = round(rng.uniform(0.6, 0.9), 2)
        rightful["lc"] = rng.choice([-1, 1]) * round(rng.uniform(0.45, 0.8), 2)
        # everyone else fails at least one criterion
        for c in cands[2:]:
            if rng.random() < 0.5:
                c["ar"] = round(rng.uniform(0.05, 0.3), 2)
                c["lc"] = rng.choice([-1, 1]) * round(rng.uniform(0.5, 1.0), 2)
            else:
                c["ar"] = round(rng.uniform(0.6, 0.9), 2)
                c["lc"] = rng.choice([-1, 1]) * round(rng.uniform(0.02, 0.2), 2)
    elif family == "insufficient":
        # constructed so NO rule can qualify anyone: high-ar candidates
        # carry small-magnitude NEGATIVE lc (fail conj rules on |lc|,
        # fail avoid_negative on sign); big-lc candidates have ar below
        # every threshold
        for c in cands:
            if rng.random() < 0.5:
                c["ar"] = round(rng.uniform(0.05, 0.25), 2)
                c["lc"] = rng.choice([-1, 1]) * round(rng.uniform(0.5, 1.6), 2)
            else:
                c["ar"] = round(rng.uniform(0.55, 0.9), 2)
                c["lc"] = -round(rng.uniform(0.05, 0.2), 2)
    elif family == "tied":
        # two candidates qualify with IDENTICAL ar, |lc| and sign —
        # tied under every rule's tie-break
        a, b = cands[0], cands[1]
        shared_ar = round(rng.uniform(0.6, 0.9), 2)
        shared_lc = round(rng.uniform(0.5, 0.9), 2)
        a["ar"] = b["ar"] = shared_ar
        a["lc"] = b["lc"] = shared_lc
        for c in cands[2:]:
            c["ar"] = round(rng.uniform(0.05, 0.25), 2)
            c["lc"] = rng.choice([-1, 1]) * round(rng.uniform(0.5, 1.4), 2)
    elif family == "missing_obs":
        # king is unverifiable; rightful unique
        king = cands[0]
        king["lc"] = rng.choice([-1, 1]) * round(rng.uniform(1.3, 1.8), 2)
        king["ar"] = round(rng.uniform(0.7, 0.95), 2)
        king["obs_available"] = False
        rightful = cands[1]
        rightful["ar"] = round(rng.uniform(0.6, 0.9), 2)
        rightful["lc"] = rng.choice([-1, 1]) * round(rng.uniform(0.45, 0.8), 2)
        for c in cands[2:]:
            if rng.random() < 0.4:
                c["obs_available"] = False
            c["ar"] = round(rng.uniform(0.05, 0.3), 2)
            c["lc"] = rng.choice([-1, 1]) * round(rng.uniform(0.5, 1.0), 2)
    elif family == "multiple_plausible":
        # several qualify, unique winner by tie-break metric — and the
        # magnitude king qualifies but does NOT win the tie-break for
        # conj_active (winner decided by ar)
        rightful = cands[1]
        rightful["ar"] = round(rng.uniform(0.85, 0.95), 2)
        rightful["lc"] = rng.choice([-1, 1]) * round(rng.uniform(0.4, 0.6), 2)
        king = cands[0]
        king["ar"] = round(rng.uniform(0.55, 0.75), 2)
        king["lc"] = rng.choice([-1, 1]) * round(rng.uniform(1.0, 1.6), 2)
        for c in cands[2:]:
            c["ar"] = round(rng.uniform(0.05, 0.3), 2)
            c["lc"] = rng.choice([-1, 1]) * round(rng.uniform(0.5, 1.0), 2)
    else:
        raise ValueError(family)

    rng.shuffle(cands)
    # relabel after shuffle so labels have no positional correlation
    for i, c in enumerate(cands):
        c["label"] = LABELS[i]
    truth = grade_task(rule_key, cands)
    # tasks whose construction didn't yield the intended answer shape
    # are the caller's problem — we report what the rule says, always
    return {"family": family, "rule_key": rule_key, "candidates": cands,
            "truth": truth, "magnitude_answer": magnitude_answer(cands)}


FAMILIES = ("misleading_magnitude", "conflicting", "insufficient",
            "tied", "missing_obs", "multiple_plausible")


def render_task_body(task):
    """Rule + labeled candidates at exactly the graded precision, without
    an output contract. Shared by reasoning and mirror renderers."""
    rule = RULES[task["rule_key"]]
    text = rule["text"].format(ar_thr=rule["ar_thr"], lc_thr=rule["lc_thr"])
    lines = []
    for c in task["candidates"]:
        if c["obs_available"]:
            lines.append(f"candidate {c['label']}: ar={c['ar']:.2f}, "
                         f"lc={c['lc']:.2f}")
        else:
            lines.append(f"candidate {c['label']}: no observations "
                         f"available")
    return f"Rule: {text}\n\n" + "\n".join(lines)


def render_task(task):
    """The model-facing task text: rule + labeled candidates + output
    instruction. Values rendered at exactly the graded precision."""
    return (render_task_body(task) +
            "\n\nReply with ONLY the candidate letter, or the word none.")
