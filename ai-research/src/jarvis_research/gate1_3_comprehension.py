"""HZN-OC-01 comprehension pretest (spec v3 + HZN-IR-08 hardening).

Four-level operational-literacy gate between tokenizer profiling and any
in-context evaluation. C1 decoding -> C2 comparison -> C3 composition ->
C4 grounding to catalog addresses. An encoding is eligible for
information-matched evaluation only if the proposer can read it —
otherwise "S1 does not beat S2" is confounded by literacy, not evidence
correspondence.

Items are generated mechanically from synthetic catalogs so the
answer key is known without a judge model; the catalogs reproduce the
real campaign's size and contrast structure (many near-zero contrasts,
rounded ties, sign ambiguity, repeated family/layer pairs so C4 probes
must discriminate addressing, not just description).

HZN-IR-08 changes:
  F1 — pretest_passed / score_items fail CLOSED: every expected level
       must be present, every level must meet its preregistered minimum
       probe count, and incomplete response sets score the missing
       items as failures (never silently truncating the denominator).
  F2 — strict whole-response grading: an integer answer must BE the
       response, not appear in it; decimals parse the entire response;
       lists must be valid JSON of valid integer ids (no coercion).
  §3 — realistic catalogs (campaign-sized, near-zero contrast
       distribution, rounded ties) and preregistered probe counts:
       C4 requires >=60 zero-failure probes across >=3 independently
       generated catalogs.
"""

import json
import random
import re

from jarvis_research.gate1_3_channel import (
    CHANNEL_LEVELS, encode_row, encode_catalog_text, decode_row,
    s3_column_order)


# Differential thresholds (HZN-IR-07 finding 3 — grounding errors act on
# the wrong component; C4 demands zero observed failures on the finite
# preregistered set).
PRETEST_THRESHOLDS = {"C1": 0.95, "C2": 0.90, "C3": 0.90, "C4": 1.00}
PRETEST_ORDER = ("C1", "C2", "C3", "C4")

# Preregistered minimum probe counts per level (HZN-IR-08 d6). A level
# with fewer items than this FAILS CLOSED — sparse positive evidence is
# not a pass. C4's >=60/zero-fail target: 59 independent successes give a
# one-sided 95% lower bound > 95% on true accuracy.
MIN_PROBES = {"C1": 40, "C2": 40, "C3": 20, "C4": 60}
MIN_C4_CATALOGS = 3


def synthetic_catalog(n_rows=248, n_layers=32, n_fams=7, seed=0,
                      near_zero_frac=0.55, tie_frac=0.15, n_unique=30):
    """Synthetic catalog reproducing the real campaign's structure:
    campaign-sized (default 248 rows, matching arm B), ~55% of
    contrasts within ±0.05 of zero, ~15% rounded to a tie with a
    neighbor, and repeated (family, layer) pairs — real addressing is
    ambiguous and C4 must discriminate it. `n_unique` rows carry
    reserved (family, layer) pairs no other row repeats — guaranteed
    unambiguous grounding probes; the rest share the remaining pair
    space so genuine ambiguity exists for the refusal probes.
    """
    rng = random.Random(seed)
    fams = (["mlp.down_proj"] +
            [f"attn.{n}_proj" for n in ("q", "k", "v", "o")] +
            [f"expert.{i}.w" for i in range(max(0, n_fams - 5))])[:n_fams]
    # Reserved unique pairs: first n_unique of the enumeration; random
    # rows sample the complement so the reservation can never be hit.
    all_pairs = [(f, l) for f in fams for l in range(n_layers)]
    reserved = set(all_pairs[:n_unique])
    open_pairs = [p for p in all_pairs if p not in reserved]
    cat = []
    pid = 0
    for f, l in sorted(reserved, key=lambda p: (p[0], p[1])):
        cat.append(_row(rng, pid, f, l, near_zero_frac))
        pid += 1
    while len(cat) < n_rows:
        fam, layer = open_pairs[rng.randrange(len(open_pairs))]
        cat.append(_row(rng, pid, fam, layer, near_zero_frac))
        pid += 1
    # Rounded-tie clusters: give some rows an identical rounded lc.
    n_ties = int(len(cat) * tie_frac)
    for i in range(0, min(n_ties, len(cat) - 1), 2):
        cat[i + 1]["obs_log_contrast"] = cat[i]["obs_log_contrast"]
        cat[i + 1]["obs_act_rank"] = cat[i]["obs_act_rank"]
    # F7: randomize row order and independently permute param_ids — the
    # reserved-unique rows must not occupy predictable positions or ids.
    rng.shuffle(cat)
    id_perm = rng.sample(range(n_rows), n_rows)
    for row, pid in zip(cat, id_perm):
        row["param_id"] = pid
    return cat


def _row(rng, pid, fam, layer, near_zero_frac):
    return {
        "param_id": pid, "family": fam, "layer": layer,
        "numel": rng.choice([147456, 589824, 262144, 9437184, 32768,
                             131072]),
        "mag_min": -rng.choice([0.002, 0.005, 0.010]),
        "mag_max": rng.choice([0.002, 0.005, 0.010]),
        "obs_available": True,
        "obs_act_rank": round(rng.uniform(0, 1), 2),
        "obs_log_contrast": (round(rng.uniform(-0.05, 0.05), 2)
                             if rng.random() < near_zero_frac
                             else round(rng.uniform(-2.0, 2.0), 2)),
    }


def gen_items(catalog, fam_code, uniform_mag, level, n_per=8, seed=0,
              counts=None):
    """Generate a four-level item set on `catalog` rendered at `level`.

    `counts` may override per-level probe counts {"C1":n,...}; n_per is
    the uniform default. Returns {"C1":[...], "C2":[...], "C3":[...],
    "C4":[...]} of (prompt, expected) pairs. C4 items come in two
    kinds: UNAMBIGUOUS probes — a description matching exactly one row,
    expected = its id; and AMBIGUOUS probes — a (family, layer)
    description matching >1 row, expected = "ambiguous" (the model must
    refuse to pick; picking an arbitrary match is a grounding failure).
    """
    rng = random.Random(seed)
    inv = {c: f for f, c in fam_code.items()}
    rows = [encode_row(e, fam_code, uniform_mag, True, level)
            for e in catalog]
    # C4 ambiguity bookkeeping on the REAL (synthetic) catalog structure.
    from collections import defaultdict
    by_pair = defaultdict(list)
    for e in catalog:
        by_pair[(e["family"], e["layer"])].append(e["param_id"])
    unique_pairs = {p for p, ids in by_pair.items() if len(ids) == 1}
    ambig_pairs = {p for p, ids in by_pair.items() if len(ids) > 1}

    counts = counts or {}
    n1 = counts.get("C1", n_per)
    n2 = counts.get("C2", n_per)
    n3 = counts.get("C3", max(1, n_per // 2))
    n4 = counts.get("C4", n_per)

    # F3: positional (s3) rows shown in isolation must carry their
    # schema — the field order is the encoding, not just the values.
    hdr = None
    if level == "s3":
        hdr = "cols: " + "|".join(
            s3_column_order(catalog, uniform_mag, True))

    items = {"C1": [], "C2": [], "C3": [], "C4": []}
    idx = rng.sample(range(len(catalog)), min(n1, len(catalog)))
    for i in idx:
        body = f"{hdr}\n{rows[i]}" if hdr else rows[i]
        items["C1"].append((body,
                            {"ar": round(catalog[i]["obs_act_rank"], 2),
                             "lc": round(catalog[i]["obs_log_contrast"], 2)}))
    made2, tries = 0, 0
    while made2 < n2 and tries < n2 * 20:
        tries += 1
        a, b = rng.sample(range(len(catalog)), 2)
        ea, eb = catalog[a], catalog[b]
        # F1: the question compares |lc| — the tie predicate must too.
        if abs(abs(ea["obs_log_contrast"])
               - abs(eb["obs_log_contrast"])) < 1e-9:
            continue
        body = (f"{hdr}\n{rows[a]}\n{rows[b]}" if hdr
                else f"{rows[a]}\n{rows[b]}")
        items["C2"].append((body,
                            ea["param_id"] if abs(ea["obs_log_contrast"])
                            > abs(eb["obs_log_contrast"])
                            else eb["param_id"]))
        made2 += 1
    made3, tries3 = 0, 0
    while made3 < n3 and tries3 < n3 * 40:
        tries3 += 1
        pool = rng.sample(range(len(catalog)), min(len(catalog), 12))
        ranked = sorted(pool,
                        key=lambda i: -abs(catalog[i]["obs_log_contrast"]))
        # F2: no boundary ties — the gap between rank 3 and rank 4 must
        # be clean or the pool is ambiguous by construction.
        if len(ranked) > 3 and abs(
                abs(catalog[ranked[2]]["obs_log_contrast"])
                - abs(catalog[ranked[3]]["obs_log_contrast"])) < 1e-9:
            continue
        hi = ranked[:3]
        body = (hdr + "\n" if hdr else "") + "\n".join(rows[i]
                                                     for i in pool)
        items["C3"].append((body,
                           sorted(catalog[i]["param_id"] for i in hi)))
        made3 += 1
    # Unambiguous grounding: unique (family, layer) rows.
    uniq_rows = [e for e in catalog
                 if (e["family"], e["layer"]) in unique_pairs]
    rng.shuffle(uniq_rows)
    for e in uniq_rows[:n4]:
        fam_tag = fam_code[e["family"]]
        items["C4"].append((
            f"The row with family tag {fam_tag} and layer {e['layer']}",
            int(e["param_id"])))
    # Ambiguous grounding (F4: deterministic sorted+seeded sampling).
    amb_list = sorted(ambig_pairs)
    rng.shuffle(amb_list)
    for pair in amb_list[:max(1, n4 // 3)]:
        fam_tag = fam_code[pair[0]]
        items["C4"].append((
            f"The row with family tag {fam_tag} and layer {pair[1]}",
            "ambiguous"))
    return items


# ---- Strict grading (HZN-IR-08 F2) -------------------------------------

_NUM_RE = re.compile(r"[+-]?\d+(?:\.\d+)?")


def grade_integer(expected, response):
    """Integer answer must BE the response (whole-response match).
    '7.9', '7 or 12', 'not 7, 9' all fail — the correct id appearing
    somewhere in the output is not grounding."""
    text = response.strip()
    if not re.fullmatch(r"[+-]?\d+", text):
        return False
    return int(text) == expected


def grade_decimal(expected, response):
    text = response.strip()
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
        return False
    return abs(float(text) - expected) <= 0.011


def grade_list(expected, response):
    """JSON list of valid integer ids; no coercion of floats/strings."""
    text = response.strip()
    try:
        got = json.loads(text)
    except json.JSONDecodeError:
        return False
    if not isinstance(got, list):
        return False
    if not all(isinstance(v, int) and not isinstance(v, bool)
               for v in got):
        return False
    return sorted(got) == sorted(expected)


def grade(item_kind, expected, response):
    if item_kind == "C1":
        # F6: decoding is EXACT — the rendered value is 2dp and the
        # correct response must reproduce it, not an adjacent hundredth.
        # Approximate numerical comprehension is a separate metric
        # (grade_c1_approx) that is reported but does not gate.
        try:
            nums = {k: round(float(v), 2) for k, v in
                    json.loads(response.strip()).items()}
        except (json.JSONDecodeError, ValueError, AttributeError):
            return False
        return (nums.get("ar") == round(expected["ar"], 2) and
                nums.get("lc") == round(expected["lc"], 2))
    if item_kind == "C4" and expected == "ambiguous":
        return response.strip().lower() == "ambiguous"
    if item_kind in ("C2", "C4"):
        return grade_integer(expected, response)
    if item_kind == "C3":
        return grade_list(expected, response)
    raise ValueError(item_kind)


def score_items(level_items, responses):
    """Score a response set; MISSING responses count as failures
    (HZN-IR-08 F1 — never silently truncate the denominator). Returns
    {lvl: {"n": total, "correct": k, "acc": k/total}}."""
    scores = {}
    for lvl, items in level_items.items():
        rs = responses.get(lvl, [])
        correct = sum(
            grade(lvl, exp, rs[i]) if i < len(rs) and rs[i] is not None
            else False
            for i, (prompt, exp) in enumerate(items))
        scores[lvl] = {"n": len(items), "correct": correct,
                       "acc": correct / len(items) if items else 0.0}
    return scores


def pretest_passed(scores, thresholds=PRETEST_THRESHOLDS,
                   min_probes=MIN_PROBES):
    """Blocking gate — FAILS CLOSED (HZN-IR-08 F1): every expected level
    must be present, at least min_probes[lvl] items scored, and acc must
    meet its differential threshold. Empty/partial score dicts fail."""
    for lvl in PRETEST_ORDER:
        v = scores.get(lvl)
        if v is None or v.get("n", 0) < min_probes.get(lvl, 0):
            return False
        if v["acc"] < thresholds.get(lvl, 1.0):
            return False
    return True


def grade_c1_approx(expected, response, tol=0.011):
    """Approximate numerical comprehension — reported separately from
    exact decoding (F6), never gates the pretest."""
    try:
        nums = {k: round(float(v), 2) for k, v in
                json.loads(response.strip()).items()}
    except (json.JSONDecodeError, ValueError, AttributeError):
        return False
    return (abs(nums.get("ar", -99) - expected["ar"]) <= tol and
            abs(nums.get("lc", -99) - expected["lc"]) <= tol)


def c4_coverage(catalog_identities):
    """C4 probes must span >=MIN_C4_CATALOGS INDEPENDENTLY generated
    catalogs (HZN-IR-08 §3 + IR-09 F5): verify distinct identities
    (content hashes and generation seeds), not just a count."""
    ids = {(c.get("catalog_sha256"), c.get("seed"))
           for c in catalog_identities}
    return len(ids) >= MIN_C4_CATALOGS and all(
        c.get("catalog_sha256") for c in catalog_identities)


def prompt_for(item_kind, prompt):
    if item_kind == "C1":
        return (f"For each encoded row, reply with ONLY a JSON object "
                f'{{"ar": <float>, "lc": <float>}}:\n{prompt}')
    if item_kind == "C2":
        return (f"Which row has the larger |lc|? Reply with ONLY the "
                f"leading id number:\n{prompt}")
    if item_kind == "C3":
        return (f"Reply with ONLY a JSON list of the 3 ids with the "
                f"largest |lc|:\n{prompt}")
    if item_kind == "C4":
        return (f"Give the id number of {prompt}. If more than one row "
                f"matches, reply with ONLY the word ambiguous. "
                f"Reply with ONLY the number:")
    raise ValueError(item_kind)
