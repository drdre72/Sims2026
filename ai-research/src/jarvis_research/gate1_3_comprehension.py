"""HZN-OC-01 comprehension pretest (C1-C4), reviewer directive §3.

Four-level decoding test for a candidate observation-channel encoding.
Items are generated MECHANICALLY from synthetic or held-out catalogs with
independently known answers — never from campaign trajectories and never
exposing intervention outcomes. A level that fails reports 'preserved but
undecodable', not 'information lost'; it is skipped, not retried.

  C1 decoding    — recover encoded fields (layer, family, signed contrast)
  C2 comparison  — relational interpretation (greater |contrast|)
  C3 composition — multi-constraint satisfaction over rows
  C4 grounding   — observation -> candidate param_id (operational gate)
"""

import json
import random
import re

from jarvis_research.gate1_3_channel import (
    CHANNEL_LEVELS, encode_catalog_text, encode_row)


def make_synthetic_catalog(seed: int, n_rows: int = 36,
                           n_families: int = 4, n_layers: int = 12):
    """Synthetic annotated catalog with unique (family, layer) pairs so
    C4 grounding has exactly one correct answer per description."""
    rng = random.Random(seed)
    fams = [f"syn.fam{i}" for i in range(n_families)]
    pairs = [(f, l) for f in fams for l in range(n_layers)]
    rng.shuffle(pairs)
    rows = []
    for i in range(n_rows):
        fam, layer = pairs[i % len(pairs)]
        rows.append({
            "param_id": i,
            "param": f"syn.layers.{layer}.fam{i}.w",
            "family": fam, "layer": layer,
            "numel": rng.choice((65536, 98304, 147456, 294912)),
            "mag_min": -0.005, "mag_max": 0.005,
            "count_max": 512,
            "obs_available": rng.random() > 0.1,
            "obs_act_rank": round(rng.random(), 2),
            "obs_log_contrast": round(rng.uniform(-2.0, 2.0), 2),
        })
    return rows


def _render_row(r, fam_code, level):
    return encode_row(r, fam_code, True, True, level)


def generate_items(catalog, fam_code, level: str, seed: int):
    """Deterministically generate one probe set for a channel level."""
    rng = random.Random(seed)
    avail = [r for r in catalog if r["obs_available"]]
    items = []

    # C1 — decoding: recover a field from one rendered row.
    for r in rng.sample(avail, min(4, len(avail))):
        row_txt = _render_row(r, fam_code, level)
        items.append({
            "level": "C1",
            "context": f"Catalog row:\n{row_txt}",
            "query": "What is the layer of this component? "
                     "Answer with the integer only.",
            "expected": str(r["layer"]),
        })
        items.append({
            "level": "C1",
            "context": f"Catalog row:\n{row_txt}",
            "query": "What is the signed contrast (lc) value of this row? "
                     "Answer with the number only.",
            "expected": f"{round(r['obs_log_contrast'], 2):+.2f}",
        })

    # C2 — comparison: pick two rows with distinct |lc|.
    for _ in range(4):
        a, b = rng.sample(avail, 2)
        if abs(round(a["obs_log_contrast"], 2)) == abs(round(b["obs_log_contrast"], 2)):
            continue
        winner = a if abs(a["obs_log_contrast"]) > abs(b["obs_log_contrast"]) else b
        items.append({
            "level": "C2",
            "context": ("Catalog rows:\n"
                        + _render_row(a, fam_code, level) + "\n"
                        + _render_row(b, fam_code, level)),
            "query": "Which component has the GREATER absolute contrast "
                     "|lc|? Answer with the component id only.",
            "expected": str(winner["param_id"]),
        })

    # C3 — composition: conjunctive filter over the full catalog.
    fams = sorted({r["family"] for r in catalog})
    target_fam = rng.choice(fams)
    layer_min = rng.choice([3, 6, 9])
    want_pos = rng.random() < 0.5
    fc = fam_code[target_fam]
    hits = sorted(r["param_id"] for r in catalog
                  if r.get("obs_available")
                  and r["family"] == target_fam
                  and r["layer"] >= layer_min
                  and (r["obs_log_contrast"] > 0) == want_pos)
    famref = fc[1:] if level == "s2" else fc
    items.append({
        "level": "C3",
        "context": "Catalog:\n" + encode_catalog_text(
            catalog, fam_code, True, True, level),
        "query": (f"List every component id with family {famref}, "
                  f"layer >= {layer_min}, and "
                  f"{'positive' if want_pos else 'negative'} lc. "
                  "Answer as a JSON list of integer ids."),
        "expected": hits,
    })

    # C4 — grounding: observation description -> param_id.
    for r in rng.sample(avail, min(3, len(avail))):
        fc_r = fam_code[r["family"]]
        famref = fc_r[1:] if level == "s2" else r["family"]
        items.append({
            "level": "C4",
            "context": "Catalog:\n" + encode_catalog_text(
                catalog, fam_code, True, True, level),
            "query": (f"An observation is described as: family {famref}, "
                      f"layer {r['layer']}. Which param_id "
                      f"({'O:id' if level == 's2' else 'id'}) does it "
                      "address? Answer with the integer id only."),
            "expected": str(r["param_id"]),
        })
    return items


def grade(item: dict, response: str) -> bool:
    """Exact/normalized match; never repaired."""
    exp = item["expected"]
    text = response.strip()
    if isinstance(exp, list):
        m = re.search(r"\[[^\]]*\]", text)
        if not m:
            return False
        try:
            got = sorted(int(x) for x in json.loads(m.group(0)))
        except Exception:
            return False
        return got == exp
    nums = re.findall(r"-?\d+\.?\d*", text)
    if not nums:
        return False
    if exp.lstrip("+-").replace(".", "", 1).isdigit() and "." in exp:
        try:
            return any(abs(float(n) - float(exp)) < 0.005 for n in nums)
        except ValueError:
            return False
    return any(int(float(n)) == int(exp) for n in nums)


def score_items(items, responses):
    """Per-level accuracy for paired (item, response) lists."""
    per = {}
    for it, resp in zip(items, responses):
        ok = grade(it, resp)
        a = per.setdefault(it["level"], [0, 0])
        a[0] += ok
        a[1] += 1
    return {lvl: {"correct": c, "n": n, "acc": c / n}
            for lvl, (c, n) in per.items()}


def run_pretest(model, tokenizer, catalog, fam_code, level: str,
                seed: int = 0, max_new: int = 48, device=None):
    """Generate items, prompt the model, grade. Returns (scores, detail).

    Deterministic decode (no sampling) — the pretest measures decoding
    literacy, not proposal behavior.
    """
    import torch
    items = generate_items(catalog, fam_code, level, seed)
    responses = []
    dev = device or next(model.parameters()).device
    for it in items:
        user = (f"{it['context']}\nQuestion: {it['query']}")
        enc = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            add_generation_prompt=True, return_tensors="pt",
            return_dict=True, enable_thinking=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False,
                                 max_new_tokens=max_new)
        responses.append(tokenizer.decode(
            gen[0, enc["input_ids"].shape[1]:],
            skip_special_tokens=True).strip())
    return score_items(items, responses), list(zip(items, responses))


# Reviewer directive 8: differential engineering targets per level —
# C4 grounding is the operational gate and must be perfect on the
# finite preflight set; deterministic address validation still runs
# during execution regardless of pretest outcome.
PRETEST_THRESHOLDS = {"C1": 0.95, "C2": 0.90, "C3": 0.90, "C4": 1.00}


def pretest_passed(scores, thresholds: dict = PRETEST_THRESHOLDS) -> bool:
    return all(v["acc"] >= thresholds.get(lvl, 1.0)
               for lvl, v in scores.items())
