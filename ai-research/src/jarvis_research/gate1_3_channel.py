"""HZN-OC-01 observation-channel encodings, audit identity, token profiling.

Exploratory component set (reviewer-authorized, spec v3): candidate
renderings of the proposer-visible catalog, a mechanical CANONICAL field-
equivalence check (HZN-IR-08 F3 — Decode(Encode(O)) must return every
proposer-visible semantic, including effective magnitude bounds and
defaults, not just the fields that happen to be rendered per-row), the
O=(a,t,v,m,p) observation-identity record with mandatory provenance, and
tokenizer-measured cost profiling.

Channel levels:
  linguistic — the Gate 1.2 hybrid pipe schema (baseline)
  s1         — space-delimited compact schema, familiar key=value
  s2         — fixed symbolic vocabulary, integer-coded obs fields
  s3         — headered positional schema (structural redundancy removed:
               field names rendered once in a header, rows positional)
"""

import hashlib
import json
import re


CHANNEL_LEVELS = ("linguistic", "s1", "s2", "s3")


def _obs_fields(entry: dict):
    """The two proposer-visible evidence fields at rendered precision."""
    if not entry.get("obs_available"):
        return None, None
    return (round(entry["obs_act_rank"], 2),
            round(entry["obs_log_contrast"], 2))


def _rendered_mag(entry: dict, uniform_mag: bool):
    """Effective magnitude interval at RENDERED precision (HZN-IR-08 F3).

    Returns (min, max) as floats of the rendered values: a symmetric cap
    renders as `m={cap*0.999:.4g}` whose canonical semantics are ±cap;
    an interval renders `mag[{lo*0.999:.4g},{hi*0.999:.4g}]`. Uniform
    arms render no per-row field — parity must then compare the
    legend-level arm bound, supplied separately.
    """
    if uniform_mag:
        return None
    if abs(entry["mag_min"] + entry["mag_max"]) < 1e-12:
        c = float(f"{entry['mag_max'] * 0.999:.4g}")
        return (-c, c)
    return (float(f"{entry['mag_min'] * 0.999:.4g}"),
            float(f"{entry['mag_max'] * 0.999:.4g}"))


def canonical_record(entry: dict, uniform_mag: bool, arm_mag=None,
                     obs: bool = True, arm_count_max: int = 512) -> dict:
    """The canonical typed candidate record — the complete set of
    proposer-visible semantics at rendered precision (HZN-IR-08 F3).

    Field parity is defined on THIS record, not on the rendered tokens:
    every channel must decode back to it exactly, including fields that
    live in the legend (uniform magnitude, count defaults).
    """
    mag = _rendered_mag(entry, uniform_mag)
    if mag is None and arm_mag is not None:
        mag = (float(f"{arm_mag[0]:.4g}"), float(f"{arm_mag[1]:.4g}"))
    ar, lc = _obs_fields(entry) if obs else (None, None)
    cnt = entry.get("count_max")
    return {
        "param_id": int(entry["param_id"]),
        "layer": int(entry["layer"]),
        "family": entry["family"],
        "numel": int(entry["numel"]),
        "mag_min": mag[0] if mag else None,
        "mag_max": mag[1] if mag else None,
        "count_max": int(cnt) if cnt is not None and cnt < 512
                    else arm_count_max,
        "obs_available": ar is not None,
        "ar": ar,
        "lc": lc,
    }


def encode_row(entry: dict, fam_code: dict, uniform_mag: bool,
               obs: bool, level: str, column_order=None) -> str:
    """Render one catalog row in the given channel level. For s3 the
    caller supplies the catalog-wide column_order so every row emits
    the same positional fields ('-' for absent values)."""
    pid, layer = entry["param_id"], entry["layer"]
    fc = fam_code[entry["family"]]
    numel = entry["numel"]
    mag = _rendered_mag(entry, uniform_mag)
    cnt = entry.get("count_max") if entry.get("count_max", 512) < 512 else None
    ar, lc = _obs_fields(entry) if obs else (None, None)

    if level == "linguistic":
        line = f"{pid}|L{layer}|{fc}|n={numel}"
        if mag is not None:
            if abs(mag[0] + mag[1]) < 1e-12:
                line += f"|m={mag[1]:.4g}"
            else:
                line += f"|mag[{mag[0]:.4g},{mag[1]:.4g}]"
        if cnt is not None:
            line += f"|c<={cnt}"
        if obs:
            line += (f"|ar={ar:.2f}|lc={lc:+.2f}" if ar is not None
                     else "|ar=NA|lc=NA")
        return line

    if level == "s1":
        parts = [str(pid), f"L{layer}", fc, f"n={numel}"]
        if mag is not None:
            if abs(mag[0] + mag[1]) < 1e-12:
                parts.append(f"m={mag[1]:.4g}")
            else:
                parts.append(f"mag[{mag[0]:.4g},{mag[1]:.4g}]")
        if cnt is not None:
            parts.append(f"c<={cnt}")
        if obs:
            parts += ([f"ar={ar:.2f}", f"lc={lc:+.2f}"] if ar is not None
                      else ["ar=NA", "lc=NA"])
        return " ".join(parts)

    if level == "s2":
        parts = [f"O:{pid}", f"L:{layer}", f"F:{fc[1:]}", f"N:{numel}"]
        if mag is not None:
            if abs(mag[0] + mag[1]) < 1e-12:
                parts.append(f"M:{mag[1]:.4g}")
            else:
                parts.append(f"M:{mag[0]:.4g},{mag[1]:.4g}")
        if cnt is not None:
            parts.append(f"C:{cnt}")
        if obs:
            if ar is None:
                parts += ["AR:--", "LC:--"]
            else:
                parts += [f"AR:{int(round(ar * 100))}",
                          f"LC:{int(round(lc * 100)):+d}"]
        return "|".join(parts)

    if level == "s3":
        # Headered positional schema: field names appear once in the
        # 'cols:' header; rows emit EVERY catalog-wide column
        # positionally, '-' for absent/NA values.
        vals = {"id": str(pid), "layer": str(layer), "fam": fc,
                "numel": str(numel)}
        if mag is not None:
            vals["mag"] = (f"{mag[1]:.4g}"
                           if abs(mag[0] + mag[1]) < 1e-12
                           else f"{mag[0]:.4g},{mag[1]:.4g}")
        if cnt is not None:
            vals["cnt"] = str(cnt)
        if ar is not None:
            vals["ar"], vals["lc"] = f"{ar:.2f}", f"{lc:+.2f}"
        if column_order is None:
            column_order = [k for k in
                            ("id", "layer", "fam", "numel", "mag", "cnt",
                             "ar", "lc") if k in vals]
        return "|".join(vals.get(c, "-") for c in column_order)

    raise ValueError(f"unknown channel level {level!r}")


def decode_row(line: str, fam_legend: dict, level: str,
               column_order=None) -> dict:
    """Recover the proposer-visible fields of a rendered row.

    Rendered-precision values (ar/lc at 2dp); absent or NA observation
    columns both decode as obs_available False (HZN-IR-08 F3 — the
    decoders must populate identical canonical structures). fam_legend
    maps famcode -> family. s3 requires column_order from the header.
    """
    inv = {c: f for f, c in fam_legend.items()}
    if level == "s2":
        fields = dict(p.split(":", 1) for p in line.split("|"))
        out = {"param_id": int(fields["O"]), "layer": int(fields["L"]),
               "family": inv.get("f" + fields["F"], fields["F"]),
               "numel": int(fields["N"])}
        if "M" in fields:
            out["mag"] = fields["M"]
        if "C" in fields:
            out["count_max"] = int(fields["C"])
        if fields.get("AR", "--") == "--":
            out["obs_available"] = False
        else:
            out["obs_available"] = True
            out["ar"] = int(fields["AR"]) / 100.0
            out["lc"] = int(fields["LC"]) / 100.0
        return out

    if level == "s3":
        vals = line.split("|")
        if column_order is None:
            raise ValueError("s3 decode requires the header column order")
        fields = dict(zip(column_order, vals))
        out = {"param_id": int(fields["id"]), "layer": int(fields["layer"]),
               "family": inv.get(fields["fam"], fields["fam"]),
               "numel": int(fields["numel"])}
        if fields.get("mag") not in (None, "-"):
            out["mag"] = fields["mag"]
        if fields.get("cnt") not in (None, "-"):
            out["count_max"] = int(fields["cnt"])
        if fields.get("ar") in (None, "-", "NA"):
            out["obs_available"] = False
        else:
            out["obs_available"] = True
            out["ar"] = float(fields["ar"])
            out["lc"] = float(fields["lc"])
        return out

    if level in ("linguistic", "s1"):
        sep = "|" if level == "linguistic" else None
        toks = line.split(sep) if sep else line.split()
        out = {"param_id": int(toks[0]),
               "layer": int(toks[1][1:]),
               "family": inv[toks[2]],
               "numel": int(toks[3][2:]),
               "obs_available": False}
        for t in toks[4:]:
            if t.startswith("m=") or t.startswith("mag["):
                out["mag"] = t
            elif t.startswith("c<="):
                out["count_max"] = int(t[3:])
            elif t.startswith("ar="):
                if t[3:] == "NA":
                    out["obs_available"] = False
                else:
                    out["obs_available"] = True
                    out["ar"] = float(t[3:])
            elif t.startswith("lc=") and t[3:] != "NA":
                out["lc"] = float(t[3:])
        return out

    raise ValueError(f"unknown channel level {level!r}")


def _canonicalize_decoded(d: dict, arm_mag=None,
                        arm_count_max: int = 512) -> dict:
    """Project a decode result onto the canonical record shape."""
    mag = d.get("mag")
    mag_pair = None
    if isinstance(mag, str):
        if mag.startswith("mag["):
            lo, hi = mag[4:-1].split(",")
            mag_pair = (float(lo), float(hi))
        elif mag.startswith("m="):
            c = float(mag[2:])
            mag_pair = (-c, c)
        elif "," in mag:
            lo, hi = mag.split(",")
            mag_pair = (float(lo), float(hi))
        else:
            c = float(mag)
            mag_pair = (-c, c)
    if mag_pair is None and arm_mag is not None:
        mag_pair = (float(f"{arm_mag[0]:.4g}"), float(f"{arm_mag[1]:.4g}"))
    return {
        "param_id": d["param_id"], "layer": d["layer"],
        "family": d["family"], "numel": d["numel"],
        "mag_min": mag_pair[0] if mag_pair else None,
        "mag_max": mag_pair[1] if mag_pair else None,
        "count_max": d.get("count_max", arm_count_max),
        "obs_available": bool(d.get("obs_available")),
        "ar": d.get("ar"), "lc": d.get("lc"),
    }


def verify_field_parity(catalog, fam_code, uniform_mag, obs, level,
                        arm_mag=None, arm_count=None,
                        arm_count_max: int = 512):
    """Canonical field-equivalence check (HZN-IR-08 F3).

    For every row, compare canonical_record(source) with
    canonicalize(decode(encode(source))) — the FULL proposer-visible
    semantics at rendered precision: param_id, layer, family, numel,
    effective magnitude interval (including uniform legend bounds),
    effective count cap, obs availability, ar, lc. Zero silent field
    omissions; legend-level fields are part of the record. For s3 the
    column order is taken from the RENDERED 'cols:' header — a header
    that mislabels its rows fails parity.
    """
    failures = []
    legend = {f: c for f, c in fam_code.items()}
    s3_cols = None
    if level == "s3":
        head = encode_catalog_text(catalog, fam_code, uniform_mag, obs,
                                   level, arm_mag, arm_count)
        hdr = next(l for l in head.splitlines() if l.startswith("cols:"))
        s3_cols = hdr[5:].strip().split("|")
    for e in catalog:
        line = encode_row(e, fam_code, uniform_mag, obs, level, s3_cols)
        d = _canonicalize_decoded(
            decode_row(line, legend, level, s3_cols), arm_mag,
            arm_count_max)
        want = canonical_record(e, uniform_mag, arm_mag, obs,
                                arm_count_max)
        for field, w in want.items():
            got = d.get(field)
            if w != got:
                failures.append((e["param_id"], field, w, got))
    return (not failures, failures)


def s3_column_order(catalog, uniform_mag: bool, obs: bool):
    cols = ["id", "layer", "fam", "numel"]
    if not uniform_mag:
        cols.append("mag")
    if any(e.get("count_max", 512) < 512 for e in catalog):
        cols.append("cnt")
    if obs:
        cols += ["ar", "lc"]
    return cols


def bounds_text(arm_mag=None, arm_count=None) -> str:
    """The 'bounds:' legend line, mirroring SelfProposer._catalog_text:
    uniform magnitude renders `mag[lo,hi]` once; non-uniform renders the
    per-row semantics sentence; the arm-wide count range always applies
    to rows without an explicit c<=N."""
    if arm_mag is not None:
        bounds = f"mag[{arm_mag[0]:.4g},{arm_mag[1]:.4g}]"
    else:
        bounds = "per-row m=cap means -cap<=m<=+cap"
    cnt = ""
    if arm_count is not None:
        cnt = f" count[{arm_count[0]},{arm_count[1]}] and <= numel"
    return f"bounds: {bounds}{cnt}\n"


def encode_catalog_text(catalog, fam_code, uniform_mag, obs, level,
                        arm_mag=None, arm_count=None) -> str:
    """families + bounds legends + rows, mirroring
    SelfProposer._catalog_text. s3 additionally emits a 'cols:' header
    carrying field names once (structural redundancy removed)."""
    legend = ", ".join(f"{c}={f}" for f, c in fam_code.items())
    head = f"families: {legend}\n" + bounds_text(arm_mag, arm_count)
    cols = None
    if level == "s3":
        cols = s3_column_order(catalog, uniform_mag, obs)
        head += "cols: " + "|".join(cols) + "\n"
    return (head + "\n".join(
        encode_row(e, fam_code, uniform_mag, obs, level, cols)
        for e in catalog))


# ---- Observation audit layer: O = (a,t,v,m,p) ----------------------

# sha256 of the empty edit chain — the provenance marker for observations
# collected on the pristine checkpoint before any candidate edit (epoch 0).
EMPTY_CHAIN_SHA = hashlib.sha256(b"[]").hexdigest()


def audit_record(param_name: str, obs_group: str, epoch: int,
                 stats: dict, *, base_checkpoint_sha: str,
                 observer_version: str,
                 applied_edit_chain_sha: str,
                 observation_context_sha: str,
                 measurement_config_sha: str,
                 edit_shas=None) -> dict:
    """Externally-established observation identity (HZN-IR-07 d7,
    HZN-IR-08 F4).

    All provenance fields are MANDATORY — an audit record that does not
    bind its measurement to a complete context is a defect, not a
    partial record. For baseline (epoch-0) collection pass
    applied_edit_chain_sha=EMPTY_CHAIN_SHA.

    a: component address (hooked module path + catalog param name)
    t: observation epoch (collection index; NOT asserted by proposer)
    v: measured values at stored precision
    m: method + context (what produced the numbers)
    p: provenance binding the observation to the ACTUAL model state —
       base checkpoint + applied-edit chain + measurement context +
       observer version + measurement config — so stale-vs-false is
       distinguishable. `edit_shas` (optional) are the per-edit hashes
       the chain commits to; they must identify the edits actually
       APPLIED (targets + values), not merely proposed instructions.
    """
    for name, v in (("base_checkpoint_sha", base_checkpoint_sha),
                    ("observation_context_sha", observation_context_sha),
                    ("measurement_config_sha", measurement_config_sha),
                    ("observer_version", observer_version),
                    ("applied_edit_chain_sha", applied_edit_chain_sha)):
        if not v or not isinstance(v, str):
            raise ValueError(f"provenance field {name} must be populated")
    return {
        "a": {"module": obs_group, "param": param_name},
        "t": epoch,
        "v": {k: stats.get(k, stats.get(f"obs_{k}"))
              for k in ("act", "hi", "lo", "hilo", "n_hi", "n_lo",
                      "var_hi", "var_lo", "log_contrast")
              if stats.get(k) is not None
              or stats.get(f"obs_{k}") is not None},
        "m": {"method": "forward_hook_rms_hilo",
              "n_hi": stats.get("n_hi"), "n_lo": stats.get("n_lo"),
              "eps": stats.get("eps")},
        "p": {"base_checkpoint_sha256": base_checkpoint_sha,
              "applied_edit_chain_sha256": applied_edit_chain_sha,
              "observation_context_sha256": observation_context_sha,
              "observer_version": observer_version,
              "measurement_config_sha256": measurement_config_sha,
              "applied_edit_shas": list(edit_shas) if edit_shas else None},
    }


def edit_chain_sha(applied_edit_shas) -> str:
    """Bind an observation to the exact modified state: sha256 of the
    ordered list of APPLIED-edit hashes (EMPTY_CHAIN_SHA at baseline).
    Each element must hash the effective edit — target + operation +
    value — not merely the proposal text."""
    shas = list(applied_edit_shas)
    if not shas:
        return EMPTY_CHAIN_SHA
    return hashlib.sha256(json.dumps(shas).encode()).hexdigest()


def audit_sha(rec: dict) -> str:
    """Canonical hash of an audit record for the observation ledger."""
    return hashlib.sha256(
        json.dumps(rec, sort_keys=True).encode()).hexdigest()


# ---- Tokenizer profiling ------------------------------------------------

def token_count(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return int(ids.shape[-1]) if hasattr(ids, "shape") else len(ids)


def profile_levels(tokenizer, catalog, fam_code, uniform_mag, obs=True,
                   arm_mag=None, arm_count=None):
    """Measured token cost per channel level on a real catalog.

    Returns per-level: total catalog-text tokens (families + bounds
    legends + any header + rows), per-row mean, ratio vs linguistic.
    Character counts are reported for contrast only — the decision
    metric is tokens (spec §2 tokenizer caution; HZN-IR-08 d9: profile
    complete proposer surfaces, legends included).
    """
    out = {}
    texts = {}
    for level in CHANNEL_LEVELS:
        txt = encode_catalog_text(catalog, fam_code, uniform_mag, obs,
                                  level, arm_mag, arm_count)
        texts[level] = txt
        n_tok = token_count(tokenizer, txt)
        out[level] = {"tokens": n_tok, "chars": len(txt),
                      "tokens_per_row": n_tok / len(catalog)}
    base = out["linguistic"]["tokens"]
    for level, r in out.items():
        r["token_ratio_vs_linguistic"] = r["tokens"] / base
    return out, texts
