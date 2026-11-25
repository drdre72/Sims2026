"""HZN-OC-01 observation-channel encodings, audit identity, token profiling.

Exploratory component set (reviewer-authorized, spec v2): candidate
renderings of the proposer-visible catalog at identical field precision,
a mechanical field-parity check (decode(encode(row)) must return every
proposer-visible field), the O=(a,t,v,m,p) observation-identity record,
and tokenizer-measured (never character-measured) cost profiling.

Channel levels:
  linguistic — the Gate 1.2 hybrid pipe schema (baseline)
  s1         — space-delimited compact schema, familiar key=value
  s2         — fixed symbolic vocabulary, integer-coded obs fields
"""

import hashlib
import json
import math


CHANNEL_LEVELS = ("linguistic", "s1", "s2")


def _obs_fields(entry: dict):
    """The two proposer-visible evidence fields at rendered precision."""
    if not entry.get("obs_available"):
        return None, None
    return (round(entry["obs_act_rank"], 2),
            round(entry["obs_log_contrast"], 2))


def _mag_field(entry: dict, uniform_mag: bool):
    if uniform_mag:
        return None
    if abs(entry["mag_min"] + entry["mag_max"]) < 1e-12:
        return entry["mag_max"] * 0.999
    return (entry["mag_min"] * 0.999, entry["mag_max"] * 0.999)


def encode_row(entry: dict, fam_code: dict, uniform_mag: bool,
               obs: bool, level: str) -> str:
    """Render one catalog row in the given channel level."""
    pid, layer = entry["param_id"], entry["layer"]
    fc = fam_code[entry["family"]]
    numel = entry["numel"]
    mag = _mag_field(entry, uniform_mag)
    cnt = entry.get("count_max") if entry.get("count_max", 512) < 512 else None
    ar, lc = _obs_fields(entry) if obs else (None, None)

    if level == "linguistic":
        line = f"{pid}|L{layer}|{fc}|n={numel}"
        if isinstance(mag, tuple):
            line += f"|mag[{mag[0]:.4g},{mag[1]:.4g}]"
        elif mag is not None:
            line += f"|m={mag:.4g}"
        if cnt is not None:
            line += f"|c<={cnt}"
        if obs:
            line += (f"|ar={ar:.2f}|lc={lc:+.2f}" if ar is not None
                     else "|ar=NA|lc=NA")
        return line

    if level == "s1":
        parts = [str(pid), f"L{layer}", fc, f"n={numel}"]
        if isinstance(mag, tuple):
            parts.append(f"mag[{mag[0]:.4g},{mag[1]:.4g}]")
        elif mag is not None:
            parts.append(f"m={mag:.4g}")
        if cnt is not None:
            parts.append(f"c<={cnt}")
        if obs:
            parts += ([f"ar={ar:.2f}", f"lc={lc:+.2f}"] if ar is not None
                      else ["ar=NA", "lc=NA"])
        return " ".join(parts)

    if level == "s2":
        parts = [f"O:{pid}", f"L:{layer}", f"F:{fc[1:]}", f"N:{numel}"]
        if isinstance(mag, tuple):
            parts.append(f"M:{mag[0]:.4g},{mag[1]:.4g}")
        elif mag is not None:
            parts.append(f"M:{mag:.4g}")
        if cnt is not None:
            parts.append(f"C:{cnt}")
        if obs:
            if ar is None:
                parts += ["AR:--", "LC:--"]
            else:
                parts += [f"AR:{int(round(ar * 100))}",
                          f"LC:{int(round(lc * 100)):+d}"]
        return "|".join(parts)

    raise ValueError(f"unknown channel level {level!r}")


def decode_row(line: str, fam_legend: dict, level: str) -> dict:
    """Recover the proposer-visible fields of a rendered row.

    Returns rendered-precision values (ar at 2dp, lc at 2dp); NA obs
    decodes as obs_available False. fam_legend maps famcode -> family.
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

    if level in ("linguistic", "s1"):
        sep = "|" if level == "linguistic" else None
        toks = line.split(sep) if sep else line.split()
        out = {"param_id": int(toks[0]),
               "layer": int(toks[1][1:]),
               "family": inv[toks[2]],
               "numel": int(toks[3][2:])}
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


def verify_field_parity(catalog, fam_code, uniform_mag, obs, level):
    """Every proposer-visible field must survive encode->decode exactly.

    Field-preservation is a mechanical invariant, not a model judgment
    (spec §6 acceptance 1). Returns (ok, failures) where failures lists
    (param_id, field, expected, got).
    """
    failures = []
    legend = {f: c for f, c in fam_code.items()}
    for e in catalog:
        line = encode_row(e, fam_code, uniform_mag, obs, level)
        d = decode_row(line, legend, level)
        want_ar, want_lc = _obs_fields(e) if obs else (None, None)
        checks = [
            ("param_id", e["param_id"], d.get("param_id")),
            ("layer", e["layer"], d.get("layer")),
            ("family", e["family"], d.get("family")),
            ("numel", e["numel"], d.get("numel")),
            ("obs_available", bool(want_ar is not None),
             d.get("obs_available")),
        ]
        if obs and want_ar is not None:
            checks += [("ar", want_ar, round(d.get("ar", -9), 2)),
                       ("lc", want_lc, round(d.get("lc", -9), 2))]
        if e.get("count_max", 512) < 512:
            checks.append(("count_max", e["count_max"], d.get("count_max")))
        for field, want, got in checks:
            if want != got:
                failures.append((e["param_id"], field, want, got))
    return (not failures, failures)


def encode_catalog_text(catalog, fam_code, uniform_mag, obs, level) -> str:
    """Legend + rows, mirroring SelfProposer._catalog_text structure."""
    legend = ", ".join(f"{c}={f}" for f, c in fam_code.items())
    return (f"families: {legend}\n"
            + "\n".join(encode_row(e, fam_code, uniform_mag, obs, level)
                        for e in catalog))


# ---- Observation audit layer: O = (a, t, v, m, p) ----------------------

def audit_record(param_name: str, obs_group: str, epoch: int,
                 stats: dict, checkpoint_sha: str,
                 obs_version: str) -> dict:
    """Externally-established observation identity.

    a: component address (hooked module path + catalog param name)
    t: observation epoch (collection index; NOT asserted by proposer)
    v: measured values at stored precision
    m: method + context (what produced the numbers)
    p: provenance — the model state the observation describes
    """
    return {
        "a": {"module": obs_group, "param": param_name},
        "t": epoch,
        "v": {k: stats.get(k) for k in
              ("act", "hi", "lo", "hilo", "n_hi", "n_lo",
               "var_hi", "var_lo", "log_contrast")
              if k in stats or f"obs_{k}" in stats},
        "m": {"method": "forward_hook_rms_hilo",
              "n_hi": stats.get("n_hi"), "n_lo": stats.get("n_lo"),
              "eps": stats.get("eps")},
        "p": {"checkpoint_sha256": checkpoint_sha,
              "observation_version": obs_version},
    }


def audit_sha(record: dict) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()


# ---- Tokenizer profiling ------------------------------------------------

def token_count(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return int(ids.shape[-1]) if hasattr(ids, "shape") else len(ids)


def profile_levels(tokenizer, catalog, fam_code, uniform_mag, obs=True):
    """Measured token cost per channel level on a real catalog.

    Returns per-level: total catalog-text tokens, per-row mean, ratio vs
    linguistic. Character counts are reported for contrast only — the
    decision metric is tokens (spec §2 tokenizer caution).
    """
    out = {}
    texts = {}
    for level in CHANNEL_LEVELS:
        txt = encode_catalog_text(catalog, fam_code, uniform_mag, obs, level)
        texts[level] = txt
        n_tok = token_count(tokenizer, txt)
        out[level] = {"tokens": n_tok, "chars": len(txt),
                      "tokens_per_row": n_tok / len(catalog)}
    base = out["linguistic"]["tokens"]
    for level, r in out.items():
        r["token_ratio_vs_linguistic"] = r["tokens"] / base
    return out, texts
