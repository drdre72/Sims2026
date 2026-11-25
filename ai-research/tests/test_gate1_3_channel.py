"""HZN-OC-01 regression tests: channel encodings + comprehension harness.

Includes the HZN-IR-08 required regression set: fail-closed gates,
strict grading, magnitude-interval parity, observation availability,
mandatory provenance, prefixed-stat normalization, ambiguous grounding,
near-zero/rounded-tie catalogs.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jarvis_research.gate1_3_channel import (
    CHANNEL_LEVELS, EMPTY_CHAIN_SHA, audit_record, audit_sha,
    canonical_record, decode_row, edit_chain_sha, encode_row,
    encode_catalog_text, token_count, verify_field_parity)
from jarvis_research.gate1_3_comprehension import (
    MIN_PROBES, PRETEST_ORDER, c4_coverage, gen_items, grade,
    grade_c1_approx, grade_integer, grade_list, pretest_passed,
    score_items, synthetic_catalog)
from jarvis_research.gate1_3_store import ObservationStore, catalog_sha256


def _row(pid, family="mlp.down_proj", layer=5, ar=0.82, lc=-0.03,
         avail=True, count_max=512, mag=(-0.005, 0.005)):
    return {
        "param_id": pid, "param": f"model.layers.{layer}.{family}.w",
        "family": family, "layer": layer, "numel": 147456,
        "mag_min": mag[0], "mag_max": mag[1], "count_max": count_max,
        "obs_available": avail,
        "obs_act_rank": ar, "obs_log_contrast": lc,
    }


FAMS = {"mlp.down_proj": "f0", "mlp.up_proj": "f1",
        "linear_attn.in_proj_a": "f2"}
ARM_MAG = (-0.005, 0.005)
ARM_COUNT = (1, 512)


class TestEncodings(unittest.TestCase):
    catalog = [_row(0), _row(7, family="mlp.up_proj", layer=11, ar=0.5,
                             lc=1.23),
               _row(12, family="linear_attn.in_proj_a", layer=0,
                    avail=False, count_max=256),
               _row(30, layer=21, ar=0.33, lc=2.5)]

    def test_field_parity_all_levels(self):
        for level in CHANNEL_LEVELS:
            with self.subTest(level=level):
                ok, failures = verify_field_parity(
                    self.catalog, FAMS, True, True, level,
                    arm_mag=ARM_MAG, arm_count=ARM_COUNT)
                self.assertTrue(ok, failures)

    def test_field_parity_nonuniform_mag(self):
        # Arm A regime: per-row m=cap — the magnitude interval MUST be
        # verified through the canonical record, not ignored (F3).
        cat = [_row(0, mag=(-0.004588, 0.004588)),
               _row(1, mag=(-0.001, 0.004588))]
        for level in CHANNEL_LEVELS:
            with self.subTest(level=level):
                ok, failures = verify_field_parity(
                    cat, FAMS, False, True, level,
                    arm_count=ARM_COUNT)
                self.assertTrue(ok, failures)

    def test_parity_catches_wrong_magnitude(self):
        # A decoder that returns the wrong magnitude interval must FAIL
        # parity — magnitude is part of the intervention surface.
        import jarvis_research.gate1_3_channel as ch
        orig = ch.decode_row
        def bad_decode(line, fam_legend, level, column_order=None):
            d = orig(line, fam_legend, level, column_order)
            if "mag" in d:
                d["mag"] = "mag[-9,9]"
            return d
        ch.decode_row = bad_decode
        try:
            cat = [_row(0, mag=(-0.002, 0.004))]
            ok, failures = verify_field_parity(
                cat, FAMS, False, True, "linguistic")
            self.assertFalse(ok)
            self.assertTrue(any(f[1] in ("mag_min", "mag_max")
                                for f in failures))
        finally:
            ch.decode_row = orig

    def test_parity_catches_lossy_encoding(self):
        import jarvis_research.gate1_3_channel as ch
        orig = ch.encode_row
        def lossy(e, fc, um, obs, level, column_order=None):
            e2 = dict(e)
            if e2.get("obs_available"):
                e2["obs_log_contrast"] = round(e2["obs_log_contrast"], 1)
            return orig(e2, fc, um, obs, "linguistic")
        ch.encode_row = lossy
        try:
            bad = _row(9, lc=0.07)
            ok, failures = verify_field_parity(
                [bad], FAMS, True, True, "linguistic", arm_mag=ARM_MAG)
            self.assertFalse(ok)
            self.assertTrue(any(f[1] == "lc" for f in failures))
        finally:
            ch.encode_row = orig

    def test_s2_integer_obs_coding(self):
        line = encode_row(self.catalog[0], FAMS, True, True, "s2")
        self.assertIn("AR:82", line)
        self.assertIn("LC:-3", line)
        big = encode_row(self.catalog[1], FAMS, True, True, "s2")
        self.assertIn("LC:+123", big)
        na = encode_row(self.catalog[2], FAMS, True, True, "s2")
        self.assertIn("AR:--", na)
        self.assertIn("LC:--", na)

    def test_s2_decode_roundtrip(self):
        d = decode_row(encode_row(self.catalog[1], FAMS, True, True, "s2"),
                       dict(FAMS), "s2")
        self.assertEqual(d["param_id"], 7)
        self.assertEqual(d["layer"], 11)
        self.assertEqual(d["family"], "mlp.up_proj")
        self.assertAlmostEqual(d["ar"], 0.5)
        self.assertAlmostEqual(d["lc"], 1.23)

    def test_absent_obs_fields_consistent(self):
        # Observation columns entirely absent (obs=False) and per-row
        # unavailability must BOTH decode to obs_available=False — all
        # encodings decode to identical canonical structures (F3).
        noobs = [_r for _r in self.catalog]
        for level in CHANNEL_LEVELS:
            with self.subTest(level=level):
                ok, failures = verify_field_parity(
                    noobs, FAMS, True, False, level, arm_mag=ARM_MAG)
                self.assertTrue(ok, failures)
                line = encode_row(self.catalog[0], FAMS, True, False, level)
                d = decode_row(line, dict(FAMS), level,
                               column_order=None if level != "s3" else
                               ["id", "layer", "fam", "numel"])
                self.assertFalse(d.get("obs_available"))

    def test_s3_headered_roundtrip(self):
        # The rendered cols: header must match the positional rows.
        txt = encode_catalog_text(self.catalog, FAMS, True, True, "s3",
                                  arm_mag=ARM_MAG, arm_count=ARM_COUNT)
        lines = txt.splitlines()
        self.assertTrue(lines[0].startswith("families:"))
        self.assertTrue(lines[1].startswith("bounds:"))
        self.assertTrue(lines[2].startswith("cols:"))
        self.assertEqual(lines[2], "cols: id|layer|fam|numel|cnt|ar|lc")
        # Row carries values only — no field names per row.
        self.assertNotIn("ar=", lines[3])
        self.assertNotIn("n=", lines[3])


class TestAuditLayer(unittest.TestCase):
    KW = {"base_checkpoint_sha": "deadbeef" * 8,
          "observer_version": "obs-v1",
          "applied_edit_chain_sha": EMPTY_CHAIN_SHA,
          "observation_context_sha": "ctx",
          "measurement_config_sha": "cfg"}

    def test_record_shape(self):
        rec = audit_record(
            "model.layers.5.mlp.down_proj.weight",
            "model.layers.5.mlp.down_proj", 0,
            {"act": 1.2, "hi": 1.3, "lo": 1.1, "hilo": 1.18,
             "n_hi": 8, "n_lo": 8, "var_hi": 0.01, "var_lo": 0.02,
             "eps": 0.0075}, **self.KW)
        self.assertEqual(rec["a"]["module"], "model.layers.5.mlp.down_proj")
        self.assertEqual(rec["t"], 0)
        self.assertEqual(rec["m"]["method"], "forward_hook_rms_hilo")
        p = rec["p"]
        self.assertEqual(p["base_checkpoint_sha256"], "deadbeef" * 8)
        self.assertEqual(p["applied_edit_chain_sha256"], EMPTY_CHAIN_SHA)
        self.assertEqual(p["observer_version"], "obs-v1")
        self.assertIn("hilo", rec["v"])

    def test_missing_provenance_fails(self):
        # F4: an audit record with empty provenance must not construct.
        for field in ("base_checkpoint_sha", "observation_context_sha",
                      "measurement_config_sha", "observer_version",
                      "applied_edit_chain_sha"):
            kw = dict(self.KW)
            kw[field] = ""
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    audit_record("p", "g", 0, {}, **kw)

    def test_prefixed_stats_normalized(self):
        # F4: obs_-prefixed statistics must land on the right keys,
        # not be retrieved as None.
        rec = audit_record("p", "g", 0,
                           {"obs_act": 1.2, "obs_log_contrast": -0.03,
                            "n_hi": 8}, **self.KW)
        self.assertEqual(rec["v"]["act"], 1.2)
        self.assertEqual(rec["v"]["log_contrast"], -0.03)

    def test_sha_stable(self):
        rec = audit_record("p", "g", 1, {"n_hi": 8}, **self.KW)
        self.assertEqual(audit_sha(rec), audit_sha(dict(rec)))

    def test_edit_chain_binding(self):
        self.assertEqual(edit_chain_sha([]), EMPTY_CHAIN_SHA)
        a = edit_chain_sha(["e1", "e2"])
        b = edit_chain_sha(["e1", "e2"])
        c = edit_chain_sha(["e2", "e1"])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        rec = audit_record("p", "g", 3, {}, applied_edit_chain_sha=a,
                           **{k: v for k, v in self.KW.items()
                              if k != "applied_edit_chain_sha"})
        self.assertEqual(rec["p"]["applied_edit_chain_sha256"], a)


class StubTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class TestProfiling(unittest.TestCase):
    def test_token_count_and_profile(self):
        from jarvis_research.gate1_3_channel import profile_levels
        cat = [_row(i, layer=i) for i in range(5)]
        self.assertEqual(token_count(StubTokenizer(), "a b c"), 3)
        out, texts = profile_levels(StubTokenizer(), cat, FAMS, True, True,
                                    arm_mag=ARM_MAG, arm_count=ARM_COUNT)
        for lvl in CHANNEL_LEVELS:
            self.assertIn(lvl, out)
            self.assertIn("token_ratio_vs_linguistic", out[lvl])
            self.assertIn("bounds:", texts[lvl])
        self.assertEqual(out["linguistic"]["token_ratio_vs_linguistic"], 1.0)


class TestComprehension(unittest.TestCase):
    def test_synthetic_catalog_realistic(self):
        cat = synthetic_catalog(248, seed=0)
        self.assertEqual(len(cat), 248)
        # Near-zero mass: most contrasts within +/-0.05.
        nz = sum(1 for r in cat
                 if abs(r["obs_log_contrast"]) <= 0.05)
        self.assertGreater(nz / len(cat), 0.4)
        # Repeated (family, layer) pairs exist — real ambiguity.
        from collections import Counter
        pairs = Counter((r["family"], r["layer"]) for r in cat)
        self.assertTrue(any(v > 1 for v in pairs.values()))
        # Rounded ties exist in lc.
        lcs = Counter(round(r["obs_log_contrast"], 2) for r in cat)
        self.assertTrue(any(v > 1 for v in lcs.values()))

    def test_strict_grading(self):
        # F2 — these all FAIL under whole-response grading:
        self.assertFalse(grade_integer(7, "7.9"))
        self.assertFalse(grade_integer(7, "Not 7. The answer is 9."))
        self.assertFalse(grade_integer(7, "7 or 12"))
        self.assertTrue(grade_integer(7, "7"))
        self.assertFalse(grade_integer(7, "17"))
        self.assertTrue(grade_list([2, 5, 9], "[5, 2, 9]"))
        self.assertFalse(grade_list([2, 5, 9], "[2.0, 5, 9]"))
        self.assertFalse(grade_list([1, 2], "1, 2"))
        self.assertFalse(grade_list([1, 2], '["1", "2"]'))
        # C4 ambiguous probes grade the refusal word exactly.
        self.assertTrue(grade("C4", "ambiguous", "ambiguous"))
        self.assertFalse(grade("C4", "ambiguous", "7"))

    def test_generated_items_verifiable(self):
        cat = synthetic_catalog(64, seed=1)
        fam_code = {f: f"f{i}" for i, f in
                    enumerate(sorted({r["family"] for r in cat}))}
        for level in CHANNEL_LEVELS:
            items = gen_items(cat, fam_code, True, level, n_per=6, seed=1)
            self.assertTrue(items["C1"])
            self.assertTrue(items["C4"])
            # C2 expected id must be the strictly-larger-|lc| row.
            from collections import defaultdict
            by_id = {r["param_id"]: r for r in cat}
            for prompt, exp in items["C2"]:
                ids = [i for i in by_id
                       if str(i) in prompt or f"O:{i}" in prompt]
                if len(ids) == 2:
                    a, b = (by_id[i] for i in ids)
                    winner = (a if abs(a["obs_log_contrast"]) >
                              abs(b["obs_log_contrast"]) else b)
                    self.assertEqual(exp, winner["param_id"])
            # C4 ambiguous prompts must name a REPEATED pair.
            pairs = defaultdict(list)
            for r in cat:
                pairs[(r["family"], r["layer"])].append(r["param_id"])
            for prompt, exp in items["C4"]:
                if exp == "ambiguous":
                    self.assertIn("layer", prompt)

    def test_gate_fails_closed(self):
        # F1 — every failure mode must fail, none may pass vacuously.
        self.assertFalse(pretest_passed({}))
        self.assertFalse(pretest_passed(
            {"C1": {"n": MIN_PROBES["C1"], "correct": MIN_PROBES["C1"],
                    "acc": 1.0}}))
        # All levels present but under minimum probe count -> fail.
        sparse = {lvl: {"n": 1, "correct": 1, "acc": 1.0}
                  for lvl in PRETEST_ORDER}
        self.assertFalse(pretest_passed(sparse))
        # Full counts, perfect -> pass.
        full = {lvl: {"n": MIN_PROBES[lvl] + 1,
                      "correct": MIN_PROBES[lvl] + 1, "acc": 1.0}
                for lvl in PRETEST_ORDER}
        self.assertTrue(pretest_passed(full))
        # One C4 miss at full count -> fail (zero-failure requirement).
        c4miss = dict(full)
        n = MIN_PROBES["C4"] + 1
        c4miss["C4"] = {"n": n, "correct": n - 1, "acc": (n - 1) / n}
        self.assertFalse(pretest_passed(c4miss))

    def test_missing_responses_fail_closed(self):
        # Fewer responses than items: missing count as wrong, never
        # silently shorten the denominator.
        items = {"C4": [("q1", 7), ("q2", 9), ("q3", 3)]}
        scores = score_items(items, {"C4": ["7"]})
        self.assertEqual(scores["C4"]["n"], 3)
        self.assertEqual(scores["C4"]["correct"], 1)
        # Missing level entirely -> zero items answered.
        scores = score_items(items, {})
        self.assertEqual(scores["C4"]["correct"], 0)


class TestIR09Fixes(unittest.TestCase):
    def test_f1_c2_uses_absolute_lc_tie(self):
        # lc=-0.05 vs +0.05 have equal |lc| — must never become a C2 item.
        cat = [_row(0, lc=-0.05), _row(1, lc=0.05)]
        items = gen_items(cat, {"mlp.down_proj": "f0"}, True, "linguistic",
                          n_per=4, seed=0, counts={"C1": 0, "C2": 4,
                                                   "C3": 0, "C4": 0})
        self.assertEqual(items["C2"], [])

    def test_f2_c3_no_boundary_ties(self):
        cat = synthetic_catalog(128, seed=7)
        fams = {f: f"f{i}" for i, f in
                enumerate(sorted({r["family"] for r in cat}))}
        items = gen_items(cat, fams, True, "linguistic", seed=7,
                          counts={"C1": 0, "C2": 0, "C3": 20, "C4": 0})
        self.assertEqual(len(items["C3"]), 20)
        by_id = {r["param_id"]: r for r in cat}
        for prompt, exp in items["C3"]:
            # Recover the 12-row pool from the rendered rows (leading
            # field of each linguistic row is the param_id).
            pool_ids = [int(l.split("|")[0]) for l in prompt.splitlines()]
            self.assertEqual(len(pool_ids), 12)
            vals = sorted((abs(by_id[i]["obs_log_contrast"])
                           for i in pool_ids), reverse=True)
            self.assertNotAlmostEqual(vals[2], vals[3], places=9)

    def test_f3_s3_rows_carry_header(self):
        cat = synthetic_catalog(64, seed=1)
        fams = {f: f"f{i}" for i, f in
                enumerate(sorted({r["family"] for r in cat}))}
        items = gen_items(cat, fams, True, "s3", seed=1)
        for lvl in ("C1", "C2", "C3"):
            for prompt, _ in items[lvl]:
                self.assertTrue(prompt.startswith("cols:"), lvl)

    def test_f6_c1_exact_decoding(self):
        exp = {"ar": 0.48, "lc": 0.05}
        self.assertTrue(grade("C1", exp, '{"ar": 0.48, "lc": 0.05}'))
        self.assertFalse(grade("C1", exp, '{"ar": 0.47, "lc": 0.05}'))
        self.assertTrue(grade_c1_approx(exp, '{"ar": 0.47, "lc": 0.05}'))

    def test_f7_permuted_ids_and_order(self):
        cat = synthetic_catalog(64, seed=3)
        ids = [r["param_id"] for r in cat]
        self.assertEqual(sorted(ids), list(range(64)))
        self.assertNotEqual(ids, sorted(ids))

    def test_f5_coverage_requires_distinct_catalogs(self):
        a = {"catalog_sha256": "x", "seed": 1}
        b = {"catalog_sha256": "y", "seed": 2}
        c = {"catalog_sha256": "z", "seed": 3}
        self.assertTrue(c4_coverage([a, b, c]))
        self.assertFalse(c4_coverage([a, a, c]))
        self.assertFalse(c4_coverage([a, b]))


class TestStore(unittest.TestCase):
    def setUp(self):
        self.recs = [
            {"param_id": 0, "layer": 5, "family": "mlp.down_proj",
             "numel": 147456, "mag_min": -0.005, "mag_max": 0.005,
             "count_max": 512, "obs_available": True,
             "ar": 0.8, "lc": -0.05},
            {"param_id": 1, "layer": 5, "family": "mlp.down_proj",
             "numel": 147456, "mag_min": -0.005, "mag_max": 0.005,
             "count_max": 512, "obs_available": True,
             "ar": 0.5, "lc": 0.31},
            {"param_id": 2, "layer": 5, "family": "attn.q_proj",
             "numel": 81920, "mag_min": -0.005, "mag_max": 0.005,
             "count_max": 512, "obs_available": True,
             "ar": 0.2, "lc": 0.80},
            {"param_id": 3, "layer": 9, "family": "attn.q_proj",
             "numel": 81920, "mag_min": -0.005, "mag_max": 0.005,
             "count_max": 256, "obs_available": False,
             "ar": None, "lc": None},
        ]
        self.store = ObservationStore(self.recs)

    def test_resolve_typed_verdicts(self):
        self.assertEqual(
            self.store.resolve("attn.q_proj", 9),
            {"status": "unique", "param_id": 3})
        self.assertEqual(
            self.store.resolve("mlp.down_proj", 5)["status"], "ambiguous")
        amb = self.store.resolve("mlp.down_proj", 5)
        self.assertEqual(amb["param_ids"], [0, 1])
        self.assertEqual(self.store.resolve("mlp.down_proj", 7)["status"],
                         "missing")

    def test_find_exact_filters(self):
        self.assertEqual([r["param_id"] for r in
                          self.store.find(family="mlp.down_proj")],
                         [0, 1])
        self.assertEqual([r["param_id"] for r in
                          self.store.find(min_abs_lc=0.1)], [1, 2])
        self.assertEqual(self.store.find(layer=9, min_abs_lc=0.1), [])

    def test_top_abs_lc(self):
        top = self.store.top_abs_lc(2)
        self.assertEqual([r["param_id"] for r in top], [2, 1])

    def test_catalog_identity(self):
        self.assertEqual(catalog_sha256(self.recs),
                         catalog_sha256(list(reversed(self.recs))))
        mod = [dict(r, param_id=r["param_id"] + 10) for r in self.recs]
        self.assertNotEqual(catalog_sha256(self.recs),
                            catalog_sha256(mod))


if __name__ == "__main__":
    unittest.main()
