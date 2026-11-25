"""HZN-OC-01 regression tests: channel encodings + comprehension harness."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jarvis_research.gate1_3_channel import (
    CHANNEL_LEVELS, audit_record, audit_sha, decode_row, encode_row,
    encode_catalog_text, token_count, verify_field_parity)
from jarvis_research.gate1_3_comprehension import (
    generate_items, grade, make_synthetic_catalog, pretest_passed,
    score_items)


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


class TestEncodings(unittest.TestCase):
    catalog = [_row(0), _row(7, family="mlp.up_proj", layer=11, ar=0.5,
                             lc=1.23),
               _row(12, family="linear_attn.in_proj_a", layer=0, ar=None,
                    lc=None, avail=False, count_max=256),
               _row(30, layer=21, ar=0.33, lc=2.5)]

    def test_field_parity_all_levels(self):
        for level in CHANNEL_LEVELS:
            with self.subTest(level=level):
                ok, failures = verify_field_parity(
                    self.catalog, FAMS, True, True, level)
                self.assertTrue(ok, failures)

    def test_field_parity_noobs(self):
        ok, failures = verify_field_parity(
            self.catalog, FAMS, True, False, "s2")
        self.assertTrue(ok, failures)

    def test_s2_integer_obs_coding(self):
        # lc=-0.03 -> LC:-3 ; lc=+1.23 -> LC:+123 ; ar=0.82 -> AR:82
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
                       {f: c for f, c in FAMS.items()}, "s2")
        self.assertEqual(d["param_id"], 7)
        self.assertEqual(d["layer"], 11)
        self.assertEqual(d["family"], "mlp.up_proj")
        self.assertAlmostEqual(d["ar"], 0.5)
        self.assertAlmostEqual(d["lc"], 1.23)

    def test_parity_catches_lossy_encoding(self):
        # A deliberately lossy level: truncating lc to 1dp must FAIL parity.
        import jarvis_research.gate1_3_channel as ch
        orig = ch.encode_row
        def lossy(e, fc, um, obs, level):
            e2 = dict(e)
            if e2.get("obs_available"):
                e2["obs_log_contrast"] = round(e2["obs_log_contrast"], 1)
            return orig(e2, fc, um, obs, "linguistic")
        ch.encode_row = lossy
        try:
            bad = _row(9, lc=0.07)  # rounds to 0.1 — parity must fail
            ok, failures = verify_field_parity(
                [bad], FAMS, True, True, "linguistic")
            self.assertFalse(ok)
            self.assertTrue(any(f[1] == "lc" for f in failures))
        finally:
            ch.encode_row = orig

    def test_count_max_parity(self):
        ok, failures = verify_field_parity(self.catalog, FAMS, True, True, "s1")
        self.assertTrue(ok, failures)


class TestAuditLayer(unittest.TestCase):
    def test_record_shape(self):
        rec = audit_record(
            "model.layers.5.mlp.down_proj.weight",
            "model.layers.5.mlp.down_proj", 0,
            {"act": 1.2, "hi": 1.3, "lo": 1.1, "hilo": 1.18,
             "n_hi": 8, "n_lo": 8, "var_hi": 0.01, "var_lo": 0.02,
             "eps": 0.0075},
            "deadbeef" * 8, "obs-v1")
        self.assertEqual(rec["a"]["module"], "model.layers.5.mlp.down_proj")
        self.assertEqual(rec["t"], 0)
        self.assertEqual(rec["m"]["method"], "forward_hook_rms_hilo")
        self.assertEqual(rec["p"]["checkpoint_sha256"], "deadbeef" * 8)
        self.assertIn("hilo", rec["v"])

    def test_sha_stable(self):
        rec = audit_record("p", "g", 1, {"n_hi": 8}, "sha", "v")
        self.assertEqual(audit_sha(rec), audit_sha(dict(rec)))


class StubTokenizer:
    """Whitespace tokenizer: 1 token per word — checks plumbing only."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class TestProfiling(unittest.TestCase):
    def test_token_count_and_profile(self):
        from jarvis_research.gate1_3_channel import profile_levels
        cat = [_row(i, layer=i) for i in range(5)]
        self.assertEqual(token_count(StubTokenizer(), "a b c"), 3)
        out, texts = profile_levels(StubTokenizer(), cat, FAMS, True, True)
        for lvl in CHANNEL_LEVELS:
            self.assertIn(lvl, out)
            self.assertIn("token_ratio_vs_linguistic", out[lvl])
        self.assertEqual(out["linguistic"]["token_ratio_vs_linguistic"], 1.0)


class TestComprehension(unittest.TestCase):
    def test_synthetic_catalog_unique_grounding(self):
        cat = make_synthetic_catalog(0)
        pairs = [(r["family"], r["layer"]) for r in cat]
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_generated_items_have_verifiable_answers(self):
        for level in CHANNEL_LEVELS:
            cat = make_synthetic_catalog(1)
            fam_code = {f: f"f{i}" for i, f in
                        enumerate(sorted({r["family"] for r in cat}))}
            items = generate_items(cat, fam_code, level, 1)
            self.assertTrue(any(i["level"] == "C1" for i in items))
            self.assertTrue(any(i["level"] == "C4" for i in items))
            for it in items:
                if it["level"] == "C2":
                    # Expected id must be the row with strictly greater |lc|.
                    import re
                    ids = [int(m) for m in
                           re.findall(r"O:(\d+)|^(\d+)\|", it["context"],
                                      re.M) for m in m if m]
                    rows = [r for r in cat if r["param_id"] in ids]
                    if len(rows) == 2:
                        a, b = rows
                        winner = (a if abs(a["obs_log_contrast"]) >
                                  abs(b["obs_log_contrast"]) else b)
                        self.assertEqual(it["expected"],
                                         str(winner["param_id"]))
                if it["level"] == "C4":
                    # Expected id resolves to a row whose (family, layer)
                    # appears verbatim in the query text.
                    row = [r for r in cat
                           if str(r["param_id"]) == it["expected"]]
                    self.assertEqual(len(row), 1)
                    self.assertIn(f"layer {row[0]['layer']}", it["query"])

    def test_grade(self):
        self.assertTrue(grade({"expected": "17"}, "The answer is 17."))
        self.assertTrue(grade({"expected": "-0.03"}, "lc is -0.03"))
        self.assertTrue(grade({"expected": [2, 5, 9]}, "[5, 2, 9]"))
        self.assertFalse(grade({"expected": "17"}, "answer: 18"))
        self.assertFalse(grade({"expected": [1, 2]}, "1, 2"))

    def test_score_and_threshold(self):
        items = [{"level": "C1", "expected": "3"},
                 {"level": "C1", "expected": "5"},
                 {"level": "C4", "expected": "9"}]
        scores = score_items(items, ["3", "5", "9"])
        self.assertEqual(scores["C1"]["acc"], 1.0)
        self.assertTrue(pretest_passed(scores))
        scores = score_items(items, ["3", "wrong 99", "9"])
        self.assertAlmostEqual(scores["C1"]["acc"], 0.5)
        self.assertFalse(pretest_passed(scores))


if __name__ == "__main__":
    unittest.main()
