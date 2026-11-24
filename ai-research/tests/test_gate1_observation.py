"""Gate 1.2 observation-layer tests — CPU-only synthetic validation."""
import json
import math
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from jarvis_research.gate1_observation import (  # noqa: E402
    resolve_hook_targets, stratified_split, annotate_catalog,
    shuffle_observations, interpreter_weights, collect_observations,
    INTERPRETER_FLOOR, LOG_CONTRAST_DECIMALS)
from jarvis_research.gate1_proposer import (  # noqa: E402
    parse_proposal, _catalog_line, PROMPT, PROMPT_OBS)
import jarvis_research.gate1_capability as _cap  # noqa: E402
from jarvis_research.gate1_capability import _run_code_sandboxed  # noqa: E402


def _catalog(entries):
    """Mini catalog rows shaped like build_catalog output."""
    return [{"param_id": i, "param": n, "family": f, "layer": l,
             "shape": [4], "numel": 4, "std": 0.1,
             "mag_min": -0.1, "mag_max": 0.1, "count_min": 1,
             "count_max": 4}
            for i, (n, f, l) in enumerate(entries)]


class ToyModel(torch.nn.Module):
    """Two layers x two families, parameter names like the real catalog."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for _ in range(2):
            block = torch.nn.Module()
            block.mlp = torch.nn.Module()
            block.mlp.gate_proj = torch.nn.Linear(4, 4, bias=False)
            block.self_attn = torch.nn.Module()
            block.self_attn.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.layers.append(block)
        self.embed = torch.nn.Embedding(256, 4)
        self.head = torch.nn.Linear(4, 256, bias=False)

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, ids):
        h = self.embed(ids)
        for b in self.layers:
            h = h + b.mlp.gate_proj(torch.relu(h)) + \
                b.self_attn.q_proj(torch.relu(h))
        return types.SimpleNamespace(logits=self.head(h))


class ToyTokenizer:
    def __call__(self, text, return_tensors="pt"):
        ids = torch.tensor(
            [[ord(c) % 250 + 1 for c in text]], dtype=torch.long)
        return types.SimpleNamespace(input_ids=ids)


CATALOG = _catalog([
    ("layers.0.mlp.gate_proj.weight", "mlp.gate_proj", 0),
    ("layers.0.self_attn.q_proj.weight", "self_attn.q_proj", 0),
    ("layers.1.mlp.gate_proj.weight", "mlp.gate_proj", 1),
    ("layers.1.self_attn.q_proj.weight", "self_attn.q_proj", 1),
])


class TestHookResolution(unittest.TestCase):
    def test_resolves_direct_owners(self):
        targets = resolve_hook_targets(ToyModel(), CATALOG)
        self.assertEqual(targets[0], "layers.0.mlp.gate_proj")
        self.assertEqual(targets[3], "layers.1.self_attn.q_proj")

    def test_unresolvable_raises(self):
        bad = CATALOG + _catalog([("ghost.9.weight", "mlp.x", 9)])
        for i, e in enumerate(bad):
            e["param_id"] = i
        with self.assertRaises(ValueError):
            resolve_hook_targets(ToyModel(), bad)


class TestCollectAndAnnotate(unittest.TestCase):
    TEXTS = {f"{d}_practice_{i}": f"{d} text number {i} longer body"
             for d in ("aa", "bb") for i in (1, 2)}

    def test_collect_and_annotate(self):
        obs = collect_observations(
            ToyModel(), ToyTokenizer(), self.TEXTS,
            {0: "layers.0.mlp.gate_proj", 1: "layers.0.self_attn.q_proj",
             2: "layers.1.mlp.gate_proj", 3: "layers.1.self_attn.q_proj"},
            checkpoint_revision="rev123")
        self.assertEqual(len(obs["hi_passages"]) +
                         len(obs["lo_passages"]), 4)
        self.assertEqual(set(obs["hi_passages"]) & set(obs["lo_passages"]),
                         set())
        # stratified: each domain contributes to both sets
        self.assertTrue(any(p.startswith("aa") for p in obs["hi_passages"]))
        self.assertTrue(any(p.startswith("aa") for p in obs["lo_passages"]))
        for g in obs["groups"].values():
            self.assertTrue(g["available"])
            self.assertEqual(g["n_samples"], 4)
            # Separate hi/lo counts and variance (HZN-IR-04 audit).
            self.assertEqual(g["n_hi"] + g["n_lo"], g["n_samples"])
            self.assertIn("var_hi", g)
            self.assertIn("var_lo", g)
            self.assertTrue(math.isfinite(g["hilo"]))
        ann = annotate_catalog(CATALOG, obs,
                               resolve_hook_targets(ToyModel(), CATALOG))
        for r in ann:
            self.assertTrue(r["obs_available"])
            self.assertEqual(r["obs_group"], r["obs_module"])
            self.assertIn("obs_act_rank", r)
            self.assertIn("obs_hilo_rank", r)
            self.assertGreaterEqual(r["obs_act_rank"], 0.0)
            self.assertLessEqual(r["obs_act_rank"], 1.0)
            # Proposer-visible contrast channel matches the group stat.
            self.assertAlmostEqual(
                r["obs_log_contrast"],
                round(math.log(r["obs_hilo"]), LOG_CONTRAST_DECIMALS))

    def test_unexecuted_module_unavailable(self):
        class Dead(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
                self.dead = torch.nn.Linear(4, 4, bias=False)

            @property
            def device(self):
                return torch.device("cpu")

            def forward(self, ids):
                return self.inner(ids)

        obs = collect_observations(
            Dead(ToyModel()), ToyTokenizer(), self.TEXTS,
            {0: "dead"}, checkpoint_revision="rev123")
        self.assertFalse(obs["groups"]["dead"]["available"])


class TestShuffle(unittest.TestCase):
    def _annotated(self):
        rows = []
        for r in CATALOG:
            row = dict(r)
            row["obs_available"] = True
            row["obs_act"] = 0.1 * (r["param_id"] + 1)
            row["obs_hi"] = 0.2 * (r["param_id"] + 1)
            row["obs_lo"] = 0.1
            row["obs_hilo"] = float(r["param_id"] + 1)
            row["obs_log_contrast"] = round(
                math.log(row["obs_hilo"]), LOG_CONTRAST_DECIMALS)
            row["obs_module"] = row["obs_group"] = f"mod{r['param_id']}"
            rows.append(row)
        # Ranks as annotate_catalog would produce them.
        from jarvis_research.gate1_observation import _family_ranks
        _family_ranks(rows, "obs_act")
        _family_ranks(rows, "obs_hilo")
        return rows

    def test_deterministic_and_coverage(self):
        rows = self._annotated()
        a = shuffle_observations(rows, seed=7)
        b = shuffle_observations(rows, seed=7)
        self.assertEqual(a["catalog"], b["catalog"])
        self.assertGreater(a["effective_coverage"], 0.0)
        self.assertEqual(a["n_groups_shuffled"], 4)
        # joint features preserved: each shuffled row's act/hi/lo came
        # from ONE source row
        for r in a["catalog"]:
            src = round(r["obs_act"] / 0.1 - 1)
            self.assertAlmostEqual(r["obs_hilo"], float(src + 1), places=5)
            self.assertAlmostEqual(r["obs_hi"], 0.2 * (src + 1), places=5)

    def test_permutation_returned_and_group_level(self):
        # Rows sharing an obs_group must receive the SAME reassigned
        # vector (HZN-IR-04 R2 defect A regression).
        rows = self._annotated()
        rows[1]["obs_group"] = rows[0]["obs_group"]  # share mod0
        rows[1]["obs_act"] = rows[0]["obs_act"]
        rows[1]["obs_hi"] = rows[0]["obs_hi"]
        rows[1]["obs_hilo"] = rows[0]["obs_hilo"]
        rows[1]["obs_log_contrast"] = rows[0]["obs_log_contrast"]
        rows[1]["obs_act_rank"] = rows[0]["obs_act_rank"]
        out = shuffle_observations(rows, seed=3)
        perm = out["permutation"]
        self.assertIsNotNone(perm)
        self.assertEqual(set(perm), {"mod0", "mod2", "mod3"})
        self.assertTrue(all(v in perm for v in perm.values()))
        c = {r["param_id"]: r for r in out["catalog"]}
        # shared-group rows carry identical obs VECTORS after
        # permutation. Rank fields are excluded: they are within-family
        # recomputations, and rows 0/1 sit in different families.
        for k in ("obs_act", "obs_hi", "obs_lo", "obs_hilo",
                  "obs_log_contrast"):
            self.assertEqual(c[0][k], c[1][k], k)

    def test_singleton_stratum_unchanged(self):
        rows = self._annotated()[:1]
        # Re-rank the sliced subset: pre-ranks must reflect THIS
        # catalog, as annotate_catalog would have produced.
        from jarvis_research.gate1_observation import _family_ranks
        _family_ranks(rows, "obs_act")
        _family_ranks(rows, "obs_hilo")
        out = shuffle_observations(rows, seed=1)
        self.assertEqual(out["effective_coverage"], 0.0)
        self.assertEqual(out["catalog"][0]["obs_hilo"], 1.0)


class TestTieAwareRanks(unittest.TestCase):
    def test_identical_measurements_identical_ranks(self):
        # HZN-IR-04 R2: param_id tiebreaks fabricated distinctions.
        from jarvis_research.gate1_observation import _family_ranks
        rows = [dict(r, obs_available=True, obs_act=0.5, obs_hilo=1.2,
                     family="fam")
                for r in CATALOG]
        _family_ranks(rows, "obs_act")
        self.assertEqual({r["obs_act_rank"] for r in rows}, {0.5})
        rows[0]["obs_act"] = 0.9
        _family_ranks(rows, "obs_act")
        self.assertEqual(rows[0]["obs_act_rank"], 1.0)
        # three tied members share the midrank of positions 0..2 -> 1/3
        self.assertEqual({r["obs_act_rank"] for r in rows[1:]},
                         {round(1 / 3, 4)})


class TestInterpreter(unittest.TestCase):
    def _row(self, pid, lc):
        r = dict(CATALOG[pid])
        r.update({"obs_available": True, "obs_log_contrast": lc,
                  "family": "mlp.gate_proj"})
        return r

    def test_direction_agnostic_and_floor(self):
        rows = [self._row(0, 0.69), self._row(1, -0.69), self._row(2, 0.0)]
        rows.append(dict(CATALOG[3], obs_available=False,
                         family="mlp.gate_proj"))
        w = interpreter_weights(rows)
        self.assertAlmostEqual(w[0], w[1], places=6)  # |+lc| == |-lc|
        self.assertEqual(w[3], INTERPRETER_FLOOR)
        self.assertLess(w[2], w[0])  # lc=0 -> |lc|=0 -> min

    def test_parity_with_proposer_visible_field(self):
        # HZN-IR-04 R1: the interpreter consumes the same rounded lc
        # the prompt renders — two raw hilo values that round to the
        # same lc MUST get identical weight.
        rows = [self._row(0, 0.14), self._row(1, 0.144),
                self._row(2, -0.9)]
        # lc is already the proposer-visible value; raw precision must
        # not leak in from elsewhere: weights differ only by |lc|.
        w = interpreter_weights(rows)
        self.assertNotEqual(w[0], w[1])
        # and after rounding to the visible precision they would match:
        r0, r1 = dict(rows[0]), dict(rows[1])
        r1["obs_log_contrast"] = 0.14
        w2 = interpreter_weights([r0, r1, rows[2]])
        self.assertAlmostEqual(w2[0], w2[1], places=6)


class TestAbstentionAndGrader(unittest.TestCase):
    def test_abstain_parses(self):
        p = parse_proposal('noise {"edits":[],"abstain":true} tail')
        self.assertTrue(p["abstain"])
        self.assertEqual(p["edits"], [])

    def test_abstain_instruction_in_base_prompt(self):
        # HZN-IR-04 §5: A0 must carry the same abstain instruction as
        # A1/A2 — it lives in the shared base prompt, not PROMPT_OBS.
        self.assertIn('"abstain":true', PROMPT)
        self.assertIn('"abstain":true', PROMPT_OBS)

    def test_catalog_line_lc_parity(self):
        fam = {"mlp.gate_proj": "f0"}
        e = dict(CATALOG[0], obs_available=True, obs_act_rank=0.75,
                 obs_log_contrast=-0.83)
        line = _catalog_line(e, fam, uniform_mag=True, obs=True)
        self.assertIn("|ar=0.75|", line)
        self.assertIn("|lc=-0.83", line)
        e2 = dict(CATALOG[0], obs_available=False)
        self.assertIn("|ar=NA|lc=NA", _catalog_line(e2, fam, True, True))

    def test_grader_confined_and_valid(self):
        src = "def f(x):\n    return x + 1\n"
        res = _run_code_sandboxed(src, [["f(1)", 2]])
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["results"], [True])
        # Fail-closed: a graded result implies containment held.
        self.assertTrue(res["confined"], res)

    def test_grader_memory_cap(self):
        # Allocating ~1 GiB inside a 512 MiB job must fail, not succeed.
        src = "def f():\n    x = ' ' * (1024**3)\n    return 1\n"
        res = _run_code_sandboxed(src, [["f()", 1]], timeout_s=30)
        self.assertFalse(res["ok"], res)
        self.assertNotIn("results", res)

    def test_grader_fail_closed_on_containment_failure(self):
        # HZN-IR-04 R4: containment failure must REFUSE grading, never
        # degrade to confined=false execution.
        orig = _cap._assign_windows_job
        _cap._assign_windows_job = lambda *a, **k: None
        try:
            res = _run_code_sandboxed("def f():\n    return 1\n",
                                      [["f()", 1]])
        finally:
            _cap._assign_windows_job = orig
        self.assertFalse(res["ok"], res)
        self.assertEqual(res.get("containment"), "refused")
        self.assertFalse(res["confined"])
        self.assertNotIn("results", res)


if __name__ == "__main__":
    unittest.main()
