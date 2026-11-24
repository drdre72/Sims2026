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
    INTERPRETER_FLOOR)
from jarvis_research.gate1_proposer import parse_proposal  # noqa: E402
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
        obs = {"groups": {}}
        rows = []
        for r in CATALOG:
            row = dict(r)
            row["obs_available"] = True
            row["obs_act"] = 0.1 * (r["param_id"] + 1)
            row["obs_hi"] = 0.2 * (r["param_id"] + 1)
            row["obs_lo"] = 0.1
            row["obs_hilo"] = float(r["param_id"] + 1)
            row["obs_module"] = row["obs_group"] = f"mod{r['param_id']}"
            rows.append(row)
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

    def test_singleton_stratum_unchanged(self):
        rows = self._annotated()[:1]
        out = shuffle_observations(rows, seed=1)
        self.assertEqual(out["effective_coverage"], 0.0)
        self.assertEqual(out["catalog"][0]["obs_hilo"], 1.0)


class TestInterpreter(unittest.TestCase):
    def test_direction_agnostic_and_floor(self):
        rows = []
        for pid, hilo in [(0, 2.0), (1, 0.5), (2, 1.0)]:
            r = dict(CATALOG[pid])
            r.update({"obs_available": True, "obs_hilo": hilo,
                      "family": "mlp.gate_proj"})
            rows.append(r)
        rows.append(dict(CATALOG[3], obs_available=False,
                         family="mlp.gate_proj"))
        w = interpreter_weights(rows)
        self.assertAlmostEqual(w[0], w[1], places=6)  # |ln2| == |ln0.5|
        self.assertEqual(w[3], INTERPRETER_FLOOR)
        self.assertLess(w[2], w[0])  # hilo=1 -> |ln|=0 -> min


class TestAbstentionAndGrader(unittest.TestCase):
    def test_abstain_parses(self):
        p = parse_proposal('noise {"edits":[],"abstain":true} tail')
        self.assertTrue(p["abstain"])
        self.assertEqual(p["edits"], [])

    def test_grader_confined_and_valid(self):
        src = "def f(x):\n    return x + 1\n"
        res = _run_code_sandboxed(src, [["f(1)", 2]])
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["results"], [True])
        # On Windows the Job Object should confine it; other platforms
        # report confined=false rather than pretending.
        self.assertIn("confined", res)

    def test_grader_memory_cap(self):
        # Allocating ~1 GiB inside a 512 MiB job must fail, not succeed.
        src = "def f():\n    x = ' ' * (1024**3)\n    return 1\n"
        res = _run_code_sandboxed(src, [["f()", 1]], timeout_s=30)
        self.assertFalse(res["ok"], res)
        self.assertNotIn("results", res)


if __name__ == "__main__":
    unittest.main()
