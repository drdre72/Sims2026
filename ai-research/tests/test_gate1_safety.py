import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jarvis_research.gate1_evaluator import Evaluator, generate_partitions, _text_hash
from jarvis_research.gate1_patch_manager import ARM_A, ARM_B, PatchManager
from jarvis_research.gate1_proposer import SelfProposer, parse_proposal


class TinyModel:
    def __init__(self):
        self.weights = {
            "model.layers.11.mlp.gate_proj.weight": torch.ones(1024),
            "model.layers.0.mlp.gate_proj.weight": torch.ones(1024),
        }

    def named_parameters(self):
        return self.weights.items()


class Gate1SafetyTests(unittest.TestCase):
    def test_partitions_disjoint_and_loader_rejects_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / (name + ".json")
                     for name in ("practice", "selection", "regression")]
            generate_partitions(*paths)
            docs = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
            groups = [set(map(_text_hash, doc["passages"].values())) for doc in docs]
            self.assertTrue(groups[0].isdisjoint(groups[1]))
            self.assertTrue(groups[0].isdisjoint(groups[2]))
            self.assertTrue(groups[1].isdisjoint(groups[2]))
            Evaluator.load(*paths)
            docs[1]["passages"]["leak"] = docs[0]["passages"][
                next(iter(docs[0]["passages"]))]
            paths[1].write_text(json.dumps(docs[1]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                Evaluator.load(*paths)

    def test_batch_drift_and_sparse_budget_fail_closed(self):
        model = TinyModel()
        name_a, name_b = model.weights
        b = PatchManager(model, None, ARM_B, 1)
        double = {"edits": [{"param": name_b, "operation": "scale_dense",
                             "magnitude": .70}] * 2}
        edits, error = b.validate(double)
        self.assertIsNone(edits)
        self.assertIn("cumulative", error)
        a = PatchManager(model, None, ARM_A, 1)
        duplicate = {"edits": [{"param": name_a, "operation": "add_sparse",
                                "magnitude": .01, "count": 1}] * 2}
        self.assertIn("duplicate", a.validate(duplicate)[1])
        model.weights["model.layers.12.mlp.gate_proj.weight"] = torch.ones(1024)
        a = PatchManager(model, None, ARM_A, 1)
        aggregate = {"edits": [
            {"param": n, "operation": "add_sparse", "magnitude": .01, "count": 300}
            for n in (name_a, "model.layers.12.mlp.gate_proj.weight")]}
        self.assertIn("aggregate", a.validate(aggregate)[1])
        self.assertEqual({}, a.cumulative_delta)

    def test_rollback_resets_drift_and_is_exact(self):
        model = TinyModel()
        b = PatchManager(model, None, ARM_B, 1)
        name = "model.layers.0.mlp.gate_proj.weight"
        edits, error = b.validate({"edits": [
            {"param": name, "operation": "scale_dense", "magnitude": .8}]})
        self.assertIsNone(error)
        b.apply(edits)
        self.assertTrue(b.rollback_all()[0])
        self.assertEqual({}, b.cumulative_delta)
        self.assertEqual(.8, b.validate({"edits": [
            {"param": name, "operation": "scale_dense",
             "magnitude": .8}]})[0][0]["magnitude"])

    def test_proposer_has_only_practice_interface_and_same_target(self):
        model = TinyModel()
        manager = PatchManager(model, None, ARM_B, 1)
        other = TinyModel()
        with self.assertRaisesRegex(ValueError, "edited target model"):
            SelfProposer(other, None, manager)
        proposer = SelfProposer(model, None, manager)
        self.assertIs(proposer.model, manager.model)
        self.assertFalse(hasattr(proposer, "selection"))
        self.assertFalse(hasattr(proposer, "regression"))
        self.assertEqual(
            {"edits": []},
            parse_proposal('reason {bad} then {"edits": []}'))

    def test_budget_prevents_duplicate_batch_even_after_prior_edit(self):
        model = TinyModel()
        b = PatchManager(model, None, ARM_B, 2)
        name = "model.layers.0.mlp.gate_proj.weight"
        edit = {"param": name, "operation": "scale_dense", "magnitude": .79}
        b.apply(b.validate({"edits": [edit]})[0])
        self.assertIsNone(b.validate({"edits": [edit, edit]})[0])
        self.assertTrue(b.rollback_all()[0])


if __name__ == "__main__":
    unittest.main()
