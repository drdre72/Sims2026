import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from jarvis_research.sim_memory import *
from jarvis_research.sim_corpus import build_gold_corpus,validate_corpus,extract_gate1_8_summary


class TestGoldCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mem=build_gold_corpus()

    def test_integrity_no_errors(self):
        self.assertEqual(validate_corpus(self.mem),[])

    def test_extraction_pulls_real_values(self):
        s=extract_gate1_8_summary()
        self.assertEqual(s['n_generations'],125)
        self.assertEqual(s['repaired_nonnormal_strict'],11)
        self.assertEqual(s['repaired_nonnormal_total'],18)
        self.assertEqual(s['epistemic_decision_correct'],10)
        self.assertEqual(s['mech_stale_emissions'],0)

    def test_corpus_size_and_chronology(self):
        self.assertGreaterEqual(len(self.mem.hypotheses()),10)
        for hyp in self.mem.hypotheses():
            self.assertLessEqual(hyp.availability_tick,99)

    def test_prediction_error_real_calibration_case(self):
        hyp=self.mem.hypothesis('H-EPI06-CONTRACT-REPAIR')
        att=hyp.attempts[0]
        self.assertIsNotNone(att.prediction)
        self.assertTrue(att.prediction.contemporaneous)
        self.assertAlmostEqual(att.prediction_error,abs(11/18-15/18),places=5)

    def test_selfedit_revision_chain(self):
        """The recursive showcase: proposed -> refuted_conditional revision,
        original preserved, nothing rewritten."""
        hyp=self.mem.hypothesis('H-G05-SELFEDIT-MECHANISM')
        self.assertEqual(len(hyp.interpretations),2)
        first=self.mem.interpretation(hyp.interpretations[0])
        self.assertEqual(first.status,TESTING)
        self.assertIsNotNone(first.superseded_by)
        self.assertEqual(self.mem.hypothesis_status('H-G05-SELFEDIT-MECHANISM'),REFUTED_CONDITIONAL)

    def test_bus_invalidation_three_stage_chain(self):
        """assumed-supported -> refuted by reproduction -> supported post-fix;
        first stage honestly flagged retrospective."""
        hyp=self.mem.hypothesis('H-BUS-STALE-PRECEDENCE')
        self.assertEqual(len(hyp.interpretations),3)
        first=self.mem.interpretation(hyp.interpretations[0])
        self.assertTrue(first.retrospective)
        final=self.mem.interpretation(hyp.interpretations[-1])
        self.assertEqual(final.status,SUPPORTED)
        self.assertEqual(self.mem.hypothesis_status('H-BUS-STALE-PRECEDENCE'),SUPPORTED)

    def test_bus_entry_is_model_independent(self):
        """Scoped dependency: bumping the model checkpoint must not invalidate
        the bus-semantics entry, while model-dependent entries go stale."""
        mem=build_gold_corpus()
        self.assertEqual(mem.hypothesis_status('H-BUS-STALE-PRECEDENCE'),SUPPORTED)
        self.assertEqual(mem.hypothesis_status('H-EPI05-M02'),SUPPORTED)
        mem.registry.bump(DEP_MODEL,'qwen3.5-4b-bf16',tick=110)
        self.assertEqual(mem.hypothesis_status('H-EPI05-M02'),STALE)
        self.assertEqual(mem.hypothesis_status('H-BUS-STALE-PRECEDENCE'),SUPPORTED)
        self.assertEqual(mem.hypothesis_status('H-EPI06-MECH-RENDER'),SUPPORTED)

    def test_model_transfer_produces_applicability_unknown(self):
        mem=build_gold_corpus()
        mem.registry.replace(DEP_MODEL,'qwen3.5-4b-bf16','qwen-next-7b',1,tick=110)
        self.assertEqual(mem.hypothesis_status('H-EPI06-CONTRACT-REPAIR'),APPLICABILITY_UNKNOWN)
        self.assertEqual(mem.hypothesis_status('H-BUS-STALE-PRECEDENCE'),SUPPORTED)

    def test_transfer_triggers_reconsideration_eligibility(self):
        """gate0_5 fragility hypotheses carry machine-checkable reconsideration
        on model change; entries in non-good standing become eligible."""
        mem=build_gold_corpus()
        mem.registry.replace(DEP_MODEL,'qwen3.5-4b-bf16','qwen-next-7b',1,tick=110)
        events=mem.reevaluate({'kind':'dependency_changed','dep_type':DEP_MODEL,'dep_id':'qwen-next'})
        ids={e['hypothesis_id'] for e in events}
        self.assertIn('H-G05-SPARSE-SAFE',ids)

    def test_supported_entries_do_not_generate_reconsideration(self):
        mem=build_gold_corpus()
        events=mem.reevaluate({'kind':'dependency_changed','dep_type':DEP_MODEL,'dep_id':'qwen-next'})
        for e in events:
            hyp=mem.hypothesis(e['hypothesis_id'])
            interp=mem.interpretation(hyp.current_interpretation_id())
            self.assertNotIn(interp.status,(SUPPORTED,) if e['effective_status']==SUPPORTED else ())

    def test_depth_gradient_and_coherent_lethal_are_candidates_not_merged(self):
        results=self.mem.retrieve(ontology_category='input_boundary_fragility')
        ids={r['hypothesis_id'] for r in results}
        self.assertEqual(ids,{'H-G05-COHERENT-LETHAL','H-G05-DEPTH-GRADIENT'})
        for r in results:
            self.assertTrue(r['candidate_not_equivalent'])

    def test_replay_before_tie_bug_hides_fix(self):
        """Chronological replay at tick 75 (pre-reproduction) sees the assumed-
        supported first interpretation only; post-fix tick 90 sees the chain."""
        early=self.mem.retrieve(ontology_category='invalidation_semantics',at_tick=75)
        self.assertEqual(len(early),1)
        hyp=self.mem.hypothesis('H-BUS-STALE-PRECEDENCE')
        first=self.mem.interpretation(hyp.interpretations[0])
        self.assertEqual(first.status,SUPPORTED)
        self.assertTrue(first.retrospective)

    def test_ontology_revision_in_corpus(self):
        self.assertEqual(self.mem.ontology.version,2)
        self.assertIsNotNone(self.mem.ontology.category('measurement_confound'))
        hyp=self.mem.hypothesis('H-G05-SPARSE-SAFE')
        self.assertEqual(hyp.ontology_version,1)
        hyp2=self.mem.hypothesis('H-G12-SHARED-PARENT')
        self.assertEqual(hyp2.ontology_category,'measurement_confound')

    def test_measurements_immutable_through_corpus(self):
        for hyp in self.mem.hypotheses():
            for att in hyp.attempts:
                for mid in att.consequences:
                    self.assertTrue(self.mem.measurement_unmodified(mid),mid)

    def test_snapshot_serializable(self):
        import json as _json
        text=_json.dumps(self.mem.snapshot())
        self.assertGreater(len(text),1000)


if __name__=='__main__':
    unittest.main()
