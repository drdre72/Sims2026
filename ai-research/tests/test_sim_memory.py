import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from jarvis_research.sim_memory import *


def base_memory():
    mem=SimMemory()
    mem.registry.register(DEP_MODEL,'qwen3.5-4b-bf16',1)
    mem.registry.register(DEP_HARNESS,'gate1_harness',1)
    mem.registry.register(DEP_EVAL,'gate1_eval_v4',1)
    mem.registry.register(DEP_ONTOLOGY,'sim_ontology',1)
    mem.lineage.add_node('ckpt-base','checkpoint','base qwen3.5-4b',tick=0)
    return mem


def add_simple_hypothesis(mem,hid='H1',cat='coherent_dense_edit',deps=None,cond=None):
    mem.propose_hypothesis(hid,'test hypothesis '+hid,'ckpt-base',cat,
        reconsider_machine=cond or [],
        depends_on=deps if deps is not None else [Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],
        initial_status=SUPPORTED,initial_rationale='test setup')
    return mem.hypothesis(hid)


class TestOntology(unittest.TestCase):
    def test_unknown_mandatory_and_not_deprecatable(self):
        ont=seed_ontology()
        self.assertIn(UNKNOWN_CATEGORY,ont.names())
        with self.assertRaises(ValueError):
            ont.revise(deprecate=[UNKNOWN_CATEGORY])
    def test_revision_preserves_history_and_never_deletes(self):
        ont=seed_ontology()
        c=OntologyCategory('new_class','domain',1,None,'x')
        ont2=ont.revise(add=[c],renames={'old':'new_class'},note='test')
        self.assertEqual(ont2.version,2)
        self.assertIsNotNone(ont2.prior)
        ont3=ont2.revise(deprecate=['coherent_dense_edit'])
        self.assertIn('coherent_dense_edit',ont3.names())
        self.assertFalse(ont3.active('coherent_dense_edit'))
        self.assertEqual(ont2.version,2)
    def test_rename_resolution_is_candidate_not_equivalence(self):
        ont=seed_ontology().revise(renames={'transcription_failure':'output_contract_failure'})
        self.assertEqual(ont.resolve('transcription_failure'),'output_contract_failure')


class TestMeasurementsImmutable(unittest.TestCase):
    def test_measurement_cannot_be_rerecorded_or_mutated(self):
        mem=base_memory()
        mem.record_measurement('m1','perplexity',{'value':11.4},'evolve',{'source':'test'})
        with self.assertRaises(ValueError):
            mem.record_measurement('m1','perplexity',{'value':12.0},'evolve',{'source':'test'})
    def test_sealed_measurements_must_be_redacted(self):
        mem=base_memory()
        with self.assertRaises(ValueError):
            mem.record_measurement('m1','answer_key',{'answer':'C'},'sealed',{'source':'x'})
        m=mem.record_measurement('m2','answer_key',{'redacted':True,'ref':'sha256:abc'},'sealed',{'source':'x'})
        self.assertEqual(m.partition,'sealed')


class TestAcceptance1ConfoundInvalidation(unittest.TestCase):
    """Discovery of a confound invalidates dependent historical interpretations
    WITHOUT modifying original measurements."""
    def test_confound_invalidates_dependents_measurement_untouched(self):
        mem=base_memory()
        mem.record_measurement('m-conf','held_out_loss',{'value':0.5},'evolve',{'source':'exp'})
        mem.propose_hypothesis('H1','loss increase means bad edit','ckpt-base','quality_collapse_perplexity',
            depends_on=[Dependency(DEP_MEASUREMENT,'m-conf',1)],initial_status=REFUTED_CONDITIONAL,
            initial_rationale='measured loss increase')
        iid=mem.hypothesis('H1').current_interpretation_id()
        self.assertEqual(mem.effective_status(iid),REFUTED_CONDITIONAL)
        mem.register_confound('m-conf','evaluation procedure contained order confound',{'source':'audit'})
        self.assertEqual(mem.effective_status(iid),STALE)
        self.assertTrue(mem.measurement_unmodified('m-conf'))
        self.assertTrue(mem.measurement('m-conf').confounded)


class TestAcceptance2RevisionPropagation(unittest.TestCase):
    """Revised interpretations propagate invalidation to downstream conclusions."""
    def test_revision_stales_downstream(self):
        mem=base_memory()
        mem.propose_hypothesis('H1','base claim','ckpt-base','sparse_scalar_edit',
            depends_on=[Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],
            initial_status=SUPPORTED,initial_rationale='base evidence')
        base_iid=mem.hypothesis('H1').current_interpretation_id()
        base_version=mem.interpretation(base_iid).version
        mem.propose_hypothesis('H2','downstream claim','ckpt-base','sparse_scalar_edit',
            depends_on=[Dependency(DEP_INTERPRETATION,base_iid,base_version)],
            initial_status=SUPPORTED,initial_rationale='relies on H1')
        down_iid=mem.hypothesis('H2').current_interpretation_id()
        self.assertEqual(mem.effective_status(down_iid),SUPPORTED)
        mem.add_interpretation('H1',SUPPORTED,'revised base claim','new evidence')
        self.assertEqual(mem.effective_status(down_iid),STALE)
    def test_refuted_basis_makes_dependent_insufficient(self):
        mem=base_memory()
        mem.propose_hypothesis('H1','base','ckpt-base','sparse_scalar_edit',
            depends_on=[Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],initial_status=SUPPORTED)
        base_iid=mem.hypothesis('H1').current_interpretation_id()
        mem.propose_hypothesis('H2','dep','ckpt-base','sparse_scalar_edit',
            depends_on=[Dependency(DEP_INTERPRETATION,base_iid,1)],initial_status=SUPPORTED)
        mem.add_interpretation('H1',REFUTED_CONDITIONAL,'base refuted','contradicting evidence')
        self.assertEqual(mem.effective_status(mem.hypothesis('H2').current_interpretation_id()),STALE)
    def test_already_refuted_basis_makes_dependent_insufficient(self):
        mem=base_memory()
        mem.propose_hypothesis('H1','base','ckpt-base','sparse_scalar_edit',
            depends_on=[Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],initial_status=REFUTED_CONDITIONAL)
        base_iid=mem.hypothesis('H1').current_interpretation_id()
        mem.propose_hypothesis('H2','dep','ckpt-base','sparse_scalar_edit',
            depends_on=[Dependency(DEP_INTERPRETATION,base_iid,1)],initial_status=SUPPORTED)
        self.assertEqual(mem.effective_status(mem.hypothesis('H2').current_interpretation_id()),INSUFFICIENT_EVIDENCE)


class TestAcceptance3RollbackWithoutAmnesia(unittest.TestCase):
    def test_rollback_preserves_all_epistemic_state(self):
        mem=base_memory()
        mem.lineage.intervene('ckpt-e1','checkpoint','after edit 1','att-1',tick=1)
        mem.lineage.intervene('ckpt-e2','checkpoint','after edit 2','att-2',tick=2)
        add_simple_hypothesis(mem)
        mem.record_measurement('m1','accuracy',{'value':0.8},'evolve',{'source':'x'})
        epistemic_before=(mem.snapshot()['hypotheses'],mem.snapshot()['measurements'],
                          mem.snapshot()['reconsideration_events'])
        node=mem.lineage.rollback_to('ckpt-base','ckpt-rb1',tick=3)
        self.assertEqual(mem.lineage.current,'ckpt-rb1')
        self.assertIn(('ckpt-base','rollback',None),node.parents)
        self.assertIsNotNone(mem.lineage.node('ckpt-e2'))
        self.assertIsNotNone(mem.hypothesis('H1'))
        self.assertIsNotNone(mem.measurement('m1'))
        epistemic_after=(mem.snapshot()['hypotheses'],mem.snapshot()['measurements'],
                         mem.snapshot()['reconsideration_events'])
        self.assertEqual(epistemic_before,epistemic_after)
        descendants=mem.lineage.descendants_of('ckpt-base')
        self.assertIn('ckpt-e1',descendants)
        self.assertIn('ckpt-rb1',descendants)


class TestAcceptance4Reconsideration(unittest.TestCase):
    """Previously rejected interventions become eligible for reconsideration
    when their recorded conditions change."""
    def test_conditional_refutation_becomes_eligible_on_condition_change(self):
        mem=base_memory()
        mem.propose_hypothesis('H1','dense L0 attenuation helps','ckpt-base','coherent_dense_edit',
            reconsider_machine=[{'kind':'dependency_changed','dep_type':DEP_MODEL,'dep_id':'qwen-next'}],
            depends_on=[Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],
            initial_status=REFUTED_CONDITIONAL,initial_rationale='refuted on current arch only')
        events=mem.reevaluate({'kind':'dependency_changed','dep_type':DEP_MODEL,'dep_id':'unrelated'})
        self.assertEqual(events,[])
        events=mem.reevaluate({'kind':'dependency_changed','dep_type':DEP_MODEL,'dep_id':'qwen-next'})
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['hypothesis_id'],'H1')
        self.assertEqual(events[0]['intrinsic_status'],REFUTED_CONDITIONAL)


class TestAcceptance5OntologyRevision(unittest.TestCase):
    def test_revision_preserves_classification_and_provenance(self):
        mem=base_memory()
        add_simple_hypothesis(mem,'H1','coherent_dense_edit')
        before=mem.hypothesis('H1')
        mem.revise_ontology(note='v0.2 refinement')
        after=mem.hypothesis('H1')
        self.assertEqual(after.ontology_category,'coherent_dense_edit')
        self.assertEqual(after.ontology_version,1)
        self.assertEqual(mem.ontology.version,2)
        iid=after.current_interpretation_id()
        deps=mem.interpretation(iid).depends_on
        self.assertEqual(mem.hypothesis_status('H1'),SUPPORTED)


class TestAcceptance6ReplayIsolation(unittest.TestCase):
    """Historical replay cannot access evidence unavailable at simulated time."""
    def test_availability_filtering(self):
        mem=base_memory()
        mem.record_measurement('m-early','accuracy',{'value':0.5},'evolve',{'source':'x'},availability_tick=5)
        mem.record_measurement('m-late','accuracy',{'value':0.9},'evolve',{'source':'x'},availability_tick=50)
        mem.propose_hypothesis('H1','h','ckpt-base','sparse_scalar_edit',
            depends_on=[Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],availability_tick=5)
        mem.record_attempt('H1','a1','layer_11','sparse_additive','ckpt-base',availability_tick=5)
        mem.record_outcome('H1','a1','rejected',['m-early'],availability_tick=5)
        mem.record_attempt('H1','a2','layer_12','sparse_additive','ckpt-base',availability_tick=45)
        mem.record_outcome('H1','a2','accepted',['m-late'],availability_tick=50)
        early=mem.retrieve(ontology_category='sparse_scalar_edit',at_tick=10)
        self.assertEqual(len(early),1)
        self.assertEqual(early[0]['measurements'],('m-early',))
        late=mem.retrieve(ontology_category='sparse_scalar_edit',at_tick=60)
        self.assertEqual(set(late[0]['measurements']),{'m-early','m-late'})
    def test_hypothesis_invisible_before_availability(self):
        mem=base_memory()
        mem.propose_hypothesis('H1','late hypothesis','ckpt-base','sparse_scalar_edit',
            depends_on=[Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],availability_tick=100)
        self.assertEqual(mem.retrieve(ontology_category='sparse_scalar_edit',at_tick=50),())
        self.assertEqual(len(mem.retrieve(ontology_category='sparse_scalar_edit',at_tick=100)),1)

    def test_match_keys_respect_availability(self):
        """A visible hypothesis must not be found through a target or dependency
        that only became available later; the at-tick interpretation head's
        dependencies are what match."""
        mem=base_memory()
        mem.propose_hypothesis('H1','h','ckpt-base','sparse_scalar_edit',
            depends_on=[Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],availability_tick=5)
        mem.record_attempt('H1','a1','layer_11','sparse_additive','ckpt-base',availability_tick=5)
        mem.record_attempt('H1','a2','layer_12','sparse_additive','ckpt-base',availability_tick=45)
        mem.record_measurement('m-x','accuracy',{'value':1},'evolve',{'source':'x'},availability_tick=45)
        mem.add_interpretation('H1',SUPPORTED,'later','r',
            depends_on=[Dependency(DEP_MEASUREMENT,'m-x',1)],availability_tick=45)
        self.assertEqual(mem.retrieve(target_component='layer_12',at_tick=10),())
        self.assertEqual(mem.retrieve(dep_type=DEP_MEASUREMENT,at_tick=10),())
        self.assertEqual(len(mem.retrieve(target_component='layer_11',at_tick=10)),1)
        self.assertEqual(len(mem.retrieve(dep_type=DEP_MODEL,at_tick=10)),1)
        self.assertEqual(len(mem.retrieve(target_component='layer_12',at_tick=45)),1)
        self.assertEqual(len(mem.retrieve(dep_type=DEP_MEASUREMENT,at_tick=45)),1)
        self.assertEqual(mem.retrieve(dep_type=DEP_MODEL,at_tick=45),())
        self.assertEqual(len(mem.retrieve(dep_type=DEP_MEASUREMENT)),1)


class TestAcceptance7SimilarityNotEquivalence(unittest.TestCase):
    def test_structural_similarity_never_merges(self):
        mem=base_memory()
        add_simple_hypothesis(mem,'H1','coherent_dense_edit')
        add_simple_hypothesis(mem,'H2','coherent_dense_edit')
        mem.record_attempt('H1','a1','layer_0','additive','ckpt-base')
        mem.record_attempt('H2','a2','layer_0','additive','ckpt-base')
        results=mem.retrieve(target_component='layer_0',ontology_category='coherent_dense_edit')
        self.assertEqual(len(results),2)
        ids={r['hypothesis_id'] for r in results}
        self.assertEqual(ids,{'H1','H2'})
        for r in results:
            self.assertTrue(r['candidate_not_equivalent'])
        self.assertIsNot(mem.hypothesis('H1'),mem.hypothesis('H2'))
        self.assertNotEqual(mem.hypothesis('H1').statement,mem.hypothesis('H2').statement)


class TestAcceptance8ScopedInvalidation(unittest.TestCase):
    """Unrelated state changes do not invalidate demonstrably independent evidence."""
    def test_independent_evidence_survives(self):
        mem=base_memory()
        mem.propose_hypothesis('H-MODEL','model dependent','ckpt-base','quality_collapse_perplexity',
            depends_on=[Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],initial_status=SUPPORTED)
        mem.propose_hypothesis('H-BUS','bus semantics are model independent','ckpt-base','invalidation_semantics',
            depends_on=[Dependency(DEP_HARNESS,'gate1_harness',1)],initial_status=SUPPORTED)
        mem.registry.bump(DEP_MODEL,'qwen3.5-4b-bf16')
        self.assertEqual(mem.hypothesis_status('H-MODEL'),STALE)
        self.assertEqual(mem.hypothesis_status('H-BUS'),SUPPORTED)
    def test_transfer_to_different_entity_gives_applicability_unknown(self):
        mem=base_memory()
        mem.propose_hypothesis('H1','model dependent','ckpt-base','quality_collapse_perplexity',
            depends_on=[Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],initial_status=SUPPORTED)
        mem.registry.replace(DEP_MODEL,'qwen3.5-4b-bf16','qwen-next-7b',1)
        self.assertEqual(mem.hypothesis_status('H1'),APPLICABILITY_UNKNOWN)


class TestInvariants(unittest.TestCase):
    def test_interpretation_cycles_rejected(self):
        mem=base_memory()
        mem.propose_hypothesis('H1','a','ckpt-base','sparse_scalar_edit',
            depends_on=[Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)])
        i1=mem.hypothesis('H1').current_interpretation_id()
        mem.propose_hypothesis('H2','b','ckpt-base','sparse_scalar_edit',
            depends_on=[Dependency(DEP_INTERPRETATION,i1,1)])
        i2=mem.hypothesis('H2').current_interpretation_id()
        with self.assertRaises(ValueError):
            mem.add_interpretation('H1',SUPPORTED,'cyclic','x',
                depends_on=[Dependency(DEP_INTERPRETATION,i2,1)])
    def test_retrospective_prediction_fabrication_prohibited(self):
        mem=base_memory()
        add_simple_hypothesis(mem)
        pred=Prediction('accuracy',0.9,'>=',contemporaneous=False,source='invented after the fact')
        with self.assertRaises(ValueError):
            mem.record_attempt('H1','a1','layer_11','sparse','ckpt-base',prediction=pred)
    def test_prediction_error_computed_mechanically(self):
        mem=base_memory()
        add_simple_hypothesis(mem)
        pred=Prediction('accuracy',0.833,'>=',contemporaneous=True,source='prereg test')
        mem.record_attempt('H1','a1','layer_11','sparse','ckpt-base',prediction=pred)
        mem.record_measurement('m1','accuracy',{'value':0.611},'evolve',{'source':'eval'})
        att=mem.record_outcome('H1','a1','rejected',['m1'])
        self.assertAlmostEqual(att.prediction_error,abs(0.611-0.833),places=6)
    def test_sealed_never_retrieved_even_at_max_clearance(self):
        mem=base_memory()
        add_simple_hypothesis(mem)
        mem.record_attempt('H1','a1','layer_11','sparse','ckpt-base')
        mem.record_measurement('m-open','accuracy',{'value':0.5},'held_out',{'source':'x'})
        mem.record_measurement('m-sealed','answer_key',{'redacted':True},'sealed',{'source':'x'})
        mem.record_outcome('H1','a1','accepted',['m-open','m-sealed'])
        for clearance in ('practice','evolve','held_out','sealed'):
            results=mem.retrieve(ontology_category='coherent_dense_edit',clearance=clearance)
            self.assertNotIn('m-sealed',results[0]['measurements'])
    def test_append_only_histories_accumulate(self):
        mem=base_memory()
        add_simple_hypothesis(mem)
        n_before=len(mem.hypothesis_history('H1'))
        mem.add_interpretation('H1',SUPPORTED,'revised','more evidence')
        self.assertGreater(len(mem.hypothesis_history('H1')),n_before)
        hist=mem.interpretation_history(mem.hypothesis('H1').interpretations[0])
        self.assertIsNotNone(hist[-1].superseded_by)
    def test_missing_measurement_dependency_is_stale_not_crash(self):
        mem=base_memory()
        mem.propose_hypothesis('H1','h','ckpt-base','sparse_scalar_edit',
            depends_on=[Dependency(DEP_MEASUREMENT,'nonexistent',1)],initial_status=SUPPORTED)
        self.assertEqual(mem.hypothesis_status('H1'),STALE)
    def test_snapshot_serializable(self):
        mem=base_memory()
        add_simple_hypothesis(mem)
        import json as _json
        _json.dumps(mem.snapshot())


if __name__=='__main__':
    unittest.main()

class TestStatusAtTickReplay(unittest.TestCase):
    """HZN-SIM-00.5 regression: retrieve(at_tick=...) must report the chain
    head's status as of the replay tick, never the end-state status. The
    preflight caught three defects where post-hoc refutations/support were
    visible at ticks where the question was still open."""

    def test_status_at_tick_replays_chain_history(self):
        mem=base_memory()
        mem.propose_hypothesis('H1','open question','ckpt-base','coherent_dense_edit',
            initial_status=TESTING,availability_tick=10)
        mem.add_interpretation('H1',REFUTED_CONDITIONAL,'refuted later','evidence',
            availability_tick=30)
        self.assertEqual(mem.status_at_tick('H1',20),TESTING)
        self.assertEqual(mem.status_at_tick('H1',40),REFUTED_CONDITIONAL)
        self.assertEqual(mem.hypothesis_status('H1'),REFUTED_CONDITIONAL)

    def test_retrieve_reports_at_tick_status_not_end_state(self):
        mem=base_memory()
        mem.propose_hypothesis('H1','open question','ckpt-base','coherent_dense_edit',
            initial_status=TESTING,availability_tick=10)
        mem.add_interpretation('H1',REFUTED_CONDITIONAL,'refuted later','evidence',
            availability_tick=30)
        self.assertEqual(mem.retrieve(ontology_category='coherent_dense_edit',
            at_tick=20)[0]['status'],TESTING)
        self.assertEqual(mem.retrieve(ontology_category='coherent_dense_edit',
            at_tick=40)[0]['status'],REFUTED_CONDITIONAL)
        self.assertEqual(mem.retrieve(ontology_category='coherent_dense_edit')[0]['status'],
            REFUTED_CONDITIONAL)

    def test_status_at_tick_none_before_first_interpretation(self):
        mem=base_memory()
        mem.propose_hypothesis('H1','later','ckpt-base','coherent_dense_edit',
            initial_status=TESTING,availability_tick=10)
        self.assertIsNone(mem.status_at_tick('H1',5))
        self.assertEqual(mem.status_at_tick('H1',10),TESTING)

class TestHistoricalEffectiveStatus(unittest.TestCase):
    """Reviewer finding (open -> resolved): at-tick status evaluation must use
    the world as of the tick — registry versions from the event log, chain
    heads as of the tick, confounds as of the tick — never end-state state."""

    def _hyp(self,mem,hid='H1',cat='coherent_dense_edit',deps=None):
        mem.propose_hypothesis(hid,'h '+hid,'ckpt-base',cat,
            depends_on=deps if deps is not None else
            [Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],
            initial_status=SUPPORTED,availability_tick=10)
        return mem.hypothesis(hid)

    def test_external_bump_only_invalidates_after_its_tick(self):
        mem=base_memory()
        self._hyp(mem)
        mem.registry.bump(DEP_MODEL,'qwen3.5-4b-bf16',tick=50)
        self.assertEqual(mem.status_at_tick('H1',40),SUPPORTED)
        self.assertEqual(mem.status_at_tick('H1',60),STALE)
        self.assertEqual(mem.hypothesis_status('H1'),STALE)

    def test_replacement_invalidates_only_after_its_tick(self):
        mem=base_memory()
        self._hyp(mem)
        mem.registry.bump(DEP_MODEL,'qwen3.5-4b-bf16',tick=50)
        mem.registry.replace(DEP_MODEL,'qwen3.5-4b-bf16','qwen-next',tick=70)
        self.assertEqual(mem.status_at_tick('H1',40),SUPPORTED)
        self.assertEqual(mem.status_at_tick('H1',65),STALE)
        self.assertEqual(mem.status_at_tick('H1',80),APPLICABILITY_UNKNOWN)
        self.assertEqual(mem.hypothesis_status('H1'),APPLICABILITY_UNKNOWN)

    def test_confound_only_invalidates_after_its_tick(self):
        mem=base_memory()
        mem.record_measurement('m-x','accuracy',{'value':1},'evolve',
            {'source':'x'},availability_tick=5)
        self._hyp(mem,deps=[Dependency(DEP_MEASUREMENT,'m-x',1)])
        mem.register_confound('m-x','late confound',{'p':1},availability_tick=30)
        self.assertEqual(mem.status_at_tick('H1',20),SUPPORTED)
        self.assertEqual(mem.status_at_tick('H1',40),STALE)
        self.assertEqual(mem.hypothesis_status('H1'),STALE)

    def test_basis_revision_only_invalidates_after_its_tick(self):
        mem=base_memory()
        mem.propose_hypothesis('H1','basis','ckpt-base','coherent_dense_edit',
            depends_on=[Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],
            initial_status=SUPPORTED,availability_tick=5)
        mem.propose_hypothesis('H2','dependent','ckpt-base','coherent_dense_edit',
            depends_on=[Dependency(DEP_INTERPRETATION,'H1:interp:1',1)],
            initial_status=SUPPORTED,availability_tick=10)
        mem.add_interpretation('H1',REFUTED_CONDITIONAL,'revised','r',availability_tick=50)
        self.assertEqual(mem.status_at_tick('H2',40),SUPPORTED)
        self.assertEqual(mem.status_at_tick('H2',60),STALE)
        self.assertEqual(mem.hypothesis_status('H2'),STALE)

    def test_unregistered_entity_is_unknown_at_early_tick(self):
        mem=base_memory()
        mem.registry.register(DEP_EVAL,'suite-x',1,tick=40)
        self._hyp(mem,deps=[Dependency(DEP_EVAL,'suite-x',1)])
        self.assertEqual(mem.status_at_tick('H1',20),APPLICABILITY_UNKNOWN)
        self.assertEqual(mem.status_at_tick('H1',60),SUPPORTED)

    def test_corpus_ontology_revision_is_encoding_time_event(self):
        from jarvis_research.sim_corpus import build_gold_corpus
        mem=build_gold_corpus()
        self.assertEqual(mem.registry.version_at(DEP_ONTOLOGY,'sim_ontology',50),1)
        self.assertEqual(mem.registry.version_at(DEP_ONTOLOGY,'sim_ontology',150),2)
        self.assertEqual(mem.registry.version(DEP_ONTOLOGY,'sim_ontology'),2)
        self.assertEqual(mem.status_at_tick('H-EPI05-BINDING',60),SUPPORTED)
        self.assertEqual(mem.hypothesis_status('H-EPI05-BINDING'),SUPPORTED)

class TestSnapshotCompleteness(unittest.TestCase):
    """Reviewer finding (open -> resolved): the snapshot must carry enough to
    reconstruct chain chronology — full interpretation chains with availability
    ticks, confound records, and the registry event log — not ID lists alone."""

    def test_snapshot_contains_full_chains_and_events(self):
        mem=base_memory()
        mem.propose_hypothesis('H1','basis','ckpt-base','coherent_dense_edit',
            depends_on=[Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],
            initial_status=TESTING,availability_tick=10)
        mem.add_interpretation('H1',REFUTED_CONDITIONAL,'revised','r',availability_tick=30)
        mem.record_measurement('m-x','accuracy',{'value':1},'evolve',
            {'source':'x'},availability_tick=5)
        mem.register_confound('m-x','c',{'p':1},availability_tick=40)
        mem.registry.bump(DEP_MODEL,'qwen3.5-4b-bf16',tick=50)
        snap=mem.snapshot()
        self.assertIn('interpretations',snap)
        self.assertIn('confounds',snap)
        self.assertIn('registry_events',snap)
        chain=snap['interpretations']['H1:interp:1']
        self.assertEqual([v['status'] for v in chain],['testing'])
        self.assertEqual([v['availability_tick'] for v in chain],[10])
        chain2=snap['interpretations']['H1:interp:2']
        self.assertEqual([v['status'] for v in chain2],['refuted_conditional'])
        self.assertEqual([v['availability_tick'] for v in chain2],[30])
        self.assertEqual(snap['confounds']['m-x'][0]['tick'],40)
        kinds=[e['kind'] for e in snap['registry_events']]
        self.assertIn('bump',kinds)

    def test_corpus_snapshot_reconstructs_chain_chronology(self):
        from jarvis_research.sim_corpus import build_gold_corpus
        snap=build_gold_corpus().snapshot()
        selfedit=snap['interpretations']['H-G05-SELFEDIT-MECHANISM:interp:1']
        self.assertEqual(len(selfedit),1)
        self.assertEqual(selfedit[0]['status'],'testing')
        self.assertEqual(selfedit[0]['availability_tick'],17)
        selfedit2=snap['interpretations']['H-G05-SELFEDIT-MECHANISM:interp:2']
        self.assertEqual(selfedit2[0]['status'],'refuted_conditional')
        self.assertEqual(selfedit2[0]['availability_tick'],23)
        # every hypothesis interpretation id resolves in the snapshot
        ids={i for h in snap['hypotheses'] for i in h['interpretations']}
        self.assertTrue(ids.issubset(set(snap['interpretations'])))
        # registry history: pre-history registrations plus the encoding-time ontology bump
        events=snap['registry_events']
        self.assertEqual(len([e for e in events if e['kind']=='register']),4)
        bump=[e for e in events if e['kind']=='bump']
        self.assertEqual(bump[0]['tick'],100)

class TestStatusOracle(unittest.TestCase):
    """Reviewer finding (open -> resolved): an independent oracle for
    intrinsic-vs-effective status. Every expected value below is hand-derived
    from the documented semantics (dependency-before-intrinsic precedence;
    conditional refutation is not staleness; invalidation propagates through
    interpretation deps; at-tick evaluation uses the world as of the tick).
    Disagreement between the oracle and the substrate is a defect in one of
    them and must be resolved, not averaged."""

    def _clean(self,mem,hid,cat='coherent_dense_edit',status=SUPPORTED,at=10,
               deps=None):
        mem.propose_hypothesis(hid,'h '+hid,'ckpt-base',cat,
            depends_on=deps if deps is not None else
            [Dependency(DEP_MODEL,'qwen3.5-4b-bf16',1)],
            initial_status=status,availability_tick=at)
        return mem.hypothesis(hid)

    def test_oracle_substrate_rows(self):
        def row_clean():
            m=base_memory(); self._clean(m,'H1')
            return m,[('end','H1',SUPPORTED),('at50','H1',SUPPORTED,50)]
        def row_bump():
            m=base_memory(); self._clean(m,'H1')
            m.registry.bump(DEP_MODEL,'qwen3.5-4b-bf16',tick=50)
            return m,[('end','H1',STALE),('at40','H1',SUPPORTED,40),
                      ('at60','H1',STALE,60)]
        def row_replace():
            m=base_memory(); self._clean(m,'H1')
            m.registry.replace(DEP_MODEL,'qwen3.5-4b-bf16','qwen-next',tick=70)
            return m,[('end','H1',APPLICABILITY_UNKNOWN),
                      ('at60','H1',SUPPORTED,60),
                      ('at80','H1',APPLICABILITY_UNKNOWN,80)]
        def row_unregistered():
            m=base_memory()
            self._clean(m,'H1',deps=[Dependency(DEP_EVAL,'never-registered',1)])
            return m,[('end','H1',APPLICABILITY_UNKNOWN),
                      ('at50','H1',APPLICABILITY_UNKNOWN,50)]
        def row_confound():
            m=base_memory()
            m.record_measurement('m-x','accuracy',{'value':1},'evolve',
                {'source':'x'},availability_tick=5)
            self._clean(m,'H1',deps=[Dependency(DEP_MEASUREMENT,'m-x',1)])
            m.register_confound('m-x','c',{'p':1},availability_tick=30)
            return m,[('end','H1',STALE),('at20','H1',SUPPORTED,20),
                      ('at40','H1',STALE,40)]
        def row_missing_measurement():
            m=base_memory()
            self._clean(m,'H1',deps=[Dependency(DEP_MEASUREMENT,'m-ghost',1)])
            return m,[('end','H1',STALE)]
        def row_basis_revision():
            m=base_memory()
            self._clean(m,'H1',at=5)
            self._clean(m,'H2',at=10,
                        deps=[Dependency(DEP_INTERPRETATION,'H1:interp:1',1)])
            m.add_interpretation('H1',REFUTED_CONDITIONAL,'revised','r',
                availability_tick=50)
            return m,[('H2at40','H2',SUPPORTED,40),('H2at60','H2',STALE,60),
                      ('H2end','H2',STALE)]
        def row_refuted_basis():
            m=base_memory()
            self._clean(m,'H1',status=REFUTED_CONDITIONAL,at=5)
            self._clean(m,'H2',at=10,
                        deps=[Dependency(DEP_INTERPRETATION,'H1:interp:1',1)])
            return m,[('end','H2',INSUFFICIENT_EVIDENCE)]
        def row_stale_propagation():
            m=base_memory()
            self._clean(m,'H1',at=5)
            self._clean(m,'H2',at=10,
                        deps=[Dependency(DEP_INTERPRETATION,'H1:interp:1',1)])
            m.registry.bump(DEP_MODEL,'qwen3.5-4b-bf16',tick=50)
            return m,[('H1end','H1',STALE),('H2end','H2',STALE)]
        def row_intrinsic_survives():
            m=base_memory()
            self._clean(m,'H1',status=REFUTED_CONDITIONAL)
            self._clean(m,'H2',status=INSUFFICIENT_EVIDENCE,at=11)
            return m,[('refuted','H1',REFUTED_CONDITIONAL),
                      ('insufficient','H2',INSUFFICIENT_EVIDENCE)]
        def row_dep_precedence():
            m=base_memory()
            self._clean(m,'H1',status=SUPPORTED)
            self._clean(m,'H2',status=REFUTED_CONDITIONAL,at=11)
            m.registry.bump(DEP_MODEL,'qwen3.5-4b-bf16',tick=50)
            return m,[('supported-intrinsic','H1',STALE),
                      ('refuted-intrinsic','H2',STALE)]
        rows=[('clean',row_clean),('external-bump-timing',row_bump),
              ('replacement-timing',row_replace),('never-registered',row_unregistered),
              ('confound-timing',row_confound),('missing-measurement',row_missing_measurement),
              ('basis-revision-timing',row_basis_revision),('refuted-basis',row_refuted_basis),
              ('stale-propagation',row_stale_propagation),
              ('intrinsic-survives-clean-deps',row_intrinsic_survives),
              ('dependency-precedes-intrinsic',row_dep_precedence)]
        for name,builder in rows:
            with self.subTest(oracle_row=name):
                mem,checks=builder()
                for check in checks:
                    label,hid,expected=check[0],check[1],check[2]
                    tick=check[3] if len(check)>3 else None
                    got=(mem.status_at_tick(hid,tick) if tick is not None
                         else mem.hypothesis_status(hid))
                    self.assertEqual(got,expected,
                        f"{name}/{label}: expected {expected}, got {got}")

    def test_oracle_corpus_rows(self):
        from jarvis_research.sim_corpus import build_gold_corpus
        mem=build_gold_corpus()
        rows=[
            ('selfedit-open-at-20','H-G05-SELFEDIT-MECHANISM',20,TESTING),
            ('selfedit-refuted-at-25','H-G05-SELFEDIT-MECHANISM',25,REFUTED_CONDITIONAL),
            ('selfedit-end','H-G05-SELFEDIT-MECHANISM',None,REFUTED_CONDITIONAL),
            ('bus-supported-at-61','H-BUS-STALE-PRECEDENCE',61,SUPPORTED),
            ('bus-refuted-at-83','H-BUS-STALE-PRECEDENCE',83,REFUTED_CONDITIONAL),
            ('bus-supported-end','H-BUS-STALE-PRECEDENCE',None,SUPPORTED),
            ('contract-testing-at-92','H-EPI06-CONTRACT-REPAIR',92,TESTING),
            ('contract-refuted-at-95','H-EPI06-CONTRACT-REPAIR',95,REFUTED_CONDITIONAL),
            ('contract-refuted-end','H-EPI06-CONTRACT-REPAIR',None,REFUTED_CONDITIONAL),
        ]
        for name,hid,tick,expected in rows:
            with self.subTest(oracle_row=name):
                got=(mem.status_at_tick(hid,tick) if tick is not None
                     else mem.hypothesis_status(hid))
                self.assertEqual(got,expected,f"{name}: expected {expected}, got {got}")
