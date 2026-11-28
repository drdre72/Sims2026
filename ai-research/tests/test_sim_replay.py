import sys,unittest,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from jarvis_research.sim_replay import (
    load_decision_points,validate_decision_points,run_preflight,cascade_probes,
    frame_retrieve,visible_hypotheses,classify_miss,run_arm_R,run_arm_Y,
    analytic_hit_probability,evaluate_decision_logic,SEED,
    MISS_NOT_AVAILABLE,MISS_CATEGORY_ABSENT,MISS_VOCABULARY_MISMATCH,
    MISS_DEPENDENCY_NOT_DECLARED,
)
from jarvis_research.sim_corpus import build_gold_corpus


class TestDecisionPointTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mem=build_gold_corpus()
        cls.data=load_decision_points()

    def test_table_valid_against_corpus(self):
        self.assertEqual(validate_decision_points(self.data,self.mem),[])

    def test_control_dp_carries_no_landmarks(self):
        for dp in self.data["decision_points"]:
            if dp["control"]:
                self.assertEqual(dp["required_landmarks"],[])

    def test_tamper_detection(self):
        """Guards the not_available_at_tick class before any run."""
        broken=json.loads(json.dumps({k:v for k,v in self.data.items()}))
        broken["decision_points"][0]["tick"]=10  # DP-1: SELFEDIT available at 17
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"dp.json"
            p.write_text(json.dumps(broken),encoding="utf-8")
            data=load_decision_points(p)
            errors=validate_decision_points(data,self.mem)
        self.assertTrue(any("H-G05-SELFEDIT-MECHANISM" in e for e in errors))


class TestArms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mem=build_gold_corpus()

    def test_frame_union_never_conjoins(self):
        """OR-union over keys: a hypothesis matching any key is included,
        one matching none is excluded (frozen frame semantics)."""
        frame=[{"ontology_category":"safe_null_surface"},
               {"ontology_category":"input_boundary_fragility"}]
        out=frame_retrieve(self.mem,frame,at_tick=99)
        ids=set(out)
        self.assertIn("H-G05-SAFE-NULL",ids)
        self.assertIn("H-G05-COHERENT-LETHAL",ids)
        self.assertIn("H-G05-DEPTH-GRADIENT",ids)
        self.assertNotIn("H-G05-SPARSE-SAFE",ids)
        single=frame_retrieve(self.mem,frame[:1],at_tick=99)
        self.assertEqual(set(single),{"H-G05-SAFE-NULL"})

    def test_dep_and_category_union(self):
        frame=[{"ontology_category":"invalidation_semantics"},{"dep_type":"harness"}]
        out=frame_retrieve(self.mem,frame,at_tick=90)
        self.assertIn("H-BUS-STALE-PRECEDENCE",out)      # category key
        self.assertIn("H-G12-SHARED-PARENT",out)          # dependency key
        self.assertNotIn("H-EPI05-M02",out)               # matches neither

    def test_recency_ordering(self):
        vis=visible_hypotheses(self.mem,90)
        y=run_arm_Y(vis,["H-EPI05-M02","H-EPI05-BINDING"],5)
        self.assertTrue(y["hit"])
        ordered=sorted(vis,key=lambda h:(-h.availability_tick,h.hypothesis_id))
        self.assertEqual({h.hypothesis_id for h in ordered[:5]},
                         {"H-EPI06-CONTRACT-REPAIR","H-BUS-STALE-PRECEDENCE",
                          "H-EPI05-M02","H-EPI05-BINDING","H-G12-SHARED-PARENT"})

    def test_equal_budget_saturates_random(self):
        vis=[h.hypothesis_id for h in visible_hypotheses(self.mem,20)]
        r=run_arm_R(vis,["H-G05-SELFEDIT-MECHANISM"],len(vis),SEED+1)
        self.assertEqual(r["sampled_hit_rate"],1.0)

    def test_analytic_probability_exact(self):
        self.assertEqual(analytic_hit_probability(4,2,1),0.5)
        self.assertEqual(analytic_hit_probability(4,1,2),0.0)
        self.assertAlmostEqual(analytic_hit_probability(11,5,3),28/462,places=9)

    def test_sampled_matches_analytic(self):
        vis=[h.hypothesis_id for h in visible_hypotheses(self.mem,90)]
        r=run_arm_R(vis,["H-EPI05-BINDING","H-BUS-STALE-PRECEDENCE","H-EPI05-M02"],5,SEED+3)
        self.assertLess(abs(r["sampled_hit_rate"]-28/462),0.05)


class TestMissClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mem=build_gold_corpus()
        cls.data={d["dp_id"]:d for d in load_decision_points()["decision_points"]}

    def test_not_available(self):
        dp=self.data["DP-1"]
        vis=visible_hypotheses(self.mem,dp["tick"])
        self.assertEqual(classify_miss(self.mem,dp,dp["frame_full"],
                                       "H-G12-CORRESPONDENCE",vis),MISS_NOT_AVAILABLE)

    def test_category_absent(self):
        """DP-6 minimal frame queries policy_boundary_confusion at tick 94,
        before any hypothesis held the category."""
        dp=self.data["DP-6"]
        vis=visible_hypotheses(self.mem,dp["tick"])
        self.assertEqual(classify_miss(self.mem,dp,dp["frame_minimal"],
                                       "H-EPI05-M02",vis),MISS_CATEGORY_ABSENT)

    def test_vocabulary_mismatch(self):
        """DP-3 full frame reaches invalidation_semantics holders but not the
        m02 landmark, whose pinned category is display_order_copying."""
        dp=self.data["DP-3"]
        vis=visible_hypotheses(self.mem,dp["tick"])
        self.assertEqual(classify_miss(self.mem,dp,dp["frame_full"],
                                       "H-EPI05-M02",vis),MISS_VOCABULARY_MISMATCH)

    def test_dependency_not_declared(self):
        """Dep-only frame: the landmark declares no dependency of that type."""
        dp=self.data["DP-6"]
        vis=visible_hypotheses(self.mem,dp["tick"])
        self.assertEqual(classify_miss(self.mem,dp,[{"dep_type":"harness"}],
                                       "H-EPI05-M02",vis),MISS_DEPENDENCY_NOT_DECLARED)


class TestCascadeProbes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cascade=cascade_probes()

    def test_historical_probes_quantify_sparsity(self):
        self.assertEqual(self.cascade["historical"]["H1_bus_chain_revision"]["count"],0)
        self.assertEqual(self.cascade["historical"]["H2_ontology_revision"]["count"],0)

    def test_model_replacement_cascade(self):
        p1=self.cascade["synthetic"]["P1_model_replacement"]
        self.assertEqual(p1["count"],10)
        self.assertEqual(set(p1["affected"]),{
            "H-G05-SPARSE-SAFE","H-G05-COHERENT-LETHAL","H-G05-DEPTH-GRADIENT",
            "H-G05-SELFEDIT-MECHANISM","H-G05-SAFE-NULL","H-G12-CORRESPONDENCE",
            "H-EPI05-BINDING","H-EPI05-M02","H-EPI06-CONTRACT-REPAIR",
            "H-EPI06-POLICY-BOUNDARY"})

    def test_harness_and_eval_cascades(self):
        self.assertEqual(self.cascade["synthetic"]["P2_harness_bump"]["count"],6)
        self.assertEqual(self.cascade["synthetic"]["P3_eval_bump"]["count"],1)
        self.assertEqual(self.cascade["synthetic"]["P3_eval_bump"]["affected"],
                         ["H-EPI06-CONTRACT-REPAIR"])

    def test_gap_probes(self):
        self.assertEqual(self.cascade["synthetic"]["P4_confound_registration"]["count"],0)
        self.assertEqual(self.cascade["synthetic"]["P5_ontology_bump"]["count"],0)


class TestVerdictLogic(unittest.TestCase):
    def _leak(self):
        return {"at_tick_violations":0,"arm_L_validity":True}

    def test_supported(self):
        m={"landmark_recall":6/7,"full_hit_dps":4,"s_ge_r_dps":4,"s_ge_y_dps":4,
           "mean_candidate_fraction":0.3,"miss_causes":[]}
        self.assertEqual(evaluate_decision_logic(m,self._leak(),5)["verdict"],
                         "RETRIEVAL_UTILITY_SUPPORTED")

    def test_precision_warning(self):
        m={"landmark_recall":6/7,"full_hit_dps":4,"s_ge_r_dps":4,"s_ge_y_dps":4,
           "mean_candidate_fraction":0.8,"miss_causes":[]}
        self.assertEqual(evaluate_decision_logic(m,self._leak(),5)["verdict"],
                         "SUPPORTED_WITH_PRECISION_WARNING")

    def test_vocabulary_gap_failure(self):
        m={"landmark_recall":3/7,"full_hit_dps":1,"s_ge_r_dps":2,"s_ge_y_dps":2,
           "mean_candidate_fraction":0.2,
           "miss_causes":[MISS_CATEGORY_ABSENT,MISS_VOCABULARY_MISMATCH]}
        self.assertEqual(evaluate_decision_logic(m,self._leak(),5)["verdict"],
                         "VOCABULARY_GAP_FAILURE")

    def test_mechanism_failure(self):
        m={"landmark_recall":3/7,"full_hit_dps":1,"s_ge_r_dps":2,"s_ge_y_dps":2,
           "mean_candidate_fraction":0.2,"miss_causes":[MISS_DEPENDENCY_NOT_DECLARED]}
        self.assertEqual(evaluate_decision_logic(m,self._leak(),5)["verdict"],
                         "MECHANISM_FAILURE")

    def test_inconclusive(self):
        m={"landmark_recall":4/7,"full_hit_dps":2,"s_ge_r_dps":2,"s_ge_y_dps":2,
           "mean_candidate_fraction":0.2,"miss_causes":[]}
        self.assertEqual(evaluate_decision_logic(m,self._leak(),5)["verdict"],
                         "INCONCLUSIVE")

    def test_leakage_invalidates(self):
        m={"landmark_recall":1.0,"full_hit_dps":5,"s_ge_r_dps":5,"s_ge_y_dps":5,
           "mean_candidate_fraction":0.1,"miss_causes":[]}
        self.assertEqual(evaluate_decision_logic(m,{"at_tick_violations":1,
                         "arm_L_validity":True},5)["verdict"],"PREFLIGHT_INVALID")


class TestPreflightRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result=run_preflight()

    def test_deterministic_digest(self):
        self.assertEqual(run_preflight()["digest"],self.result["digest"])

    def test_leakage_invariants(self):
        self.assertEqual(self.result["leakage"]["at_tick_violations"],0)
        self.assertTrue(self.result["leakage"]["arm_L_validity"])
        counts=self.result["leakage"]["arm_L_future_counts"]
        self.assertGreater(counts["DP-1"],0)
        self.assertEqual(counts["DP-5"],0)

    def test_status_fidelity_expected_catches(self):
        """The bug detector: replay filters availability, not chain history."""
        expected={("DP-1","H-G05-SELFEDIT-MECHANISM"),
                  ("DP-3","H-EPI06-CONTRACT-REPAIR"),
                  ("DP-4","H-BUS-STALE-PRECEDENCE")}
        seen={(r["dp_id"],m["hypothesis_id"])
              for r in self.result["decision_points"]
              for m in r["status_mismatches"]}
        self.assertEqual(seen,expected)

    def test_expected_battery_outcome(self):
        m=self.result["metrics"]
        self.assertEqual(m["surfaced_required"],6)
        self.assertEqual(m["total_required"],7)
        self.assertEqual(m["full_hit_dps"],4)
        self.assertEqual(m["scored_dps"],5)
        self.assertEqual(m["s_ge_r_dps"],4)
        self.assertEqual(m["s_ge_y_dps"],4)
        self.assertEqual(m["miss_causes"],[MISS_VOCABULARY_MISMATCH])
        self.assertGreater(m["mean_candidate_fraction"],0.5)

    def test_verdict(self):
        self.assertEqual(self.result["verdict"]["verdict"],
                         "SUPPORTED_WITH_PRECISION_WARNING")

    def test_export_roundtrip(self):
        import tempfile
        from jarvis_research.sim_replay import export_evidence
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"evidence.json"
            export_evidence(self.result,p)
            reloaded=json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(reloaded["digest"],self.result["digest"])


if __name__=="__main__":
    unittest.main()
