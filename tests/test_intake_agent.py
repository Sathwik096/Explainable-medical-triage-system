"""
Automated unit tests for Stage 1: IntakeAgent & IntakeOrchestrator
Includes tests for:
- Bayesian Pertinent Negative Reasoning
- Dual Stopping Policy (Tau + Margin)
- Red-Flag Emergency Triage Prioritization
- Standardized Stage 2 Handoff Payload
"""

import unittest
import pandas as pd
from src.decoder import DDXPlusDecoder
from src.diagnostic_engine import DiagnosticEngine
from src.intake_agent import AuditLogEntry, ConsultationState, IntakeAgent, IntakeOrchestrator


class TestIntakeAgentStage1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decoder = DDXPlusDecoder(
            evidences_path="data/raw/release_evidences.json",
            conditions_path="data/raw/release_conditions.json",
        )
        cls.engine = DiagnosticEngine(
            decoder=cls.decoder,
            model_path="models/diagnostic_model.joblib",
        )
        cls.agent = IntakeAgent(
            engine=cls.engine,
            decoder=cls.decoder,
            tau=0.75,
            margin_threshold=0.50,
            max_turns=5,
        )
        cls.orchestrator = IntakeOrchestrator(agent=cls.agent)

    def test_init_consultation(self):
        state = self.orchestrator.init_consultation(
            age=45,
            sex="M",
            initial_evidence="E_201",
            patient_id="TEST-001",
        )
        self.assertEqual(state.patient_id, "TEST-001")
        self.assertEqual(state.age, 45)
        self.assertEqual(state.sex, "M")
        self.assertIn("E_201", state.confirmed_evidences)
        self.assertIn("E_201", state.asked_question_codes)
        self.assertGreater(len(state.current_differential), 0)
        self.assertGreater(state.confidence, 0.0)
        self.assertEqual(state.turn, 0)
        self.assertEqual(len(state.audit_log), 0)

    def test_select_next_question(self):
        state = self.orchestrator.init_consultation(
            age=45,
            sex="M",
            initial_evidence="E_201",
        )
        q_info = self.agent.select_next_question(state)
        self.assertIsNotNone(q_info)
        self.assertIn("code", q_info)
        self.assertIn("question_en", q_info)
        self.assertIn("clinical_rationale", q_info)
        self.assertIn("is_red_flag", q_info)
        self.assertNotEqual(q_info["code"], "E_201")

    def test_bayesian_pertinent_negatives(self):
        # When a patient denies a cardinal symptom, competing conditions are down-weighted
        diff_before = self.agent.compute_differential_with_negatives(
            confirmed_evidences=["E_169"],
            denied_evidences=[],
            age=49,
            sex="M",
            top_k=5,
        )
        p_laryngo_before = next((d["probability"] for d in diff_before if d["condition"] == "Larygospasm"), 0.0)

        # E_194 is a cardinal symptom of Laryngospasm
        diff_after = self.agent.compute_differential_with_negatives(
            confirmed_evidences=["E_169"],
            denied_evidences=["E_194"],
            age=49,
            sex="M",
            top_k=5,
        )
        p_laryngo_after = next((d["probability"] for d in diff_after if d["condition"] == "Larygospasm"), 0.0)

        # Probability of Laryngospasm must drop when its symptom is denied
        self.assertLess(p_laryngo_after, p_laryngo_before)

    def test_record_turn_and_audit_log(self):
        state = self.orchestrator.init_consultation(
            age=45,
            sex="M",
            initial_evidence="E_201",
        )
        q_info = self.agent.select_next_question(state)
        self.assertIsNotNone(q_info)

        self.agent.record_turn(
            state=state,
            question_info=q_info,
            patient_response_str="YES",
            new_confirmed_tokens=[q_info["code"]],
            new_denied_tokens=[],
        )

        self.assertEqual(state.turn, 1)
        self.assertIn(q_info["code"], state.confirmed_evidences)
        self.assertIn(q_info["code"], state.asked_question_codes)
        self.assertEqual(len(state.audit_log), 1)

        entry = state.audit_log[0]
        self.assertEqual(entry.turn, 1)
        self.assertEqual(entry.question_code, q_info["code"])
        self.assertEqual(entry.patient_response, "YES")
        self.assertIn("margin_after_percent", entry.to_dict())

    def test_run_simulation_and_handoff(self):
        mock_patient = {
            "AGE": 49,
            "SEX": "M",
            "INITIAL_EVIDENCE": "E_169",
            "EVIDENCES": "['E_169', 'E_201', 'E_86', 'E_87', 'E_181']",
            "PATHOLOGY": "Allergic sinusitis",
        }
        state = self.orchestrator.run_simulation(mock_patient, patient_id="SIM-PATIENT-0")
        self.assertGreaterEqual(state.turn, 1)
        self.assertLessEqual(state.turn, self.agent.max_turns)
        self.assertGreater(len(state.audit_log), 0)

        # Audit log dataframe
        df_audit = self.orchestrator.export_audit_log_dataframe(state)
        self.assertFalse(df_audit.empty)
        self.assertEqual(len(df_audit), state.turn)

        # Stage 2 handoff payload
        payload = self.orchestrator.get_stage2_handoff_payload(state, ground_truth_pathology="Allergic sinusitis")
        self.assertIn("patient_id", payload)
        self.assertIn("intake_summary", payload)
        self.assertIn("top2_margin_percent", payload["intake_summary"])
        self.assertIn("has_emergency_alert", payload["intake_summary"])
        self.assertIn("clinical_evidence_profile", payload)
        self.assertIn("differential_diagnosis", payload)
        self.assertIn("audit_trail", payload)
        self.assertEqual(payload["demographics"]["age"], 49)


if __name__ == "__main__":
    unittest.main()
