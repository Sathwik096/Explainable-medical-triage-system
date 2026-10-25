"""
Unit tests for DiagnosticEngine
"""

import os
import tempfile
import unittest
import numpy as np
import pandas as pd
from src.decoder import DDXPlusDecoder
from src.diagnostic_engine import DiagnosticEngine


class TestDiagnosticEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decoder = DDXPlusDecoder(
            evidences_path="data/raw/release_evidences.json",
            conditions_path="data/raw/release_conditions.json",
        )
        # Create a small multi-patient DataFrame across 3 conditions for fast unit testing
        cls.toy_df = pd.DataFrame([
            {"AGE": 25, "SEX": "M", "PATHOLOGY": "GERD", "EVIDENCES": "['E_201', 'E_53', 'E_54_@_V_161']"},
            {"AGE": 30, "SEX": "F", "PATHOLOGY": "GERD", "EVIDENCES": "['E_201', 'E_53']"},
            {"AGE": 40, "SEX": "M", "PATHOLOGY": "Bronchitis", "EVIDENCES": "['E_201', 'E_91']"},
            {"AGE": 50, "SEX": "F", "PATHOLOGY": "Bronchitis", "EVIDENCES": "['E_201', 'E_91', 'E_53']"},
            {"AGE": 60, "SEX": "M", "PATHOLOGY": "URTI", "EVIDENCES": "['E_181', 'E_169']"},
            {"AGE": 65, "SEX": "F", "PATHOLOGY": "URTI", "EVIDENCES": "['E_181', 'E_201']"},
        ] * 10)  # Replicate to have 60 rows for train_test_split

    def test_unfitted_raises_error(self):
        engine = DiagnosticEngine(decoder=self.decoder)
        with self.assertRaises(RuntimeError):
            engine.predict_proba(["E_201"], age=30, sex="M")

    def test_fit_and_metrics(self):
        engine = DiagnosticEngine(decoder=self.decoder)
        metrics = engine.fit(self.toy_df, test_size=0.2, random_state=42)
        self.assertTrue(engine.is_trained)
        self.assertIn("top1_accuracy", metrics)
        self.assertIn("top3_accuracy", metrics)
        self.assertEqual(metrics["n_classes"], 3)
        self.assertGreater(metrics["vocab_size"], 0)

    def test_predict_proba_and_differential(self):
        engine = DiagnosticEngine(decoder=self.decoder)
        engine.fit(self.toy_df, test_size=0.2, random_state=42)

        probs = engine.predict_proba(["E_201", 'E_53'], age=28, sex="M")
        self.assertEqual(len(probs), 3)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)

        diff = engine.get_differential(["E_201", 'E_53'], age=28, sex="M", top_k=2)
        self.assertEqual(len(diff), 2)
        # Check ordering: highest probability first
        self.assertGreaterEqual(diff[0]["probability"], diff[1]["probability"])
        self.assertIn("icd10", diff[0])
        self.assertIn("severity", diff[0])

    def test_save_and_load(self):
        engine = DiagnosticEngine(decoder=self.decoder)
        engine.fit(self.toy_df, test_size=0.2, random_state=42)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "test_engine.joblib")
            engine.save(save_path)
            self.assertTrue(os.path.exists(save_path))

            loaded_engine = DiagnosticEngine(decoder=self.decoder, model_path=save_path)
            self.assertTrue(loaded_engine.is_trained)
            self.assertEqual(loaded_engine.vocab, engine.vocab)

            p1 = engine.predict_proba(["E_201"], age=30, sex="F")
            p2 = loaded_engine.predict_proba(["E_201"], age=30, sex="F")
            for k in p1:
                self.assertAlmostEqual(p1[k], p2[k], places=5)


if __name__ == "__main__":
    unittest.main()
