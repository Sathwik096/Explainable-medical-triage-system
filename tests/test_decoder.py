"""
Unit tests for DDXPlusDecoder
"""

import os
import unittest
import pandas as pd
from src.decoder import DDXPlusDecoder


class TestDDXPlusDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decoder = DDXPlusDecoder(
            evidences_path="data/raw/release_evidences.json",
            conditions_path="data/raw/release_conditions.json",
        )

    def test_decode_binary_evidence(self):
        # E_91 is fever/chills (Binary, symptom)
        decoded = self.decoder.decode_token("E_91")
        self.assertEqual(decoded["code"], "E_91")
        self.assertEqual(decoded["data_type"], "B")
        self.assertFalse(decoded["is_antecedent"])
        self.assertTrue(decoded["value_meaning"])
        self.assertIn("fever", decoded["question_en"].lower())

    def test_decode_categorical_evidence(self):
        # E_54_@_V_161 is pain characterization
        decoded = self.decoder.decode_token("E_54_@_V_161")
        self.assertEqual(decoded["code"], "E_54")
        self.assertEqual(decoded["value_code"], "V_161")
        self.assertFalse(decoded["is_antecedent"])
        self.assertEqual(decoded["value_meaning"], "sensitive")

    def test_decode_antecedent(self):
        # E_48 is an antecedent (live with 4 or more people)
        decoded = self.decoder.decode_token("E_48")
        self.assertTrue(decoded["is_antecedent"])

    def test_decode_condition(self):
        cond = self.decoder.decode_condition("GERD")
        self.assertEqual(cond["name"], "GERD")
        self.assertIn("K21", cond["icd10"])
        self.assertEqual(cond["severity"], 3)

    def test_decode_patient_and_vignette(self):
        sample_patient = {
            "AGE": 45,
            "SEX": "M",
            "PATHOLOGY": "GERD",
            "INITIAL_EVIDENCE": "E_201",
            "EVIDENCES": "['E_201', 'E_91', 'E_48']",
            "DIFFERENTIAL_DIAGNOSIS": "[['GERD', 0.85], ['Bronchitis', 0.15]]",
        }
        decoded = self.decoder.decode_patient(sample_patient)
        self.assertEqual(decoded["age"], 45)
        self.assertEqual(decoded["sex"], "M")
        self.assertEqual(decoded["pathology"]["name"], "GERD")
        self.assertEqual(decoded["total_symptoms"], 2)
        self.assertEqual(decoded["total_antecedents"], 1)

        vignette = self.decoder.to_clinical_vignette(decoded)
        self.assertIn("CLINICAL PATIENT VIGNETTE", vignette)
        self.assertIn("Age: 45 | Sex: Male", vignette)
        self.assertIn("GERD", vignette)

    def test_decode_dataframe(self):
        sample_df = pd.DataFrame([
            {
                "AGE": 30,
                "SEX": "F",
                "PATHOLOGY": "Acute laryngitis",
                "INITIAL_EVIDENCE": "E_201",
                "EVIDENCES": "['E_201']",
                "DIFFERENTIAL_DIAGNOSIS": "[['Acute laryngitis', 0.9]]",
            }
        ])
        decoded_df = self.decoder.decode_dataframe(sample_df)
        self.assertEqual(len(decoded_df), 1)
        self.assertEqual(decoded_df.iloc[0]["PATHOLOGY"], "Acute laryngitis")
        self.assertEqual(decoded_df.iloc[0]["NUM_SYMPTOMS"], 1)


if __name__ == "__main__":
    unittest.main()
