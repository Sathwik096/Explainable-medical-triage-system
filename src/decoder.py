"""
DDXPlus Clinical Preprocessing & Evidence Decoder Pipeline
==========================================================
Decodes machine-readable DDXPlus tokens (e.g. E_48, E_54_@_V_161, E_56_@_4)
into rich, structured, human-readable clinical representations.

Distinguishes between:
- Acute symptoms vs. Antecedents (medical history, risk factors, demographics)
- Binary vs. Categorical vs. Multi-choice evidence types
- Pathology metadata (ICD-10 codes, clinical severity levels 1-5)
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class DDXPlusDecoder:
    """
    Decoder and clinical feature extractor for the DDXPlus benchmark dataset.
    """

    def __init__(
        self,
        evidences_path: Union[str, Path] = "data/raw/release_evidences.json",
        conditions_path: Union[str, Path] = "data/raw/release_conditions.json",
    ):
        self.evidences_path = Path(evidences_path)
        self.conditions_path = Path(conditions_path)

        if not self.evidences_path.exists():
            raise FileNotFoundError(f"Evidences file not found at: {self.evidences_path}")
        if not self.conditions_path.exists():
            raise FileNotFoundError(f"Conditions file not found at: {self.conditions_path}")

        with open(self.evidences_path, "r", encoding="utf-8") as f:
            self.evidences: Dict[str, Any] = json.load(f)

        with open(self.conditions_path, "r", encoding="utf-8") as f:
            self.conditions: Dict[str, Any] = json.load(f)

    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decodes a single evidence token (e.g., 'E_91', 'E_54_@_V_161', 'E_56_@_4').

        Returns a dictionary containing:
            - token: the original raw token
            - code: base evidence code (e.g. 'E_54')
            - question_en: English clinical question
            - is_antecedent: True if risk factor/medical history, False if acute symptom
            - data_type: 'B' (Binary), 'C' (Categorical), 'M' (Multi-choice)
            - value_code: value identifier (e.g. 'V_161', '4', or None for binary)
            - value_meaning: human-readable English text for value (or True for binary)
            - clinical_text: human-readable formatted string for summary / prompt
        """
        if "_@_" in token:
            code, val_code = token.split("_@_", 1)
        else:
            code, val_code = token, None

        meta = self.evidences.get(code, {})
        question_en = meta.get("question_en", f"Unknown question for {code}")
        is_antecedent = meta.get("is_antecedent", False)
        data_type = meta.get("data_type", "B")

        if val_code is None:
            # Binary positive evidence
            value_meaning = True
            clinical_text = f"{question_en} -> Present"
        else:
            # Categorical, multi-choice, or continuous/integer
            value_meanings_map = meta.get("value_meaning", {})
            mapped_val = value_meanings_map.get(val_code)

            if isinstance(mapped_val, dict):
                value_meaning = mapped_val.get("en", mapped_val.get("fr", val_code))
            elif mapped_val is not None:
                value_meaning = mapped_val
            else:
                value_meaning = val_code

            clinical_text = f"{question_en} -> {value_meaning}"

        return {
            "token": token,
            "code": code,
            "question_en": question_en,
            "is_antecedent": is_antecedent,
            "data_type": data_type,
            "value_code": val_code,
            "value_meaning": value_meaning,
            "clinical_text": clinical_text,
        }

    def decode_condition(self, condition_name: str) -> Dict[str, Any]:
        """
        Decodes a pathology/condition name into ICD-10 code, severity, and associations.
        """
        meta = self.conditions.get(condition_name, {})
        return {
            "name": condition_name,
            "icd10": meta.get("icd10-id", "Unknown"),
            "severity": meta.get("severity", 0),
            "expected_symptoms_count": len(meta.get("symptoms", {})),
            "expected_antecedents_count": len(meta.get("antecedents", {})),
        }

    def decode_patient(self, row: Union[pd.Series, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Decodes a full patient record from raw DDXPlus columns:
        AGE, SEX, PATHOLOGY, INITIAL_EVIDENCE, EVIDENCES, DIFFERENTIAL_DIAGNOSIS.
        """
        # Parse EVIDENCES
        raw_evidences = row.get("EVIDENCES")
        if isinstance(raw_evidences, str):
            evidences_list = ast.literal_eval(raw_evidences)
        elif isinstance(raw_evidences, list):
            evidences_list = raw_evidences
        else:
            evidences_list = []

        # Parse INITIAL_EVIDENCE
        init_ev = row.get("INITIAL_EVIDENCE")
        decoded_init_ev = self.decode_token(str(init_ev)) if init_ev else None

        # Separate symptoms vs antecedents
        decoded_symptoms: List[Dict[str, Any]] = []
        decoded_antecedents: List[Dict[str, Any]] = []

        for token in evidences_list:
            decoded = self.decode_token(str(token))
            if decoded["is_antecedent"]:
                decoded_antecedents.append(decoded)
            else:
                decoded_symptoms.append(decoded)

        # Parse DIFFERENTIAL_DIAGNOSIS
        raw_diff = row.get("DIFFERENTIAL_DIAGNOSIS")
        if isinstance(raw_diff, str):
            diff_list = ast.literal_eval(raw_diff)
        elif isinstance(raw_diff, list):
            diff_list = raw_diff
        else:
            diff_list = []

        decoded_diff = []
        for item in diff_list:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                d_name, prob = item[0], float(item[1])
                cond_meta = self.decode_condition(d_name)
                decoded_diff.append({
                    "condition": d_name,
                    "probability": round(prob, 4),
                    "icd10": cond_meta["icd10"],
                    "severity": cond_meta["severity"],
                })

        # Ground truth pathology
        pathology_name = str(row.get("PATHOLOGY", ""))
        decoded_pathology = self.decode_condition(pathology_name)

        return {
            "age": int(row.get("AGE", 0)),
            "sex": str(row.get("SEX", "Unknown")),
            "pathology": decoded_pathology,
            "initial_evidence": decoded_init_ev,
            "symptoms": decoded_symptoms,
            "antecedents": decoded_antecedents,
            "total_symptoms": len(decoded_symptoms),
            "total_antecedents": len(decoded_antecedents),
            "differential_diagnosis": decoded_diff,
        }

    def to_clinical_vignette(self, decoded_patient: Dict[str, Any]) -> str:
        """
        Converts a decoded patient dictionary into a clear, formatted clinical vignette note.
        Suitable for LLM prompts, agent reasoning traces, and physician review.
        """
        age = decoded_patient["age"]
        sex = "Male" if decoded_patient["sex"] == "M" else "Female" if decoded_patient["sex"] == "F" else decoded_patient["sex"]
        
        init_ev = decoded_patient.get("initial_evidence")
        init_ev_text = init_ev["clinical_text"] if init_ev else "None reported"

        symptoms_lines = []
        for s in decoded_patient["symptoms"]:
            symptoms_lines.append(f"  - {s['clinical_text']}")
        symptoms_str = "\n".join(symptoms_lines) if symptoms_lines else "  - None reported"

        antecedents_lines = []
        for a in decoded_patient["antecedents"]:
            antecedents_lines.append(f"  - {a['clinical_text']}")
        antecedents_str = "\n".join(antecedents_lines) if antecedents_lines else "  - None reported"

        pathology = decoded_patient["pathology"]
        diff_lines = []
        for d in decoded_patient.get("differential_diagnosis", [])[:5]:
            diff_lines.append(f"  - {d['condition']} (prob: {d['probability']*100:.1f}%, severity: {d['severity']})")
        diff_str = "\n".join(diff_lines) if diff_lines else "  - None listed"

        vignette = f"""================ CLINICAL PATIENT VIGNETTE ================
DEMOGRAPHICS:
  Age: {age} | Sex: {sex}

CHIEF COMPLAINT (INITIAL EVIDENCE):
  {init_ev_text}

ACUTE SYMPTOMS ({decoded_patient['total_symptoms']}):
{symptoms_str}

MEDICAL HISTORY & RISK FACTORS (ANTECEDENTS, {decoded_patient['total_antecedents']}):
{antecedents_str}

GROUND TRUTH PATHOLOGY:
  {pathology['name']} (ICD-10: {pathology['icd10']}, Severity: {pathology['severity']}/5)

TOP DIFFERENTIAL DIAGNOSES:
{diff_str}
==========================================================="""
        return vignette

    def decode_dataframe(self, df: pd.DataFrame, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Decodes a DataFrame of raw patient records into a DataFrame with structured clinical columns.
        """
        target_df = df.iloc[:max_rows].copy() if max_rows else df.copy()

        decoded_rows = []
        for _, row in target_df.iterrows():
            dec = self.decode_patient(row)
            decoded_rows.append({
                "AGE": dec["age"],
                "SEX": dec["sex"],
                "PATHOLOGY": dec["pathology"]["name"],
                "PATHOLOGY_ICD10": dec["pathology"]["icd10"],
                "PATHOLOGY_SEVERITY": dec["pathology"]["severity"],
                "INITIAL_EVIDENCE_NAME": dec["initial_evidence"]["question_en"] if dec["initial_evidence"] else "",
                "INITIAL_EVIDENCE_VALUE": str(dec["initial_evidence"]["value_meaning"]) if dec["initial_evidence"] else "",
                "NUM_SYMPTOMS": dec["total_symptoms"],
                "NUM_ANTECEDENTS": dec["total_antecedents"],
                "SYMPTOMS_SUMMARY": "; ".join([s["clinical_text"] for s in dec["symptoms"]]),
                "ANTECEDENTS_SUMMARY": "; ".join([a["clinical_text"] for a in dec["antecedents"]]),
            })

        return pd.DataFrame(decoded_rows)
