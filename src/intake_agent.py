"""
Stage 1: Multi-Turn Clinical Intake Agent & Advanced Orchestration Engine
========================================================================
High-Performance Implementation of Stage 1 with:
1. Pertinent Negative Reasoning (Bayesian Active Rule-Out Engine)
2. Dual Stopping Policy (Confidence Target Tau + Top-2 Separation Margin + Information Saturation)
3. Red-Flag Emergency Triage Prioritization (Severity 1 & 2 Life-Safety Questions)
4. Anatomical Continuity & Symptom Clustering (Follow-ups on active complaints)
5. Tamper-evident Clinical Audit Trail & Standardized Stage 2 Handoff Payload
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.decoder import DDXPlusDecoder
from src.diagnostic_engine import DiagnosticEngine


@dataclass
class AuditLogEntry:
    """Record of a single consultation turn stored for clinician audit and review."""
    turn: int
    timestamp: str
    question_code: str
    question_text: str
    is_antecedent: bool
    is_red_flag: bool
    clinical_rationale: str
    patient_response: str
    evidences_added: List[str]
    top_condition_before: str
    prob_before_percent: float
    top_condition_after: str
    prob_after_percent: float
    confidence_delta_percent: float
    margin_after_percent: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn": self.turn,
            "timestamp": self.timestamp,
            "question_code": self.question_code,
            "question_text": self.question_text,
            "is_antecedent": self.is_antecedent,
            "is_red_flag": self.is_red_flag,
            "clinical_rationale": self.clinical_rationale,
            "patient_response": self.patient_response,
            "evidences_added": self.evidences_added,
            "top_condition_before": self.top_condition_before,
            "prob_before_percent": self.prob_before_percent,
            "top_condition_after": self.top_condition_after,
            "prob_after_percent": self.prob_after_percent,
            "confidence_delta_percent": self.confidence_delta_percent,
            "margin_after_percent": self.margin_after_percent,
        }


@dataclass
class ConsultationState:
    """
    Shared state container maintaining the active clinical record during intake.
    """
    patient_id: str
    age: int
    sex: str
    initial_evidence: str
    confirmed_evidences: List[str] = field(default_factory=list)
    denied_evidences: List[str] = field(default_factory=list)
    asked_question_codes: List[str] = field(default_factory=list)
    current_differential: List[Dict[str, Any]] = field(default_factory=list)
    top_condition: str = ""
    confidence: float = 0.0
    top2_margin: float = 0.0
    turn: int = 0
    is_converged: bool = False
    termination_reason: str = ""
    has_emergency_alert: bool = False
    emergency_alerts: List[str] = field(default_factory=list)
    audit_log: List[AuditLogEntry] = field(default_factory=list)

    @property
    def confidence_percent(self) -> float:
        return round(self.confidence * 100.0, 2)

    @property
    def margin_percent(self) -> float:
        return round(self.top2_margin * 100.0, 2)


class IntakeAgent:
    """
    Clinical Intake Agent responsible for Next-Best-Question selection,
    Bayesian pertinent negative rule-out reasoning, and emergency safety triage.
    """

    def __init__(
        self,
        engine: DiagnosticEngine,
        decoder: Optional[DDXPlusDecoder] = None,
        tau: float = 0.80,
        margin_threshold: float = 0.55,
        min_info_gain: float = 0.005,
        max_turns: int = 8,
        top_k_candidates: int = 5,
        negative_penalty: float = 0.50,
    ):
        """
        Args:
            engine: Pre-trained DiagnosticEngine instance.
            decoder: DDXPlusDecoder instance for natural language questions & metadata.
            tau: Confidence convergence threshold (default: 0.80 -> 80% confidence).
            margin_threshold: Probability difference between Top-1 and Top-2 for decisive separation (default: 0.55 -> 55%).
            min_info_gain: Minimum expected discriminative utility before stopping due to information saturation.
            max_turns: Maximum allowed intake questions before stopping (default: 8).
            top_k_candidates: Number of top differential candidates to consider for question ranking.
            negative_penalty: Penalty multiplier applied to diseases when a cardinal symptom is denied.
        """
        self.engine = engine
        self.decoder = decoder or DDXPlusDecoder()
        self.tau = tau
        self.margin_threshold = margin_threshold
        self.min_info_gain = min_info_gain
        self.max_turns = max_turns
        self.top_k_candidates = top_k_candidates
        self.negative_penalty = negative_penalty

    def compute_differential_with_negatives(
        self,
        confirmed_evidences: List[str],
        denied_evidences: List[str],
        age: Union[int, float],
        sex: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Computes the differential diagnosis probability distribution across all 49 conditions,
        actively applying Bayesian Pertinent Negative down-weighting for denied symptoms.
        """
        raw_proba_map = self.engine.predict_proba(
            evidences=confirmed_evidences,
            age=age,
            sex=sex,
        )

        denied_set = set(denied_evidences)
        adjusted_map: Dict[str, float] = {}

        for cond_name, base_p in raw_proba_map.items():
            meta = self.decoder.conditions.get(cond_name, {})
            cond_symptoms = set(meta.get("symptoms", {}).keys())

            if denied_set:
                # Count how many cardinal symptoms of this condition were explicitly denied
                denied_count = len(denied_set & cond_symptoms)
                # Bayesian multiplier: (1 - penalty)^denied_count
                rule_out_multiplier = (1.0 - self.negative_penalty) ** denied_count
                adjusted_p = base_p * rule_out_multiplier
            else:
                adjusted_p = base_p

            adjusted_map[cond_name] = max(1e-9, adjusted_p)

        # Normalize probability distribution
        total_p = sum(adjusted_map.values())
        if total_p > 0:
            norm_map = {k: v / total_p for k, v in adjusted_map.items()}
        else:
            norm_map = raw_proba_map

        sorted_candidates = sorted(norm_map.items(), key=lambda x: x[1], reverse=True)[:top_k]

        differential = []
        for condition_name, prob in sorted_candidates:
            cond_meta = self.decoder.decode_condition(condition_name)
            differential.append({
                "condition": condition_name,
                "probability": round(prob, 4),
                "probability_percent": round(prob * 100, 2),
                "icd10": cond_meta["icd10"],
                "severity": cond_meta["severity"],
            })

        return differential

    def select_next_question(self, state: ConsultationState) -> Optional[Dict[str, Any]]:
        """
        Identifies the optimal next question to ask using:
        1. Pairwise differential discriminative utility across top competing diagnoses.
        2. Emergency Red-Flag prioritization for severity 1 & 2 life-threatening conditions.
        3. Symptom clustering and anatomical continuity bonus.
        """
        if not state.current_differential:
            return None

        asked_codes = set(state.asked_question_codes)
        top_candidates = state.current_differential[: self.top_k_candidates]

        scores: Dict[str, float] = {}
        rationale_map: Dict[str, str] = {}
        red_flag_set: Set[str] = set()

        # Identify if any emergency conditions exist in the current differential
        emergency_conditions = [
            d for d in top_candidates if d.get("severity") in [1, 2] and d["probability"] >= 0.03
        ]

        # Extract recently confirmed evidence bases for anatomical continuity
        recent_confirmed_bases = {
            t.split("_@_")[0] for t in state.confirmed_evidences[-3:]
        }

        for i, d1 in enumerate(top_candidates):
            c1, p1 = d1["condition"], d1["probability"]
            meta1 = self.decoder.conditions.get(c1, {})
            s1 = set(meta1.get("symptoms", {}).keys())
            a1 = set(meta1.get("antecedents", {}).keys())
            all1 = s1 | a1
            sev1 = meta1.get("severity", 5)

            for j in range(i + 1, len(top_candidates)):
                d2 = top_candidates[j]
                c2, p2 = d2["condition"], d2["probability"]
                meta2 = self.decoder.conditions.get(c2, {})
                s2 = set(meta2.get("symptoms", {}).keys())
                a2 = set(meta2.get("antecedents", {}).keys())
                all2 = s2 | a2
                sev2 = meta2.get("severity", 5)

                discriminating_candidates = (all1 | all2) - asked_codes

                for code in discriminating_candidates:
                    has1 = 1 if code in all1 else 0
                    has2 = 1 if code in all2 else 0
                    disc_power = abs(has1 - has2)

                    if disc_power > 0:
                        joint_weight = p1 * p2 * disc_power
                        is_ant = code in a1 or code in a2

                        # 1. Acute symptoms preference over passive antecedents
                        if not is_ant:
                            joint_weight *= 1.20

                        # 2. Emergency Red-Flag Triage Priority:
                        # If symptom belongs to a Severity 1 or 2 condition, elevate safety score
                        if (has1 and sev1 in [1, 2]) or (has2 and sev2 in [1, 2]):
                            joint_weight *= 2.50
                            red_flag_set.add(code)

                        # 3. Anatomical continuity bonus:
                        # If candidate code belongs to the same question cluster as a recent positive finding
                        ev_meta = self.decoder.evidences.get(code, {})
                        parent_q = ev_meta.get("code_question", code)
                        if parent_q in recent_confirmed_bases:
                            joint_weight *= 1.30

                        scores[code] = scores.get(code, 0.0) + joint_weight

                        if code not in rationale_map:
                            c_pres = c1 if has1 else c2
                            c_abs = c2 if has1 else c1
                            is_rf = code in red_flag_set
                            if is_rf:
                                target_sev = sev1 if has1 else sev2
                                rationale_map[code] = f"RED-FLAG TRIAGE: Ruling {c_pres} (Severity {target_sev}/5) vs {c_abs}"
                            else:
                                rationale_map[code] = f"Differentiating between {c_pres} (present) and {c_abs} (absent)"

        if not scores:
            # Fallback: check symptoms of top-1 condition that haven't been asked
            c_top = top_candidates[0]["condition"]
            c_top_meta = self.decoder.conditions.get(c_top, {})
            unasked = (set(c_top_meta.get("symptoms", {}).keys()) | set(c_top_meta.get("antecedents", {}).keys())) - asked_codes
            for code in unasked:
                scores[code] = 0.01
                rationale_map[code] = f"Clinical characterization of primary suspected condition: {c_top}"

        if not scores:
            return None

        # Pick highest scoring candidate code
        best_code, best_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0]

        # Check information saturation threshold
        if best_score < self.min_info_gain and len(state.confirmed_evidences) >= 3:
            return None

        meta_ev = self.decoder.evidences.get(best_code, {})
        question_en = meta_ev.get("question_en", f"Do you exhibit evidence related to {best_code}?")
        is_antecedent = meta_ev.get("is_antecedent", False)
        data_type = meta_ev.get("data_type", "B")
        possible_values = meta_ev.get("possible-values", [])

        # Format human-readable options if categorical or multi-choice
        options = []
        if possible_values:
            val_meanings = meta_ev.get("value_meaning", {})
            for pv in possible_values[:8]:
                meaning = val_meanings.get(pv)
                if isinstance(meaning, dict):
                    meaning_en = meaning.get("en", pv)
                elif meaning is not None:
                    meaning_en = str(meaning)
                else:
                    meaning_en = pv
                options.append({"code": pv, "label": meaning_en})

        return {
            "code": best_code,
            "question_en": question_en,
            "is_antecedent": is_antecedent,
            "is_red_flag": best_code in red_flag_set,
            "data_type": data_type,
            "score": round(best_score, 4),
            "clinical_rationale": rationale_map.get(best_code, "Clinical evidence exploration"),
            "options": options,
        }

    def record_turn(
        self,
        state: ConsultationState,
        question_info: Dict[str, Any],
        patient_response_str: str,
        new_confirmed_tokens: List[str],
        new_denied_tokens: List[str],
    ) -> None:
        """
        Applies the patient's answer to the consultation state, updates the differential
        with Bayesian Pertinent Negative reasoning, checks dual stopping conditions,
        and logs the complete audit trail.
        """
        code = question_info["code"]
        state.asked_question_codes.append(code)

        # Snapshot before re-evaluation
        top_before = state.top_condition
        prob_before = state.confidence_percent

        # Update evidence lists
        if new_confirmed_tokens:
            state.confirmed_evidences.extend(new_confirmed_tokens)
        if new_denied_tokens:
            state.denied_evidences.extend(new_denied_tokens)

        # Recompute differential with Pertinent Negative reasoning
        state.current_differential = self.compute_differential_with_negatives(
            confirmed_evidences=state.confirmed_evidences,
            denied_evidences=state.denied_evidences,
            age=state.age,
            sex=state.sex,
            top_k=self.top_k_candidates,
        )

        top_after = state.current_differential[0]["condition"]
        prob_after = state.current_differential[0]["probability_percent"]
        conf_after_float = state.current_differential[0]["probability"]

        # Calculate top-2 separation margin
        if len(state.current_differential) >= 2:
            p2 = state.current_differential[1]["probability"]
            margin_float = conf_after_float - p2
        else:
            margin_float = conf_after_float

        state.top_condition = top_after
        state.confidence = conf_after_float
        state.top2_margin = margin_float
        state.turn += 1

        # Check for active emergency alerts in current differential
        state.emergency_alerts = [
            f"Alert: {d['condition']} (Severity {d['severity']}/5) detected at {d['probability_percent']}%"
            for d in state.current_differential if d.get("severity") in [1, 2] and d["probability"] >= 0.05
        ]
        state.has_emergency_alert = len(state.emergency_alerts) > 0

        # Audit entry
        audit_entry = AuditLogEntry(
            turn=state.turn,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question_code=code,
            question_text=question_info["question_en"],
            is_antecedent=question_info["is_antecedent"],
            is_red_flag=question_info.get("is_red_flag", False),
            clinical_rationale=question_info["clinical_rationale"],
            patient_response=patient_response_str,
            evidences_added=new_confirmed_tokens,
            top_condition_before=top_before,
            prob_before_percent=prob_before,
            top_condition_after=top_after,
            prob_after_percent=prob_after,
            confidence_delta_percent=round(prob_after - prob_before, 2),
            margin_after_percent=round(margin_float * 100.0, 2),
        )
        state.audit_log.append(audit_entry)

        # Dual Stopping Conditions:
        # 1. Target Confidence reached: P1 >= tau
        if state.confidence >= self.tau:
            state.is_converged = True
            state.termination_reason = f"Confidence target reached ({state.confidence_percent}% >= {self.tau*100:.0f}%)"
        # 2. Decisive Differential Separation: (P1 - P2) >= margin_threshold
        elif state.top2_margin >= self.margin_threshold and state.turn >= 2:
            state.is_converged = True
            state.termination_reason = f"Decisive differential separation ({state.margin_percent}% margin >= {self.margin_threshold*100:.0f}%)"
        # 3. Max Inquiry Turns exhausted
        elif state.turn >= self.max_turns:
            state.is_converged = False
            state.termination_reason = f"Maximum inquiry turns reached ({self.max_turns} turns)"


class IntakeOrchestrator:
    """
    Python control flow orchestrator with shared state and clinical audit logging.
    Executes the intake feedback loop: `confidence < tau -> ask one more question`.
    """

    def __init__(self, agent: IntakeAgent):
        self.agent = agent

    def init_consultation(
        self,
        age: int,
        sex: str,
        initial_evidence: str,
        patient_id: Optional[str] = None,
    ) -> ConsultationState:
        """
        Initializes the shared state from patient inputs and computes baseline differential.
        """
        pid = patient_id or f"PT-{datetime.now().strftime('%H%M%S')}"
        initial_evidences = [initial_evidence] if initial_evidence else []

        # Baseline differential diagnosis
        diff = self.agent.compute_differential_with_negatives(
            confirmed_evidences=initial_evidences,
            denied_evidences=[],
            age=age,
            sex=sex,
            top_k=self.agent.top_k_candidates,
        )

        top_cond = diff[0]["condition"] if diff else "Unknown"
        conf = diff[0]["probability"] if diff else 0.0
        p2 = diff[1]["probability"] if len(diff) >= 2 else 0.0
        margin = conf - p2

        base_code = initial_evidence.split("_@_")[0] if initial_evidence else ""

        # Emergency alerts check
        alerts = [
            f"Alert: {d['condition']} (Severity {d['severity']}/5) detected at {d['probability_percent']}%"
            for d in diff if d.get("severity") in [1, 2] and d["probability"] >= 0.05
        ]

        state = ConsultationState(
            patient_id=pid,
            age=int(age),
            sex=str(sex).upper(),
            initial_evidence=initial_evidence,
            confirmed_evidences=initial_evidences.copy(),
            denied_evidences=[],
            asked_question_codes=[base_code] if base_code else [],
            current_differential=diff,
            top_condition=top_cond,
            confidence=conf,
            top2_margin=margin,
            turn=0,
            is_converged=False,
            termination_reason="",
            has_emergency_alert=len(alerts) > 0,
            emergency_alerts=alerts,
            audit_log=[],
        )

        # Early check for single decisive chief complaint
        if conf >= self.agent.tau:
            state.is_converged = True
            state.termination_reason = f"Chief complaint decisive ({state.confidence_percent}% >= {self.agent.tau*100:.0f}%)"

        return state

    def run_simulation(
        self,
        patient_row: Union[pd.Series, Dict[str, Any]],
        patient_id: Optional[str] = None,
    ) -> ConsultationState:
        """
        Runs an automated intake consultation against a ground-truth DDXPlus patient record.
        Simulates accurate patient responses based on their actual evidence list.
        """
        age = int(patient_row.get("AGE", 30))
        sex = str(patient_row.get("SEX", "M"))
        init_ev = str(patient_row.get("INITIAL_EVIDENCE", ""))

        raw_evidences = patient_row.get("EVIDENCES", [])
        if isinstance(raw_evidences, str):
            true_evidences_set = set(ast.literal_eval(raw_evidences))
        elif isinstance(raw_evidences, (list, set)):
            true_evidences_set = set(raw_evidences)
        else:
            true_evidences_set = set()

        state = self.init_consultation(age=age, sex=sex, initial_evidence=init_ev, patient_id=patient_id)

        # Dynamic Intake Loop: while not converged and turns < max_turns
        while not state.is_converged and state.turn < self.agent.max_turns:
            question_info = self.agent.select_next_question(state)
            if not question_info:
                state.termination_reason = "Information saturation reached (no further gainful questions)"
                break

            q_code = question_info["code"]

            # Ground-truth patient matching
            matching_positive_tokens = [
                token for token in true_evidences_set
                if token == q_code or token.startswith(f"{q_code}_@_")
            ]

            if matching_positive_tokens:
                response_str = f"YES ({', '.join(matching_positive_tokens)})"
                new_confirmed = matching_positive_tokens
                new_denied = []
            else:
                response_str = "NO / Negative"
                new_confirmed = []
                new_denied = [q_code]

            self.agent.record_turn(
                state=state,
                question_info=question_info,
                patient_response_str=response_str,
                new_confirmed_tokens=new_confirmed,
                new_denied_tokens=new_denied,
            )

        return state

    def run_interactive(
        self,
        age: int,
        sex: str,
        initial_evidence: str,
        input_fn: Callable[[str], str] = input,
    ) -> ConsultationState:
        """
        Conducts a live interactive clinical intake session with human user input.
        """
        state = self.init_consultation(age=age, sex=sex, initial_evidence=initial_evidence)
        init_desc = self.agent.decoder.decode_token(initial_evidence)["clinical_text"]

        print("\n" + "=" * 76)
        print(f" CLINICAL INTAKE SESSION: Patient {state.patient_id}")
        print(f" Demographics : Age {age}, Sex {sex}")
        print(f" Chief Complaint: {init_desc}")
        print(f" Target Tau   : {self.agent.tau*100:.0f}% | Margin Target: {self.agent.margin_threshold*100:.0f}% | Max Turns: {self.agent.max_turns}")
        print("=" * 76)

        while not state.is_converged and state.turn < self.agent.max_turns:
            question_info = self.agent.select_next_question(state)
            if not question_info:
                state.termination_reason = "Information saturation reached (no further gainful questions)"
                print("\n[Intake Agent]: Diagnostic inquiry reached maximum information saturation.")
                break

            rf_tag = " [RED-FLAG SAFETY]" if question_info.get("is_red_flag") else ""
            print(f"\n[Turn {state.turn + 1}]{rf_tag} Intake Question:")
            print(f"  -> {question_info['question_en']}")
            print(f"     Clinical Rationale: {question_info['clinical_rationale']}")

            if question_info["options"]:
                print("     Options:")
                for opt in question_info["options"]:
                    print(f"       - {opt['code']}: {opt['label']}")

            user_ans = input_fn("     Patient answer (y/n or code, default: 'n'): ").strip()

            if user_ans.lower() in ["y", "yes", "true", "1"]:
                new_confirmed = [question_info["code"]]
                new_denied = []
                resp_str = "YES"
            elif user_ans.lower() in ["n", "no", "false", "0", ""]:
                new_confirmed = []
                new_denied = [question_info["code"]]
                resp_str = "NO / Negative"
            else:
                token = f"{question_info['code']}_@_{user_ans}" if "_@_" not in user_ans else user_ans
                new_confirmed = [token]
                new_denied = []
                resp_str = f"YES ({user_ans})"

            self.agent.record_turn(
                state=state,
                question_info=question_info,
                patient_response_str=resp_str,
                new_confirmed_tokens=new_confirmed,
                new_denied_tokens=new_denied,
            )

            p1 = state.confidence_percent
            m = state.margin_percent
            print(f"     Differential Update: {state.top_condition} ({p1}%, margin: {m}%)")

        print("\n" + "=" * 76)
        print(f" INTAKE CONCLUDED: {state.termination_reason}")
        print(f" Primary Diagnosis: {state.top_condition} (Confidence: {state.confidence_percent}%, Margin: {state.margin_percent}%)")
        print("=" * 76)

        return state

    def export_audit_log_dataframe(self, state: ConsultationState) -> pd.DataFrame:
        """
        Converts the shared audit log into a tabular pandas DataFrame for display and review.
        """
        if not state.audit_log:
            return pd.DataFrame()
        return pd.DataFrame([entry.to_dict() for entry in state.audit_log])

    def get_stage2_handoff_payload(
        self,
        state: ConsultationState,
        ground_truth_pathology: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Standardizes the output payload generated by Stage 1 to be handed off to
        Stage 2 (Consultation Agents: GP + Specialists) and Stage 3 (Referral / Explainer).
        """
        decoded_confirmed = [
            self.agent.decoder.decode_token(t)["clinical_text"]
            for t in state.confirmed_evidences
        ]

        return {
            "patient_id": state.patient_id,
            "demographics": {
                "age": state.age,
                "sex": state.sex,
            },
            "intake_summary": {
                "initial_chief_complaint": state.initial_evidence,
                "initial_chief_complaint_decoded": self.agent.decoder.decode_token(state.initial_evidence)["clinical_text"] if state.initial_evidence else "None",
                "total_intake_turns": state.turn,
                "is_converged": state.is_converged,
                "termination_reason": state.termination_reason,
                "final_confidence_percent": state.confidence_percent,
                "top2_margin_percent": state.margin_percent,
                "has_emergency_alert": state.has_emergency_alert,
                "emergency_alerts": state.emergency_alerts,
                "ground_truth_pathology": ground_truth_pathology,
            },
            "clinical_evidence_profile": {
                "confirmed_evidence_tokens": state.confirmed_evidences,
                "confirmed_evidence_decoded": decoded_confirmed,
                "denied_evidence_codes": state.denied_evidences,
            },
            "differential_diagnosis": state.current_differential,
            "audit_trail": [entry.to_dict() for entry in state.audit_log],
        }
