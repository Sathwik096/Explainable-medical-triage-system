"""
Stage 1: Multi-Turn Clinical Intake Agent - Standalone Execution Script
======================================================================
Executes Stage 1 of the Explainable Medical Triage System with:
- Bayesian Pertinent Negative Reasoning (Active Rule-Out)
- Dual Stopping Policy (Tau + Top-2 Separation Margin)
- Red-Flag Emergency Triage Priority
- Full Chronological Clinical Audit Trail Table
- Stage 2 Standardized Handoff Payload
"""

import argparse
import ast
import json
import pandas as pd

from src.decoder import DDXPlusDecoder
from src.diagnostic_engine import DiagnosticEngine
from src.intake_agent import IntakeAgent, IntakeOrchestrator


def print_banner(text: str) -> None:
    print("\n" + "=" * 82)
    print(f" {text}")
    print("=" * 82)


def print_audit_table(audit_df: pd.DataFrame) -> None:
    if audit_df.empty:
        print("Audit log is empty.")
        return

    cols = ["turn", "question_code", "is_red_flag", "patient_response", "top_condition_after", "prob_after_percent", "margin_after_percent", "confidence_delta_percent"]
    sub_df = audit_df[cols]
    
    print("\n--- CLINICAL AUDIT LOG (Chronological Review) ---")
    header = f"{'Turn':<5} | {'Code':<8} | {'RedFlag':<7} | {'Patient Answer':<18} | {'Top Diagnosis':<24} | {'Conf':<7} | {'Margin':<7} | {'Delta':<7}"
    print(header)
    print("-" * len(header))
    for _, row in sub_df.iterrows():
        t = row["turn"]
        c = row["question_code"]
        rf = "ALERT" if row["is_red_flag"] else "No"
        ans = str(row["patient_response"])[:16]
        top_d = str(row["top_condition_after"])[:22]
        conf = f"{row['prob_after_percent']:.1f}%"
        margin = f"{row['margin_after_percent']:.1f}%"
        delta = f"{row['confidence_delta_percent']:+.1f}%"
        print(f"{t:<5} | {c:<8} | {rf:<7} | {ans:<18} | {top_d:<24} | {conf:<7} | {margin:<7} | {delta:<7}")
    print("-" * len(header))


def run_simulated_case(
    patient_idx: int = 0,
    tau: float = 0.80,
    margin_threshold: float = 0.55,
    max_turns: int = 8,
) -> None:
    print_banner(f"STAGE 1: ADVANCED INTAKE AGENT - PATIENT #{patient_idx} SIMULATION")
    
    print("[1/4] Loading Clinical Knowledge Base & Diagnostic Engine...")
    decoder = DDXPlusDecoder()
    engine = DiagnosticEngine(decoder=decoder, model_path="models/diagnostic_model.joblib")
    agent = IntakeAgent(
        engine=engine,
        decoder=decoder,
        tau=tau,
        margin_threshold=margin_threshold,
        max_turns=max_turns,
    )
    orchestrator = IntakeOrchestrator(agent=agent)

    df = pd.read_csv("data/processed/ddxplus_train_10k.csv")
    patient = df.iloc[patient_idx]
    
    age = patient["AGE"]
    sex = patient["SEX"]
    init_ev = patient["INITIAL_EVIDENCE"]
    true_pathology = patient["PATHOLOGY"]
    true_evs = ast.literal_eval(patient["EVIDENCES"])

    init_decoded = decoder.decode_token(init_ev)["clinical_text"]
    print(f"\n[Patient Profile]")
    print(f"  - Demographics : Age {age}, Sex {sex}")
    print(f"  - Chief Complaint: {init_ev} ({init_decoded})")
    print(f"  - Ground Truth : {true_pathology}")
    print(f"  - Total Real Evidences: {len(true_evs)}")
    print(f"  - Target Tau   : {tau*100:.0f}% | Margin Target: {margin_threshold*100:.0f}% | Max Allowed Turns: {max_turns}")

    print("\n[2/4] Executing Stage 1 Dynamic Intake Loop...")
    state = orchestrator.init_consultation(age=age, sex=sex, initial_evidence=init_ev, patient_id=f"DDX-PT-{patient_idx}")
    print(f"  Turn 0 Baseline: {state.top_condition} ({state.confidence_percent}%, margin: {state.margin_percent}%)")

    while not state.is_converged and state.turn < max_turns:
        q_info = agent.select_next_question(state)
        if not q_info:
            state.termination_reason = "Information saturation reached (no further gainful questions)"
            break

        q_code = q_info["code"]
        q_text = q_info["question_en"]
        rationale = q_info["clinical_rationale"]
        is_rf = q_info.get("is_red_flag", False)

        matching = [t for t in true_evs if t == q_code or t.startswith(f"{q_code}_@_")]
        if matching:
            resp = f"YES ({', '.join(matching)})"
            new_conf = matching
            new_den = []
        else:
            resp = "NO"
            new_conf = []
            new_den = [q_code]

        agent.record_turn(
            state=state,
            question_info=q_info,
            patient_response_str=resp,
            new_confirmed_tokens=new_conf,
            new_denied_tokens=new_den,
        )

        turn_log = state.audit_log[-1]
        rf_badge = " [RED-FLAG SAFETY ALERT]" if is_rf else ""
        print(f"\n  [Turn {turn_log.turn}]{rf_badge} Next Best Question: [{q_code}] {q_text}")
        print(f"    - Rationale : {rationale}")
        print(f"    - Answer    : {resp}")
        print(f"    - Updated Diff: {state.top_condition} ({state.confidence_percent}%, margin: {state.margin_percent}%) [Delta: {turn_log.confidence_delta_percent:+.1f}%]")

    # 4. Results & Clinician Review
    print_banner("STAGE 1 COMPLETED: CLINICAL INTAKE SUMMARY")
    print(f"  Termination Reason : {state.termination_reason}")
    print(f"  Total Turns Asked  : {state.turn}")
    print(f"  Confidence Achieved: {state.confidence_percent}% (Target: {tau*100:.0f}%)")
    print(f"  Differential Margin: {state.margin_percent}% (Target: {margin_threshold*100:.0f}%)")
    print(f"  Top Candidate      : {state.top_condition}")
    print(f"  Ground Truth Match : {'MATCH (Correct Diagnosis)' if state.top_condition == true_pathology else 'DIFFERENT'}")

    if state.has_emergency_alert:
        print("\n  [EMERGENCY SAFETY TRIAGE]")
        for alert in state.emergency_alerts:
            print(f"    ! {alert}")

    print("\n[Final Top-5 Differential Diagnosis]")
    for rank, d in enumerate(state.current_differential[:5], 1):
        print(f"  {rank}. {d['condition']:<28} | Prob: {d['probability_percent']:>5.1f}% | ICD-10: {d['icd10']:<8} | Severity: {d['severity']}/5")

    audit_df = orchestrator.export_audit_log_dataframe(state)
    print_audit_table(audit_df)

    handoff = orchestrator.get_stage2_handoff_payload(state, ground_truth_pathology=true_pathology)
    print("\n[Stage 2 Handoff Payload Preview]")
    preview = {
        "patient_id": handoff["patient_id"],
        "demographics": handoff["demographics"],
        "intake_summary": handoff["intake_summary"],
        "confirmed_symptoms_count": len(handoff["clinical_evidence_profile"]["confirmed_evidence_tokens"]),
        "denied_symptoms_count": len(handoff["clinical_evidence_profile"]["denied_evidence_codes"]),
        "top_diagnosis": handoff["differential_diagnosis"][0]["condition"] if handoff["differential_diagnosis"] else None,
        "ready_for_stage_2_consultation_agents": True,
    }
    print(json.dumps(preview, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Run Stage 1: Multi-Turn Clinical Intake Agent")
    parser.add_argument("--patient-idx", type=int, default=0, help="Index of patient from ddxplus_train_10k.csv (default: 0)")
    parser.add_argument("--tau", type=float, default=0.80, help="Confidence convergence threshold (default: 0.80)")
    parser.add_argument("--margin", type=float, default=0.55, help="Top-2 separation margin threshold (default: 0.55)")
    parser.add_argument("--max-turns", type=int, default=8, help="Maximum intake inquiry turns (default: 8)")
    parser.add_argument("--interactive", action="store_true", help="Run interactive intake session with manual input")
    args = parser.parse_args()

    if args.interactive:
        decoder = DDXPlusDecoder()
        engine = DiagnosticEngine(decoder=decoder, model_path="models/diagnostic_model.joblib")
        agent = IntakeAgent(
            engine=engine,
            decoder=decoder,
            tau=args.tau,
            margin_threshold=args.margin,
            max_turns=args.max_turns,
        )
        orchestrator = IntakeOrchestrator(agent=agent)
        
        print("\n--- Enter Initial Patient Information ---")
        age_in = int(input("Age: ") or "45")
        sex_in = input("Sex (M/F): ") or "M"
        init_ev = input("Initial Evidence Token (e.g. E_201 for Cough, E_91 for Fever, E_169 for Itchy nose): ") or "E_201"
        
        state = orchestrator.run_interactive(age=age_in, sex=sex_in, initial_evidence=init_ev)
        audit_df = orchestrator.export_audit_log_dataframe(state)
        print_audit_table(audit_df)
    else:
        run_simulated_case(
            patient_idx=args.patient_idx,
            tau=args.tau,
            margin_threshold=args.margin,
            max_turns=args.max_turns,
        )


if __name__ == "__main__":
    main()
