"""
Explainable Medical Triage System - Interactive Stage 1 Clinical Dashboard
========================================================================
Streamlit Web Application to showcase:
- Patient Input ingestion (Demographics & Chief Complaint)
- Multi-Turn Intake Agent with Next-Best Question policy
- Real-time Bayesian Pertinent Negative Reasoning
- Red-Flag Emergency Triage Safety Alerts
- Dynamic Diagnostic Uncertainty Collapse Visualization
- Chronological Tamper-Evident Clinical Audit Log
- Standardized Stage 2 Handoff Payload Export
"""

import ast
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

from src.decoder import DDXPlusDecoder
from src.diagnostic_engine import DiagnosticEngine
from src.intake_agent import IntakeAgent, IntakeOrchestrator, ConsultationState

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Explainable Medical Triage | Stage 1",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 10px;
    }
    .emergency-card {
        background-color: #FEF2F2;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 5px solid #EF4444;
        margin-bottom: 15px;
    }
    .success-card {
        background-color: #F0FDF4;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 5px solid #22C55E;
        margin-bottom: 15px;
    }
    .badge-sev1 { background-color: #EF4444; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
    .badge-sev2 { background-color: #F97316; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
    .badge-sev3 { background-color: #FBBF24; color: black; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; }
    .badge-sev4 { background-color: #3B82F6; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
    .badge-sev5 { background-color: #6B7280; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Resource Loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_system():
    base_dir = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
    decoder = DDXPlusDecoder(
        evidences_path=base_dir / "data" / "raw" / "release_evidences.json",
        conditions_path=base_dir / "data" / "raw" / "release_conditions.json",
    )
    engine = DiagnosticEngine(
        decoder=decoder,
        model_path=base_dir / "models" / "diagnostic_model.joblib",
    )
    return decoder, engine, base_dir


@st.cache_data
def load_patient_samples(base_dir):
    df = pd.read_csv(base_dir / "data" / "processed" / "ddxplus_train_10k.csv")
    return df


decoder, engine, base_dir = load_system()
df_patients = load_patient_samples(base_dir)


# -----------------------------------------------------------------------------
# Sidebar: Configuration & Controls
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/stethoscope.png", width=64)
st.sidebar.title("Stage 1 Controller")
st.sidebar.caption("Intake Agent & Dynamic Inquiry Loop")

mode = st.sidebar.radio(
    "Consultation Mode",
    ["🩺 DDXPlus Benchmark Simulation", "✍️ Interactive Custom Patient"],
)

st.sidebar.divider()
st.sidebar.subheader("Policy Hyperparameters")
tau = st.sidebar.slider("Confidence Target (tau)", min_value=0.50, max_value=0.95, value=0.80, step=0.05)
margin_thresh = st.sidebar.slider("Separation Margin Target", min_value=0.20, max_value=0.80, value=0.55, step=0.05)
max_turns = st.sidebar.slider("Maximum Inquiry Turns", min_value=3, max_value=12, value=8, step=1)
neg_penalty = st.sidebar.slider("Bayesian Negative Penalty", min_value=0.20, max_value=0.80, value=0.50, step=0.05)

agent = IntakeAgent(
    engine=engine,
    decoder=decoder,
    tau=tau,
    margin_threshold=margin_thresh,
    max_turns=max_turns,
    negative_penalty=neg_penalty,
)
orchestrator = IntakeOrchestrator(agent=agent)


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown('<div class="main-header">Explainable Medical Triage System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Stage 1: Multi-Turn Clinical Intake Agent & Dynamic Uncertainty Collapse</div>', unsafe_allow_html=True)

# Architecture Progress Breadcrumb
st.markdown("""
<div style="background-color: #EEF2F6; padding: 10px 16px; border-radius: 8px; margin-bottom: 20px; font-size: 0.9rem;">
    <b>Pipeline:</b> &nbsp;
    <span style="color: #1E3A8A; font-weight: bold;">[Active] Patient Input ──► Intake Agent (Dynamic Inquiry)</span> &nbsp;──►&nbsp; 
    <span style="color: #9CA3AF;">[Stage 2] Consultation Agents</span> &nbsp;──►&nbsp; 
    <span style="color: #9CA3AF;">[Stage 3] Referral</span> &nbsp;──►&nbsp; 
    <span style="color: #9CA3AF;">[Stage 4] Explainer</span>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "consultation_state" not in st.session_state:
    st.session_state.consultation_state = None
if "selected_patient_idx" not in st.session_state:
    st.session_state.selected_patient_idx = 0


# -----------------------------------------------------------------------------
# Mode 1: Benchmark Case Simulation
# -----------------------------------------------------------------------------
if mode == "🩺 DDXPlus Benchmark Simulation":
    col_sel, col_act = st.columns([3, 1])
    with col_sel:
        preset_idx = st.selectbox(
            "Select Benchmark Patient Record (from 10k representative split):",
            options=list(range(20)),
            format_func=lambda i: f"Patient #{i}: {df_patients.iloc[i]['PATHOLOGY']} (Age {df_patients.iloc[i]['AGE']}, {df_patients.iloc[i]['SEX']}) - Chief: {df_patients.iloc[i]['INITIAL_EVIDENCE']}",
        )
    with col_act:
        st.write("")
        st.write("")
        if st.button("🚀 Start Consultation", use_container_width=True, type="primary"):
            st.session_state.selected_patient_idx = preset_idx
            row = df_patients.iloc[preset_idx]
            st.session_state.consultation_state = orchestrator.init_consultation(
                age=row["AGE"],
                sex=row["SEX"],
                initial_evidence=row["INITIAL_EVIDENCE"],
                patient_id=f"DDX-{preset_idx:04d}",
            )

    curr_row = df_patients.iloc[st.session_state.selected_patient_idx]
    true_pathology = curr_row["PATHOLOGY"]
    true_evs = set(ast.literal_eval(curr_row["EVIDENCES"]))

    if st.session_state.consultation_state is not None:
        state: ConsultationState = st.session_state.consultation_state

        # Top Clinical Summary Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Demographics", f"Age {state.age}, {state.sex}")
        with c2:
            st.metric("Chief Complaint", state.initial_evidence, help=decoder.decode_token(state.initial_evidence)["clinical_text"])
        with c3:
            st.metric("Top Candidate", state.top_condition, f"{state.confidence_percent}%")
        with c4:
            st.metric("Inquiry Turns", f"{state.turn} / {max_turns}", f"Margin: {state.margin_percent}%")

        # Emergency Alert Banner
        if state.has_emergency_alert:
            st.markdown(f"""
            <div class="emergency-card">
                <b>⚠️ CLINICAL EMERGENCY SAFETY ALERT:</b> High-severity conditions detected in differential!<br/>
                {"<br/>".join(state.emergency_alerts)}
            </div>
            """, unsafe_allow_html=True)

        if state.is_converged:
            st.markdown(f"""
            <div class="success-card">
                <b>✅ INTAKE CONCLUDED:</b> {state.termination_reason}<br/>
                <b>Final Working Diagnosis:</b> {state.top_condition} ({state.confidence_percent}%) &nbsp;|&nbsp; 
                <b>Ground Truth:</b> {true_pathology} ({"MATCH" if state.top_condition == true_pathology else "DISCREPANCY"})
            </div>
            """, unsafe_allow_html=True)

        # Main Layout: Q&A on left, Live Differential on right
        col_qa, col_diff = st.columns([3, 2])

        with col_qa:
            st.subheader("Interactive Inquiry Loop")

            if not state.is_converged and state.turn < max_turns:
                next_q = agent.select_next_question(state)

                if next_q:
                    rf_badge = "🚨 RED-FLAG SAFETY QUESTION" if next_q.get("is_red_flag") else "📋 Clinical Inquiry"
                    st.info(f"**Turn {state.turn + 1} • {rf_badge}**\n\n### {next_q['question_en']}\n\n*Rationale: {next_q['clinical_rationale']}*")

                    # Simulation Action Buttons
                    q_code = next_q["code"]
                    matching = [t for t in true_evs if t == q_code or t.startswith(f"{q_code}_@_")]

                    col_b1, col_b2, col_b3 = st.columns([1, 1, 2])
                    with col_b1:
                        if st.button("Step Simulator", type="primary", use_container_width=True):
                            if matching:
                                resp = f"YES ({', '.join(matching)})"
                                new_conf = matching
                                new_den = []
                            else:
                                resp = "NO"
                                new_conf = []
                                new_den = [q_code]

                            agent.record_turn(state, next_q, resp, new_conf, new_den)
                            st.rerun()

                    with col_b2:
                        if st.button("Fast-Forward", use_container_width=True):
                            while not state.is_converged and state.turn < max_turns:
                                q_step = agent.select_next_question(state)
                                if not q_step:
                                    state.termination_reason = "Information saturation reached"
                                    break
                                m_step = [t for t in true_evs if t == q_step["code"] or t.startswith(f"{q_step['code']}_@_")]
                                r_step = f"YES ({', '.join(m_step)})" if m_step else "NO"
                                agent.record_turn(state, q_step, r_step, m_step if m_step else [], [] if m_step else [q_step["code"]])
                            st.rerun()

                    with col_b3:
                        st.caption(f"Patient Ground Truth response to `{q_code}`: **{'YES' if matching else 'NO'}**")
                else:
                    st.warning("No further gainful discriminative questions available. Inquiry saturated.")
            else:
                st.success("Consultation complete. Ready for clinician review and Stage 2 handoff.")

            # Clinical Audit Log
            st.subheader("Tamper-Evident Clinical Audit Log")
            audit_df = orchestrator.export_audit_log_dataframe(state)
            if not audit_df.empty:
                disp_df = audit_df[["turn", "question_code", "is_red_flag", "patient_response", "top_condition_after", "prob_after_percent", "margin_after_percent", "confidence_delta_percent"]]
                st.dataframe(disp_df, use_container_width=True, hide_index=True)

        with col_diff:
            st.subheader("Live Differential Diagnosis")
            for rank, item in enumerate(state.current_differential[:5], 1):
                p_val = item["probability"]
                p_pct = item["probability_percent"]
                sev = item.get("severity", 4)
                sev_badge = f'<span class="badge-sev{sev}">Sev {sev}/5</span>'

                st.markdown(f"**{rank}. {item['condition']}** &nbsp; {sev_badge} &nbsp; `ICD-10: {item['icd10']}`", unsafe_allow_html=True)
                st.progress(min(1.0, float(p_val)))
                st.caption(f"Confidence: {p_pct}%")

            # Uncertainty Collapse Chart
            st.subheader("Uncertainty Collapse")
            if len(state.audit_log) > 0:
                chart_data = pd.DataFrame({
                    "Turn": [0] + [e.turn for e in state.audit_log],
                    "Confidence": [state.audit_log[0].prob_before_percent] + [e.prob_after_percent for e in state.audit_log],
                    "Top-2 Margin": [0.0] + [e.margin_after_percent for e in state.audit_log],
                })
                st.line_chart(chart_data.set_index("Turn"), color=["#1E3A8A", "#10B981"])

        # Stage 2 Handoff Section
        st.divider()
        st.subheader("Standardized Stage 2 Handoff Payload (Consultation Agents)")
        handoff_payload = orchestrator.get_stage2_handoff_payload(state, ground_truth_pathology=true_pathology)

        with st.expander("🔍 Inspect Full Stage 2 JSON Payload (Click to Expand)", expanded=False):
            st.json(handoff_payload)
            st.download_button(
                "📥 Download Handoff JSON",
                data=json.dumps(handoff_payload, indent=2),
                file_name=f"stage2_handoff_{state.patient_id}.json",
                mime="application/json",
            )


# -----------------------------------------------------------------------------
# Mode 2: Interactive Custom Patient
# -----------------------------------------------------------------------------
else:
    st.subheader("Manual Clinical Consultation Simulator")
    st.caption("Enter patient demographics and chief complaint to conduct a live interactive triage intake session.")

    c_age, c_sex, c_init = st.columns([1, 1, 2])
    with c_age:
        custom_age = st.number_input("Patient Age", min_value=1, max_value=105, value=45)
    with c_sex:
        custom_sex = st.selectbox("Biological Sex", ["M", "F"])
    with c_init:
        # Predefined common chief complaints for quick selection
        common_symptoms = {
            "E_201": "Cough (E_201)",
            "E_91": "Fever (E_91)",
            "E_53": "Pain somewhere (E_53)",
            "E_169": "Itchy nose / throat (E_169)",
            "E_66": "Severe shortness of breath (E_66)",
            "E_45": "Coughing up blood (E_45)",
            "E_181": "Nasal congestion / clear runny nose (E_181)",
        }
        custom_ev = st.selectbox(
            "Chief Presenting Complaint",
            options=list(common_symptoms.keys()),
            format_func=lambda k: common_symptoms[k],
        )

    if st.button("Start Live Intake", type="primary"):
        st.session_state.consultation_state = orchestrator.init_consultation(
            age=custom_age,
            sex=custom_sex,
            initial_evidence=custom_ev,
            patient_id=f"LIVE-PT-01",
        )
        st.rerun()

    if st.session_state.consultation_state is not None:
        state: ConsultationState = st.session_state.consultation_state

        c_l, c_r = st.columns([3, 2])

        with c_l:
            st.subheader(f"Intake Consultation: Patient {state.patient_id}")
            st.write(f"**Chief Complaint:** {decoder.decode_token(state.initial_evidence)['clinical_text']}")

            if not state.is_converged and state.turn < max_turns:
                next_q = agent.select_next_question(state)

                if next_q:
                    rf_badge = "🚨 RED-FLAG SAFETY" if next_q.get("is_red_flag") else "📋 Clinical Inquiry"
                    st.info(f"### Turn {state.turn + 1}: {next_q['question_en']}\n\n*Clinical intent: {next_q['clinical_rationale']}*")

                    ans_col1, ans_col2 = st.columns(2)
                    with ans_col1:
                        if st.button("👍 Yes / Present", use_container_width=True, type="primary"):
                            agent.record_turn(state, next_q, "YES", [next_q["code"]], [])
                            st.rerun()
                    with ans_col2:
                        if st.button("👎 No / Denied", use_container_width=True):
                            agent.record_turn(state, next_q, "NO", [], [next_q["code"]])
                            st.rerun()
                else:
                    st.warning("Diagnostic inquiries exhausted.")
            else:
                st.success(f"**Intake Complete:** {state.termination_reason}")
                st.write(f"**Final Top Candidate:** {state.top_condition} ({state.confidence_percent}%)")

            # Audit table
            audit_df = orchestrator.export_audit_log_dataframe(state)
            if not audit_df.empty:
                st.subheader("Consultation Audit Log")
                st.dataframe(audit_df[["turn", "question_code", "is_red_flag", "patient_response", "top_condition_after", "prob_after_percent"]], use_container_width=True)

        with c_r:
            st.subheader("Current Differential Ranking")
            for rank, item in enumerate(state.current_differential[:5], 1):
                st.write(f"**{rank}. {item['condition']}** ({item['probability_percent']}%) - `Severity {item['severity']}/5`")
                st.progress(float(item["probability"]))

            if state.has_emergency_alert:
                st.error("⚠️ Red-flag conditions active in differential diagnosis!")

