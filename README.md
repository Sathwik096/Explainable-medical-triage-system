# Agentic Diagnostic Assistant & Explainable Medical Triage System

An intelligent multi-turn clinical diagnostic and triage assistant built on top of the DDXPlus benchmark dataset.

## System Architecture Overview

```text
┌─────────────────┐      ┌─────────────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│  Patient Input  │ ───► │      Intake Agent       │ ───► │ Consultation Agents  │ ───► │  Referral Agent  │
│ age, sex, init  │      │       • next best       │      │ PLANNED • GP + spec. │      │ PLANNED • urg.   │
└─────────────────┘      │        question         │      └──────────────────────┘      └────────┬─────────┘
                         └───────────┬─────────────┘                                             │
                                     │                                                           ▼
                             confidence < tau                                           ┌──────────────────┐
                                     │                                                  │ Explainer Agent  │
                             ask one more question ◄──┐                                 │ PLANNED • report │
                                     │                │                                 └──────────────────┘
                                     └────────────────┘
```

- **Stage 1 (Implemented)**: **Patient Input + Intake Agent + Bayesian Negative Reasoning + Red-Flag Triage + Audit Trail**
- **Stage 2 (Planned)**: Consultation Agents (General Practitioner + Specialists)
- **Stage 3 (Planned)**: Referral Agent (Urgency scoring + Destination routing)
- **Stage 4 (Planned)**: Explainer Agent (Plain-language clinical justification)

---

## Project Structure
```text
Explainable-medical-triage-system/
│
├── data/
│   ├── raw/
│   │   ├── release_conditions.json         # Pathology metadata (49 conditions, ICD-10, severity, symptoms)
│   │   └── release_evidences.json          # Evidence metadata (223 codes, questions, data types, value mappings)
│   └── processed/
│       ├── ddxplus_train_10k.csv           # 10,000-row representative development split
│       └── ddxplus_train_10k_decoded.csv   # Decoded clinical features, symptoms, and antecedents
│
├── models/
│   └── diagnostic_model.joblib             # Trained multi-class diagnostic model artifact
│
├── notebooks/
│   ├── 01_ddxplus_exploration.ipynb        # Step 1: Raw dataset exploration & statistics
│   ├── 02_clinical_decoding_pipeline.ipynb # Step 1A: Clinical decoding & vignette generation
│   ├── 03_diagnostic_baseline.ipynb        # Step 2: Diagnostic baseline training & simulation
│   └── 04_stage1_intake_agent.ipynb        # Step 3: Stage 1 Multi-turn Intake Agent & Audit Trail
│
├── src/
│   ├── __init__.py                         # Package entry point
│   ├── decoder.py                          # DDXPlusDecoder class
│   ├── diagnostic_engine.py                # DiagnosticEngine probabilistic classifier
│   └── intake_agent.py                     # Advanced IntakeAgent, IntakeOrchestrator, ConsultationState
│
├── tests/
│   ├── test_decoder.py                     # Unit tests for decoder
│   ├── test_diagnostic_engine.py           # Unit tests for diagnostic engine
│   └── test_intake_agent.py                # Unit tests for Stage 1 Intake Agent & Bayesian Negatives
│
├── app.py                                  # Interactive Streamlit Web Application
├── predict.py                              # Standalone diagnostic inference script
├── run_intake_stage.py                     # Standalone Stage 1 interactive/simulation runner
├── requirements.txt
└── README.md
```

---

## Completed Phases & Progress Tracking

### Step 1: DDXPlus Dataset Setup & Understanding
- **Columns**: `AGE`, `SEX`, `PATHOLOGY`, `EVIDENCES`, `INITIAL_EVIDENCE`, `DIFFERENTIAL_DIAGNOSIS`
- **Integrity**: 49 disease classes, 223 evidence codes, 0 missing values.
- **Evidence Structure**: Binary (`B`), Categorical (`C`), and Multi-choice (`M`) evidence formats.
- **Notebook**: [`notebooks/01_ddxplus_exploration.ipynb`](file:///notebooks/01_ddxplus_exploration.ipynb)

### Step 1A: Clinical Preprocessing & Evidence Decoder Pipeline
- Built the `DDXPlusDecoder` in [`src/decoder.py`](file:///src/decoder.py):
  - **Token Decoding**: Converts codes like `E_48`, `E_54_@_V_161`, and `E_56_@_4` into readable English clinical concepts.
  - **Clinical Categorization**: Distinguishes acute presenting symptoms from medical history and risk factors (antecedents).
  - **Vignette Generator**: Produces formatted physician-grade clinical notes (`to_clinical_vignette()`).
- **Notebook**: [`notebooks/02_clinical_decoding_pipeline.ipynb`](file:///notebooks/02_clinical_decoding_pipeline.ipynb)

### Step 2: Diagnostic Baseline Engine
- Built the `DiagnosticEngine` in [`src/diagnostic_engine.py`](file:///src/diagnostic_engine.py):
  - **Probabilistic Inference**: Produces probability distributions over all 49 pathologies.
  - **Benchmark Accuracy**:
    - **Top-1 Accuracy**: **99.45%**
    - **Top-3 Accuracy**: **100.00%**
    - **Top-5 Accuracy**: **100.00%**
  - **Artifact**: Serialized to [`models/diagnostic_model.joblib`](file:///models/diagnostic_model.joblib).
- **Notebook**: [`notebooks/03_diagnostic_baseline.ipynb`](file:///notebooks/03_diagnostic_baseline.ipynb)

### Step 3 / Stage 1: Advanced Clinical Intake Agent & Dynamic Inquiry Loop
- Built `IntakeAgent`, `IntakeOrchestrator`, and `ConsultationState` in [`src/intake_agent.py`](file:///src/intake_agent.py):
  - **Bayesian Pertinent Negative Reasoning**: Denied symptoms actively down-weight diseases expecting that symptom, accelerating diagnostic convergence.
  - **Red-Flag Emergency Prioritization**: Life-safety questions for Severity 1 & 2 conditions (e.g., Laryngospasm, Anaphylaxis, Boerhaave, Epiglottitis) are automatically elevated before exploring benign symptoms.
  - **Dual Stopping Policy**: Concludes inquiry when top candidate confidence crosses $\tau$ (default 80%) **OR** when differential margin $(P_1 - P_2) \ge 0.55$.
  - **Tamper-Evident Clinical Audit Log**: Records every question, natural language clinical text, clinical rationale, patient response, top condition before/after, and confidence delta for physician review.
  - **Standardized Stage 2 Handoff Payload**: Packages confirmed/denied symptoms, updated differential ranking, and consultation history for downstream Consultation Agents (GP + Specialists).
- **Interactive UI**: [`app.py`](file:///app.py) (Streamlit Clinical Dashboard).
- **Notebook**: [`notebooks/04_stage1_intake_agent.ipynb`](file:///notebooks/04_stage1_intake_agent.ipynb).
- **Runner Script**: [`run_intake_stage.py`](file:///run_intake_stage.py).

---

## Execution Commands

### 1. Launch Interactive Streamlit Web Dashboard
```powershell
streamlit run app.py
```
*Provides a visual patient consultation simulator with real-time probability bars, emergency red-flag badges, uncertainty collapse charts, and audit trail tables.*

### 2. Run Automated Unit Tests (15 Tests)
```powershell
python -m unittest discover tests
```

### 3. Run Standalone Stage 1 Runner (Terminal)
```powershell
# Simulate Intake on Patient 0
python run_intake_stage.py --patient-idx 0

# Simulate on any patient index with custom threshold tau & margin
python run_intake_stage.py --patient-idx 5 --tau 0.85 --margin 0.60 --max-turns 10

# Run Interactive Live Intake Session
python run_intake_stage.py --interactive
```

### 4. Run Standalone Diagnostic Inference
```powershell
python predict.py
```

### 5. Explore Jupyter Notebooks
Launch Jupyter to explore each stage step-by-step:
```powershell
jupyter lab
```
Or open [`notebooks/04_stage1_intake_agent.ipynb`](file:///notebooks/04_stage1_intake_agent.ipynb) directly in VS Code.
