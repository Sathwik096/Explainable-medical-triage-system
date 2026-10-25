# Agentic Diagnostic Assistant

An intelligent multi-turn clinical diagnostic assistant built on top of the DDXPlus benchmark dataset.

## Project Structure
```text
agentic-diagnostic-assistant/
│
├── data/
│   ├── raw/
│   │   ├── release_conditions.json      # Pathology metadata (49 conditions, ICD-10, severity, symptoms)
│   │   ├── release_evidences.json       # Evidence metadata (223 codes, questions, data types, value mappings)
│   │   └── release_train_patients.csv   # Full train split (1,025,602 patient records)
│   └── processed/
│       ├── ddxplus_train_10k.csv        # 10,000-row representative development split
│       └── ddxplus_train_10k_decoded.csv# Decoded clinical features, symptoms, and antecedents
│
├── models/
│   └── diagnostic_model.joblib          # Trained multi-class diagnostic model artifact
│
├── notebooks/
│   ├── 01_ddxplus_exploration.ipynb     # Step 1 interactive exploration & statistics
│   ├── 02_clinical_decoding_pipeline.ipynb # Step 1A clinical decoding & vignette generation
│   └── 03_diagnostic_baseline.ipynb     # Step 2 diagnostic baseline training & simulation
│
├── src/
│   ├── __init__.py                      # Package entry point
│   ├── decoder.py                       # DDXPlusDecoder class
│   └── diagnostic_engine.py             # DiagnosticEngine probabilistic classifier
│
├── tests/
│   ├── test_decoder.py                  # Automated unit tests for decoder
│   └── test_diagnostic_engine.py        # Automated unit tests for diagnostic engine
│
├── requirements.txt
└── README.md
```

## Step 1: DDXPlus Dataset Setup & Understanding
Explored raw dataset structure:
- **Columns**: `AGE`, `SEX`, `PATHOLOGY`, `EVIDENCES`, `INITIAL_EVIDENCE`, `DIFFERENTIAL_DIAGNOSIS`
- **Integrity**: 1,025,602 training records, 49 diseases, 0 missing values.
- **Evidence Structure**: Binary (`B`), Categorical (`C`), and Multi-choice (`M`) evidence formats.

## Step 1A: Clinical Preprocessing & Evidence Decoder Pipeline
Built the `DDXPlusDecoder` in [`src/decoder.py`](file:///d:/sathwik/Project/Agentic-diagnostic-assistant/src/decoder.py):
- **Token Decoding**: Converts codes like `E_48`, `E_54_@_V_161`, and `E_56_@_4` into readable English concepts and values.
- **Clinical Categorization**: Automatically separates acute presenting symptoms from medical history and risk factors (antecedents).
- **Vignette Generator**: Produces formatted physician-grade clinical notes (`to_clinical_vignette()`).
- **Batch Processing**: Produces `ddxplus_train_10k_decoded.csv` for downstream diagnostic modeling and the Multi-turn Intake Agent.

## Step 2: Diagnostic Engine Baseline (Differential Diagnosis)
Built the `DiagnosticEngine` in [`src/diagnostic_engine.py`](file:///d:/sathwik/Project/Agentic-diagnostic-assistant/src/diagnostic_engine.py):
- **Feature Vectorizer**: Maps age, sex, and sparse multi-hot evidence tokens into unified vectors.
- **Probabilistic Inference**: Produces probability distributions over all 49 pathologies.
- **Benchmark Accuracy**:
  - **Top-1 Accuracy**: **99.45%**
  - **Top-3 Accuracy**: **100.00%**
  - **Top-5 Accuracy**: **100.00%**
- **Dynamic Uncertainty Collapse**: Verified that initial chief complaints yield an open differential (e.g., ~25% top candidate), converging toward >99% confidence as the patient interview progresses.
- **Model Artifact**: Serialized to [`models/diagnostic_model.joblib`](file:///d:/sathwik/Project/Agentic-diagnostic-assistant/models/diagnostic_model.joblib).
