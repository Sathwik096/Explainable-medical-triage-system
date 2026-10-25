"""
Quick inference test script for DiagnosticEngine
"""

from src.diagnostic_engine import DiagnosticEngine

def main():
    # 1. Load the pre-trained engine
    engine = DiagnosticEngine(model_path="models/diagnostic_model.joblib")

    # 2. Test patient profile: 45-year-old male with Cough (E_201) and Fever (E_91)
    evidences = ["E_201", "E_91"]
    age = 45
    sex = "M"

    diff = engine.get_differential(evidences, age=age, sex=sex, top_k=5)

    print("\n=== Differential Diagnosis for [Cough (E_201) + Fever (E_91)] ===")
    for idx, c in enumerate(diff, 1):
        print(f"{idx}. {c['condition']:<30} | Prob: {c['probability_percent']:>5.1f}% | ICD-10: {c['icd10']:<8} | Severity: {c['severity']}/5")
    print("=" * 72 + "\n")

if __name__ == "__main__":
    main()
