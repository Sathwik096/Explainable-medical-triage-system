"""
Diagnostic Engine Baseline & Differential Diagnosis Engine
==========================================================
Provides real-time multi-class diagnostic inference and differential diagnosis
ranking across the 49 DDXPlus pathologies.

Capable of evaluating:
- Partial evidence (initial chief complaint only)
- Dynamic incremental evidence as a multi-turn intake agent asks questions
- Complete evidence profiles
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.decoder import DDXPlusDecoder


class DiagnosticEngine:
    """
    Probabilistic diagnostic inference engine for DDXPlus conditions.
    """

    def __init__(
        self,
        decoder: Optional[DDXPlusDecoder] = None,
        model_path: Optional[Union[str, Path]] = None,
    ):
        self.decoder = decoder or DDXPlusDecoder()
        self.classifier: Optional[LogisticRegression] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.vocab: List[str] = []
        self.token_to_idx: Dict[str, int] = {}
        self.is_trained: bool = False

        if model_path and Path(model_path).exists():
            self.load(model_path)

    @property
    def feature_dim(self) -> int:
        """Total number of input features: Age + Sex + len(vocab)."""
        return len(self.vocab) + 2

    def build_feature_vector(
        self,
        evidences: Union[List[str], str],
        age: Union[int, float],
        sex: str,
    ) -> np.ndarray:
        """
        Converts patient age, sex, and evidence list into a normalized 1D feature vector.
        """
        if isinstance(evidences, str):
            try:
                ev_list = ast.literal_eval(evidences)
            except Exception:
                ev_list = [evidences]
        else:
            ev_list = evidences or []

        vec = np.zeros(self.feature_dim, dtype=np.float32)
        # Normalized age [0, 1]
        vec[0] = float(age) / 100.0
        # Biological sex (M = 1.0, F/other = 0.0)
        vec[1] = 1.0 if str(sex).upper() == "M" else 0.0

        for token in ev_list:
            idx = self.token_to_idx.get(str(token))
            if idx is not None:
                vec[2 + idx] = 1.0

        return vec

    def fit(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        c_param: float = 1.0,
        max_iter: int = 500,
    ) -> Dict[str, float]:
        """
        Fits the feature encoder and multi-class classifier on patient records.
        Returns validation metrics (top-1, top-3, top-5 accuracy).
        """
        # Parse evidences
        evidences_series = df["EVIDENCES"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else (x or [])
        )

        # Build vocabulary across all observed tokens
        all_tokens = set()
        for ev_list in evidences_series:
            all_tokens.update(ev_list)

        self.vocab = sorted(list(all_tokens))
        self.token_to_idx = {token: i for i, token in enumerate(self.vocab)}

        # Build feature matrix X
        n_samples = len(df)
        X = np.zeros((n_samples, self.feature_dim), dtype=np.float32)

        for i, (_, row) in enumerate(df.iterrows()):
            X[i] = self.build_feature_vector(
                evidences=evidences_series.iloc[i],
                age=row["AGE"],
                sex=row["SEX"],
            )

        # Encode target pathologies
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df["PATHOLOGY"])

        # Train / validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train multi-class Logistic Regression with softmax
        self.classifier = LogisticRegression(
            C=c_param,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=random_state,
        )
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

        # Compute evaluation metrics
        val_probs = self.classifier.predict_proba(X_val)
        val_preds = self.classifier.predict(X_val)

        metrics = {
            "top1_accuracy": float(accuracy_score(y_val, val_preds)),
            "top3_accuracy": float(top_k_accuracy_score(y_val, val_probs, k=3)),
            "top5_accuracy": float(top_k_accuracy_score(y_val, val_probs, k=5)),
            "n_classes": len(self.label_encoder.classes_),
            "vocab_size": len(self.vocab),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }
        return metrics

    def predict_proba(
        self,
        evidences: Union[List[str], str],
        age: Union[int, float],
        sex: str,
    ) -> Dict[str, float]:
        """
        Returns a dictionary mapping every pathology to its predicted probability.
        """
        if not self.is_trained or self.classifier is None or self.label_encoder is None:
            raise RuntimeError("DiagnosticEngine must be fitted or loaded before prediction.")

        vec = self.build_feature_vector(evidences, age, sex).reshape(1, -1)
        probs = self.classifier.predict_proba(vec)[0]

        return {
            str(cls_name): float(probs[i])
            for i, cls_name in enumerate(self.label_encoder.classes_)
        }

    def get_differential(
        self,
        evidences: Union[List[str], str],
        age: Union[int, float],
        sex: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Returns the top-K candidate diseases ranked by probability, enriched with
        ICD-10 codes, clinical severity, and probability percentages.
        """
        proba_map = self.predict_proba(evidences, age, sex)
        sorted_candidates = sorted(proba_map.items(), key=lambda x: x[1], reverse=True)[:top_k]

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

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Saves the trained model, vocabulary, and label encoder to a joblib bundle.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained DiagnosticEngine.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "classifier": self.classifier,
            "label_encoder": self.label_encoder,
            "vocab": self.vocab,
            "token_to_idx": self.token_to_idx,
        }
        joblib.dump(bundle, filepath)

    def load(self, filepath: Union[str, Path]) -> None:
        """
        Loads a pre-trained model bundle.
        """
        bundle = joblib.load(filepath)
        self.classifier = bundle["classifier"]
        self.label_encoder = bundle["label_encoder"]
        self.vocab = bundle["vocab"]
        self.token_to_idx = bundle["token_to_idx"]
        self.is_trained = True
