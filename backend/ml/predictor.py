"""
============================================================
MedAI — ML Predictor (Production Safe)
============================================================
"""

import json
import joblib
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger(__name__)

# ✅ FIX: absolute paths (WORKS ON RAILWAY)
BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "backend" / "ml" / "saved_models"
DATA_DIR = BASE_DIR / "backend" / "data"


class MLPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        self.disease_info: Dict = {}
        self.is_loaded = False

        self._load_all()

    def _load_all(self):
        try:
            logger.info(f"📂 MODEL_DIR: {MODEL_DIR}")
            logger.info(f"📂 DATA_DIR: {DATA_DIR}")

            # ================= MODEL =================
            model_path = MODEL_DIR / "best_model.pkl"
            self.model = joblib.load(model_path)
            logger.info(f"✅ Loaded model: {model_path}")

            # ================= LABEL ENCODER =================
            le_path = MODEL_DIR / "label_encoder.pkl"
            self.label_encoder = joblib.load(le_path)

            # ================= METADATA =================
            meta_path = MODEL_DIR / "model_metadata.json"
            with open(meta_path) as f:
                metadata = json.load(f)

            self.feature_names = metadata["feature_names"]
            self.class_names = metadata["class_names"]

            # ================= DISEASE INFO =================
            disease_path = DATA_DIR / "disease_info.json"
            with open(disease_path) as f:
                self.disease_info = json.load(f)

            self.is_loaded = True

            logger.success(
                f"🚀 Predictor ready | Features: {len(self.feature_names)} | Classes: {len(self.class_names)}"
            )

        except Exception as e:
            logger.error(f"❌ ML LOAD FAILED: {e}")
            raise

    # ================= VECTOR =================
    def symptoms_to_vector(self, symptoms: List[str]):
        vector = np.zeros(len(self.feature_names), dtype=float)

        matched = []
        unmatched = []

        for symptom in symptoms:
            normalized = symptom.lower().strip().replace(" ", "_").replace("-", "_")

            if normalized in self.feature_names:
                idx = self.feature_names.index(normalized)
                vector[idx] = 1.0
                matched.append(normalized)
            else:
                for feat in self.feature_names:
                    if normalized in feat or feat in normalized:
                        idx = self.feature_names.index(feat)
                        vector[idx] = 1.0
                        matched.append(feat)
                        break
                else:
                    unmatched.append(symptom)

        return vector.reshape(1, -1), matched

    # ================= PREDICT =================
    def predict(self, symptoms: List[str], top_n: int = 3):
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        if not symptoms:
            return {"error": "No symptoms provided", "predictions": []}

        vector, matched = self.symptoms_to_vector(symptoms)

        if not matched:
            return {
                "error": "No recognized symptoms",
                "predictions": []
            }

        probs = self.model.predict_proba(vector)[0]
        top_indices = np.argsort(probs)[::-1][:top_n]

        predictions = []

        for i, idx in enumerate(top_indices):
            confidence = float(probs[idx])

            if confidence < 0.02:
                continue

            disease = self.label_encoder.inverse_transform([idx])[0]
            info = self.disease_info.get(disease, {})

            predictions.append({
                "rank": i + 1,
                "disease": disease,
                "confidence": round(confidence * 100, 2),
                "severity": info.get("severity", "unknown"),
                "specialist": info.get("specialist", "General Physician"),
                "description": info.get("description", "")
            })

        return {
            "predictions": predictions,
            "matched_symptoms": matched
        }

    def get_all_symptoms(self):
        return self.feature_names