"""
============================================================
MedAI — ML Predictor (Inference)
============================================================
Loads trained models and runs predictions.
Used by the FastAPI backend to get disease predictions
from symptom inputs.
============================================================
"""

import json
import joblib
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

MODEL_DIR = Path("backend/ml/saved_models")
DATA_DIR = Path("backend/data")


class MLPredictor:
    """
    Production-ready ML predictor for symptom-to-disease mapping.
    Loads the best trained model and returns structured predictions.
    """

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        self.disease_info: Dict = {}
        self.is_loaded = False
        self._load_all()

    def _load_all(self):
        """Load model, encoder, metadata, and disease info."""
        try:
            # Load model
            model_path = MODEL_DIR / "best_model.pkl"
            self.model = joblib.load(model_path)
            logger.info(f"✅ ML model loaded: {model_path}")

            # Load label encoder
            le_path = MODEL_DIR / "label_encoder.pkl"
            self.label_encoder = joblib.load(le_path)

            # Load metadata
            meta_path = MODEL_DIR / "model_metadata.json"
            with open(meta_path) as f:
                metadata = json.load(f)
            self.feature_names = metadata["feature_names"]
            self.class_names = metadata["class_names"]

            # Load disease info (descriptions, severity, etc.)
            disease_path = DATA_DIR / "disease_info.json"
            with open(disease_path) as f:
                self.disease_info = json.load(f)

            self.is_loaded = True
            logger.info(f"✅ Predictor ready: {len(self.feature_names)} features, {len(self.class_names)} classes")

        except FileNotFoundError as e:
            logger.error(f"❌ Model files not found: {e}")
            logger.error("Run: python -m backend.ml.train_model first")
            raise

    def symptoms_to_vector(self, symptoms: List[str]) -> np.ndarray:
        """
        Convert a list of symptom strings to a binary feature vector.
        Each position corresponds to a symptom in self.feature_names.

        Args:
            symptoms: List of symptom strings (e.g., ["fever", "headache"])

        Returns:
            numpy array of shape (1, n_features) with 1s for present symptoms
        """
        vector = np.zeros(len(self.feature_names), dtype=float)

        matched = []
        unmatched = []

        for symptom in symptoms:
            # Normalize: lowercase, strip, replace spaces with underscores
            normalized = symptom.lower().strip().replace(" ", "_").replace("-", "_")

            if normalized in self.feature_names:
                idx = self.feature_names.index(normalized)
                vector[idx] = 1.0
                matched.append(normalized)
            else:
                # Try partial matching for flexibility
                for feat in self.feature_names:
                    if normalized in feat or feat in normalized:
                        idx = self.feature_names.index(feat)
                        vector[idx] = 1.0
                        matched.append(feat)
                        break
                else:
                    unmatched.append(symptom)

        if unmatched:
            logger.debug(f"Unmatched symptoms: {unmatched}")
        logger.debug(f"Matched {len(matched)} symptoms: {matched}")

        return vector.reshape(1, -1), matched

    def predict(self, symptoms: List[str], top_n: int = 3) -> Dict:
        """
        Predict diseases from symptom list.

        Args:
            symptoms: List of symptom strings
            top_n: Number of top predictions to return

        Returns:
            Dictionary with predictions, confidence scores, and metadata
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call _load_all() first.")

        if not symptoms:
            return {"error": "No symptoms provided", "predictions": []}

        # Convert symptoms to feature vector
        vector, matched_symptoms = self.symptoms_to_vector(symptoms)

        if matched_symptoms == []:
            return {
                "error": "None of the provided symptoms were recognized.",
                "predictions": [],
                "matched_symptoms": [],
                "input_symptoms": symptoms
            }

        # Get class probabilities
        probabilities = self.model.predict_proba(vector)[0]

        # Get top N predictions
        top_indices = np.argsort(probabilities)[::-1][:top_n]

        predictions = []
        for rank, idx in enumerate(top_indices):
            disease_name = self.label_encoder.inverse_transform([idx])[0]
            confidence = float(probabilities[idx])

            # Skip very low confidence predictions
            if confidence < 0.02:
                continue

            # Get disease metadata
            info = self.disease_info.get(disease_name, {})

            prediction = {
                "rank": rank + 1,
                "disease": disease_name,
                "confidence": round(confidence * 100, 2),  # as percentage
                "confidence_level": self._confidence_level(confidence),
                "severity": info.get("severity", "unknown"),
                "specialist": info.get("specialist", "General Physician"),
                "urgency": info.get("urgency", "Consult a doctor"),
                "description": info.get("description", ""),
                "matching_symptoms": self._get_matching_symptoms(matched_symptoms, disease_name),
            }
            predictions.append(prediction)

        return {
            "predictions": predictions,
            "matched_symptoms": matched_symptoms,
            "unmatched_symptoms": [s for s in symptoms if s.lower().strip().replace(" ", "_") not in matched_symptoms],
            "input_symptoms": symptoms,
            "total_symptoms_matched": len(matched_symptoms),
            "model_info": {
                "type": type(self.model).__name__,
                "num_features": len(self.feature_names),
                "num_classes": len(self.class_names)
            }
        }

    def _confidence_level(self, confidence: float) -> str:
        """Convert numeric confidence to human-readable label."""
        if confidence >= 0.70:
            return "High"
        elif confidence >= 0.40:
            return "Moderate"
        elif confidence >= 0.20:
            return "Low"
        else:
            return "Very Low"

    def _get_matching_symptoms(self, provided_symptoms: List[str], disease: str) -> List[str]:
        """Return symptoms that match the predicted disease's known symptoms."""
        info = self.disease_info.get(disease, {})
        disease_symptoms = info.get("all_symptoms", [])
        return [s for s in provided_symptoms if s in disease_symptoms]

    def get_all_symptoms(self) -> List[str]:
        """Return full list of known symptoms."""
        return self.feature_names

    def get_feature_importance(self, top_n: int = 20) -> List[Dict]:
        """Return top N most important features from the model."""
        if not hasattr(self.model, "feature_importances_"):
            return []
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        return [
            {"symptom": self.feature_names[i], "importance": round(float(importances[i]), 6)}
            for i in indices
        ]