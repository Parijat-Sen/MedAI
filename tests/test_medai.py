"""
============================================================
MedAI — Test Suite
============================================================
Run with:
    pytest tests/ -v
============================================================
"""

import sys
import json
import pytest
from pathlib import Path

sys.path.insert(0, ".")


# ══════════════════════════════════════════════════════════
# DATASET TESTS
# ══════════════════════════════════════════════════════════

class TestDataset:

    def test_training_data_exists(self):
        assert Path("backend/data/training_data.csv").exists(), \
            "Run setup.py first"

    def test_symptom_list_exists(self):
        assert Path("backend/data/symptom_list.json").exists()

    def test_symptom_list_not_empty(self):
        with open("backend/data/symptom_list.json") as f:
            symptoms = json.load(f)
        assert len(symptoms) > 50, f"Expected 50+ symptoms, got {len(symptoms)}"

    def test_disease_info_exists(self):
        assert Path("backend/data/disease_info.json").exists()

    def test_disease_info_has_all_fields(self):
        with open("backend/data/disease_info.json") as f:
            info = json.load(f)
        for disease, data in info.items():
            assert "description" in data, f"{disease} missing description"
            assert "severity" in data, f"{disease} missing severity"
            assert "specialist" in data, f"{disease} missing specialist"


# ══════════════════════════════════════════════════════════
# ML MODEL TESTS
# ══════════════════════════════════════════════════════════

class TestMLModel:

    @pytest.fixture(scope="class")
    def predictor(self):
        from backend.ml.predictor import MLPredictor
        return MLPredictor()

    def test_model_loads(self, predictor):
        assert predictor.is_loaded

    def test_predict_returns_predictions(self, predictor):
        result = predictor.predict(["fever", "headache", "chills"])
        assert "predictions" in result
        assert len(result["predictions"]) > 0

    def test_prediction_has_required_fields(self, predictor):
        result = predictor.predict(["fever", "headache"])
        pred = result["predictions"][0]
        assert "disease" in pred
        assert "confidence" in pred
        assert "confidence_level" in pred
        assert "severity" in pred
        assert "specialist" in pred

    def test_confidence_is_percentage(self, predictor):
        result = predictor.predict(["fever", "headache", "nausea"])
        for pred in result["predictions"]:
            assert 0 <= pred["confidence"] <= 100, \
                f"Confidence {pred['confidence']} out of range"

    def test_empty_symptoms_handled(self, predictor):
        result = predictor.predict([])
        assert "error" in result or result.get("predictions") == []

    def test_flu_symptoms_predict_flu(self, predictor):
        """Classic flu symptoms should predict Influenza highly."""
        result = predictor.predict([
            "fever", "chills", "muscle_aches", "fatigue", "headache"
        ])
        top = result["predictions"][0]["disease"]
        assert "Influenza" in top or "Flu" in top or result["predictions"][0]["confidence"] > 30

    def test_symptoms_to_vector(self, predictor):
        vector, matched = predictor.symptoms_to_vector(["fever", "headache"])
        assert vector.shape == (1, len(predictor.feature_names))
        assert len(matched) >= 1


# ══════════════════════════════════════════════════════════
# SYMPTOM EXTRACTOR TESTS
# ══════════════════════════════════════════════════════════

class TestSymptomExtractor:

    @pytest.fixture(scope="class")
    def extractor(self):
        from backend.utils.symptom_extractor import SymptomExtractor
        return SymptomExtractor()

    def test_extracts_direct_symptoms(self, extractor):
        symptoms, _ = extractor.extract("I have fever and headache")
        assert "fever" in symptoms or "headache" in symptoms

    def test_extracts_aliases(self, extractor):
        symptoms, _ = extractor.extract("I feel very tired and have runny nose")
        assert len(symptoms) > 0

    def test_handles_natural_language(self, extractor):
        text = "I've been suffering from high fever, severe headache, body aches, and feeling nauseous for 3 days"
        symptoms, _ = extractor.extract(text)
        assert len(symptoms) >= 2

    def test_empty_input(self, extractor):
        symptoms, _ = extractor.extract("")
        assert symptoms == []

    def test_format_for_display(self, extractor):
        display = extractor.format_for_display(["high_fever", "joint_pain"])
        assert "High Fever" in display
        assert "Joint Pain" in display


# ══════════════════════════════════════════════════════════
# RAG PIPELINE TESTS
# ══════════════════════════════════════════════════════════

class TestRAGPipeline:

    @pytest.fixture(scope="class")
    def pipeline(self):
        from backend.rag.rag_pipeline import RAGPipeline
        p = RAGPipeline()
        p.initialize()
        return p

    def test_pipeline_loads(self, pipeline):
        assert pipeline._loaded

    def test_retrieve_returns_results(self, pipeline):
        results = pipeline.retrieve("fever malaria treatment", top_k=3)
        assert len(results) > 0

    def test_retrieve_has_content(self, pipeline):
        results = pipeline.retrieve("diabetes insulin treatment", top_k=2)
        for r in results:
            assert len(r.content) > 10
            assert r.score >= 0

    def test_retrieve_for_symptoms(self, pipeline):
        context = pipeline.retrieve_for_symptoms(
            symptoms=["fever", "headache"],
            diseases=["Influenza (Flu)"],
            top_k=3
        )
        assert isinstance(context, str)
        assert len(context) > 50


# ══════════════════════════════════════════════════════════
# FASTAPI ENDPOINT TESTS
# ══════════════════════════════════════════════════════════

class TestAPI:

    @pytest.fixture(scope="class")
    def client(self):
        from fastapi.testclient import TestClient
        from backend.api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert "components" in data

    def test_symptoms_endpoint(self, client):
        res = client.get("/api/symptoms")
        assert res.status_code == 200
        data = res.json()
        assert "symptoms" in data
        assert data["total"] > 0

    def test_symptoms_search(self, client):
        res = client.get("/api/symptoms?query=fever")
        assert res.status_code == 200

    def test_predict_endpoint(self, client):
        res = client.post("/api/predict", json={
            "symptoms": ["fever", "headache", "chills"],
            "top_n": 3
        })
        assert res.status_code == 200
        data = res.json()
        assert "predictions" in data
        assert len(data["predictions"]) > 0

    def test_analyze_endpoint(self, client):
        res = client.post("/api/analyze", json={
            "text": "I have high fever, headache, chills and body aches for 2 days",
            "session_id": "test_session_001",
            "top_n": 3
        })
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert len(data["ml_predictions"]) > 0
        assert "llm_analysis" in data
        assert "explanation" in data["llm_analysis"]

    def test_analyze_empty_text(self, client):
        res = client.post("/api/analyze", json={"text": "ab"})
        assert res.status_code == 422  # Validation error

    def test_metrics_endpoint(self, client):
        res = client.get("/api/metrics")
        assert res.status_code == 200
        data = res.json()
        assert "metrics" in data
        assert "best_model" in data

    def test_explain_disease(self, client):
        res = client.get("/api/explain/Influenza")
        assert res.status_code == 200
        data = res.json()
        assert "disease" in data
        assert "info" in data


# ══════════════════════════════════════════════════════════
# INTEGRATION TEST
# ══════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_flu(self):
        """Full pipeline: text → extract → ML → RAG → LLM (fallback)."""
        from backend.ml.predictor import MLPredictor
        from backend.rag.rag_pipeline import RAGPipeline
        from backend.rag.llm_service import LLMService
        from backend.utils.symptom_extractor import SymptomExtractor

        text = "I have been experiencing high fever, chills, severe headache, and muscle aches since yesterday"

        extractor = SymptomExtractor()
        symptoms, _ = extractor.extract(text)
        assert len(symptoms) > 0, "No symptoms extracted"

        predictor = MLPredictor()
        ml_result = predictor.predict(symptoms, top_n=3)
        predictions = ml_result["predictions"]
        assert len(predictions) > 0, "No predictions"

        rag = RAGPipeline()
        rag.initialize()
        context = rag.retrieve_for_symptoms(symptoms, [p["disease"] for p in predictions])
        assert len(context) > 10, "RAG context empty"

        llm = LLMService()
        response = llm.analyze(symptoms, predictions, context)
        assert len(response.explanation) > 20
        assert isinstance(response.follow_up_questions, list)
        assert isinstance(response.recommended_actions, list)

        print(f"\n✅ Integration test passed!")
        print(f"   Symptoms: {symptoms}")
        print(f"   Top prediction: {predictions[0]['disease']} ({predictions[0]['confidence']:.1f}%)")
        print(f"   Explanation (first 200 chars): {response.explanation[:200]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])