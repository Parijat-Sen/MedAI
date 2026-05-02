"""
============================================================
MedAI — FastAPI Backend (Main Application)
============================================================
Complete REST API with:
  /api/analyze      — Full symptom analysis (ML + RAG + LLM)
  /api/predict      — ML-only prediction (fast)
  /api/chat         — Follow-up conversation
  /api/symptoms     — Get all known symptoms
  /api/health       — Health check
  /api/metrics      — Model evaluation metrics
  /api/explain      — Feature importance explanation
============================================================
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field, validator

from loguru import logger

# ── Load environment variables ────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Internal modules ──────────────────────────────────────
import sys
sys.path.insert(0, ".")

from backend.ml.predictor import MLPredictor
from backend.rag.rag_pipeline import RAGPipeline
from backend.rag.llm_service import LLMService
from backend.utils.symptom_extractor import SymptomExtractor


# ══════════════════════════════════════════════════════════
# APP STARTUP — Load models once at boot
# ══════════════════════════════════════════════════════════

# Global service instances (initialized on startup)
ml_predictor: Optional[MLPredictor] = None
rag_pipeline: Optional[RAGPipeline] = None
llm_service: Optional[LLMService] = None
symptom_extractor: Optional[SymptomExtractor] = None

# In-memory chat sessions {session_id: [{role, content}]}
chat_sessions: Dict[str, List[Dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup."""
    global ml_predictor, rag_pipeline, llm_service, symptom_extractor

    logger.info("🚀 MedAI starting up...")

    # ── ML Model ──────────────────────────────────────────
    logger.info("Loading ML model...")
    try:
        ml_predictor = MLPredictor()
        logger.success("✅ ML model loaded")
    except Exception as e:
        logger.error(f"❌ ML model failed: {e}")
        logger.warning("Run setup.py first to train the model.")

    # ── RAG Pipeline ──────────────────────────────────────
    logger.info("Initializing RAG pipeline...")
    try:
        rag_pipeline = RAGPipeline()
        rag_pipeline.initialize()
        logger.success("✅ RAG pipeline ready")
    except Exception as e:
        logger.error(f"❌ RAG pipeline failed: {e}")
        logger.warning("Run setup.py first to build the FAISS index.")

    # ── LLM Service ───────────────────────────────────────
    logger.info("Initializing LLM service...")
    llm_service = LLMService()
    logger.success(f"✅ LLM service ready: {llm_service.manager.provider}")

    # ── Symptom Extractor ─────────────────────────────────
    logger.info("Loading symptom extractor...")
    known_symptoms = ml_predictor.get_all_symptoms() if ml_predictor else []
    symptom_extractor = SymptomExtractor(known_symptoms)
    logger.success("✅ Symptom extractor ready")

    logger.success("🏥 MedAI is ready to serve!")
    yield

    # Cleanup on shutdown
    logger.info("MedAI shutting down...")


# ══════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════

app = FastAPI(
    title="MedAI Clinical Decision Support API",
    description=(
        "Symptom-Based Disease Prediction using ML + LLM + RAG. "
        "For educational and research purposes only."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# CORS — allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production: restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
static_dir = Path("frontend/static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ══════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════

class AnalyzeRequest(BaseModel):
    """Request schema for full analysis."""
    text: str = Field(..., min_length=3, max_length=2000,
                      example="I have high fever, severe headache, and body aches since 3 days")
    session_id: Optional[str] = Field(None, example="user_123")
    top_n: int = Field(3, ge=1, le=5)

    @validator("text")
    def validate_text(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Text too short. Please describe your symptoms.")
        return v.strip()


class PredictRequest(BaseModel):
    """Request schema for ML-only prediction."""
    symptoms: List[str] = Field(..., min_items=1,
                                example=["fever", "headache", "chills"])
    top_n: int = Field(3, ge=1, le=5)


class ChatRequest(BaseModel):
    """Request schema for follow-up chat."""
    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(..., example="user_123")


class SymptomsSearchRequest(BaseModel):
    """Request schema for symptom autocomplete."""
    query: str = Field(..., min_length=2)


# ══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════

def require_ml():
    if ml_predictor is None:
        raise HTTPException(503, "ML model not loaded. Run setup.py first.")

def require_rag():
    if rag_pipeline is None:
        raise HTTPException(503, "RAG pipeline not loaded. Run setup.py first.")


# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = Path("frontend/templates/index.html")
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse("<h1>MedAI API is running. Visit /api/docs for documentation.</h1>")


@app.get("/api/health")
async def health_check():
    """System health check — returns status of all components."""
    return {
        "status": "healthy",
        "app": "MedAI Clinical Decision Support",
        "version": "1.0.0",
        "components": {
            "ml_model": ml_predictor is not None,
            "rag_pipeline": rag_pipeline is not None,
            "llm_service": llm_service is not None,
            "llm_provider": llm_service.manager.provider if llm_service else "unavailable",
        }
    }


@app.post("/api/analyze")
async def analyze_symptoms(request: AnalyzeRequest):
    """
    MAIN ENDPOINT — Full symptom analysis pipeline.

    Workflow:
      1. Extract symptoms from free text
      2. ML prediction (Random Forest / XGBoost)
      3. RAG retrieval (FAISS vector search)
      4. LLM explanation generation
      5. Return structured response

    Returns combined ML + RAG + LLM result.
    """
    require_ml()
    start_time = time.time()

    # 1. Extract symptoms from free text
    extracted_symptoms, unrecognized = symptom_extractor.extract(request.text)
    logger.info(f"Extracted {len(extracted_symptoms)} symptoms from: '{request.text[:80]}...'")

    if not extracted_symptoms:
        raise HTTPException(
            400,
            detail=(
                "Could not extract recognized symptoms from your input. "
                "Please describe symptoms clearly (e.g., 'fever', 'headache', 'cough'). "
                f"Unrecognized terms: {unrecognized}"
            )
        )

    # 2. ML prediction
    ml_result = ml_predictor.predict(extracted_symptoms, top_n=request.top_n)
    predictions = ml_result.get("predictions", [])

    if not predictions:
        raise HTTPException(400, "Could not generate predictions. Try adding more symptoms.")

    # 3. RAG retrieval
    rag_context = "Medical context not available."
    if rag_pipeline:
        disease_names = [p["disease"] for p in predictions]
        try:
            rag_context = rag_pipeline.retrieve_for_symptoms(
                symptoms=extracted_symptoms,
                diseases=disease_names,
                top_k=5
            )
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")

    # 4. LLM explanation
    chat_history = chat_sessions.get(request.session_id, []) if request.session_id else []

    llm_response = llm_service.analyze(
        symptoms=extracted_symptoms,
        ml_predictions=predictions,
        rag_context=rag_context,
        chat_history=chat_history
    )

    # 5. Update chat session
    if request.session_id:
        if request.session_id not in chat_sessions:
            chat_sessions[request.session_id] = []
        chat_sessions[request.session_id].append({"role": "user", "content": request.text})
        chat_sessions[request.session_id].append({
            "role": "assistant",
            "content": llm_response.explanation[:500]  # Store summary only
        })

    elapsed = round(time.time() - start_time, 2)

    # Build final response
    return {
        "status": "success",
        "elapsed_seconds": elapsed,
        "input": {
            "raw_text": request.text,
            "extracted_symptoms": extracted_symptoms,
            "unrecognized_terms": unrecognized,
            "symptom_display": symptom_extractor.format_for_display(extracted_symptoms)
        },
        "ml_predictions": predictions,
        "llm_analysis": {
            "explanation": llm_response.explanation,
            "follow_up_questions": llm_response.follow_up_questions,
            "recommended_actions": llm_response.recommended_actions,
            "red_flags": llm_response.red_flags,
            "confidence_note": llm_response.confidence_note,
            "disclaimer": llm_response.disclaimer,
            "model_used": llm_response.model_used
        },
        "rag_context_used": len(rag_context) > 50,
        "session_id": request.session_id
    }


@app.post("/api/predict")
async def predict_only(request: PredictRequest):
    """
    ML-only prediction (fast endpoint — no LLM call).
    Useful for quick lookups or when LLM is not needed.
    """
    require_ml()

    result = ml_predictor.predict(request.symptoms, top_n=request.top_n)

    return {
        "status": "success",
        "predictions": result.get("predictions", []),
        "matched_symptoms": result.get("matched_symptoms", []),
        "model_info": result.get("model_info", {}),
        "note": "ML-only prediction. Use /api/analyze for full LLM explanation."
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Follow-up chat endpoint.
    Continues conversation started by /api/analyze.
    """
    session = chat_sessions.get(request.session_id, [])
    if not session:
        raise HTTPException(
            400,
            "No active session found. Please run /api/analyze first to start a session."
        )

    # Build context from session history
    context = "\n".join(
        f"{'Patient' if m['role'] == 'user' else 'MedAI'}: {m['content']}"
        for m in session[-6:]  # Last 3 exchanges
    )

    llm_response = llm_service.chat(request.message, context)

    # Update session
    session.append({"role": "user", "content": request.message})
    session.append({"role": "assistant", "content": llm_response.explanation[:500]})
    chat_sessions[request.session_id] = session

    return {
        "status": "success",
        "response": {
            "explanation": llm_response.explanation,
            "follow_up_questions": llm_response.follow_up_questions,
            "recommended_actions": llm_response.recommended_actions,
            "red_flags": llm_response.red_flags,
            "disclaimer": llm_response.disclaimer
        },
        "session_id": request.session_id
    }


@app.get("/api/symptoms")
async def get_symptoms(query: Optional[str] = None):
    """
    Return list of all known symptoms.
    Optionally filter with ?query=partial_name for autocomplete.
    """
    require_ml()

    all_symptoms = ml_predictor.get_all_symptoms()

    if query:
        filtered = [s for s in all_symptoms if query.lower() in s.lower()]
        return {"symptoms": filtered[:20], "total": len(filtered)}

    return {"symptoms": all_symptoms, "total": len(all_symptoms)}


@app.get("/api/metrics")
async def get_model_metrics():
    """Return model evaluation metrics from training."""
    metadata_path = Path("backend/ml/saved_models/model_metadata.json")
    if not metadata_path.exists():
        raise HTTPException(404, "Model metadata not found. Train the model first.")

    with open(metadata_path) as f:
        metadata = json.load(f)

    return {
        "status": "success",
        "best_model": metadata.get("best_model"),
        "metrics": metadata.get("metrics", {}),
        "top_features": metadata.get("top_features", [])[:15],
        "training_info": {
            "num_features": metadata.get("num_features"),
            "num_classes": metadata.get("num_classes"),
            "training_samples": metadata.get("training_samples"),
            "class_names": metadata.get("class_names")
        }
    }


@app.get("/api/explain/{disease}")
async def explain_disease(disease: str):
    """Return detailed information about a specific disease."""
    disease_path = Path("backend/data/disease_info.json")
    if not disease_path.exists():
        raise HTTPException(404, "Disease info not found.")

    with open(disease_path) as f:
        disease_info = json.load(f)

    # Case-insensitive search
    for name, info in disease_info.items():
        if disease.lower() in name.lower() or name.lower() in disease.lower():
            return {"disease": name, "info": info}

    raise HTTPException(404, f"Disease '{disease}' not found in database.")


@app.get("/api/compare")
async def compare_ml_llm():
    """
    Returns a comparison of ML-only vs ML+LLM predictions.
    Uses example cases to demonstrate the difference.
    """
    require_ml()

    examples = [
        {
            "symptoms": ["fever", "chills", "muscle_aches", "headache"],
            "description": "Classic flu-like presentation"
        },
        {
            "symptoms": ["chest_pain_pressure", "shortness_of_breath", "sweating", "pain_radiating_to_arm_jaw"],
            "description": "Cardiac emergency symptoms"
        },
        {
            "symptoms": ["frequent_urination", "excessive_thirst", "fatigue", "blurred_vision"],
            "description": "Metabolic symptoms"
        }
    ]

    results = []
    for example in examples:
        ml_result = ml_predictor.predict(example["symptoms"], top_n=3)
        results.append({
            "description": example["description"],
            "symptoms": example["symptoms"],
            "ml_predictions": ml_result.get("predictions", []),
            "note": "Add LLM API key to see full natural language explanations"
        })

    return {
        "comparison": results,
        "ml_model": type(ml_predictor.model).__name__,
        "message": "ML predictions shown. Full /api/analyze includes RAG + LLM enhancement."
    }