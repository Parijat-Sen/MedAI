"""
MedAI — FastAPI Backend (FINAL PRODUCTION READY)
"""

from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from typing import List, Optional, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator

from loguru import logger
from dotenv import load_dotenv

import os
import sys
import uuid

# =========================
# LOAD ENV
# =========================
load_dotenv()
sys.path.insert(0, ".")

# =========================
# IMPORTS
# =========================
from backend.ml.predictor import MLPredictor
from backend.rag.rag_pipeline import RAGPipeline
from backend.rag.llm_service import LLMService
from backend.utils.symptom_extractor import SymptomExtractor


# =========================
# GLOBAL SERVICES
# =========================
ml_predictor = None
rag_pipeline = None
llm_service = None
symptom_extractor = None

# ✅ chat memory
chat_sessions: Dict[str, List[Dict]] = {}


# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
TEMPLATES_DIR = FRONTEND_DIR / "templates"

logger.info(f"BASE_DIR: {BASE_DIR}")
logger.info(f"STATIC_DIR exists: {STATIC_DIR.exists()}")
logger.info(f"TEMPLATES_DIR exists: {TEMPLATES_DIR.exists()}")


# =========================
# SCHEMAS
# =========================
class AnalyzeRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    top_n: int = 3

    @validator("text")
    def validate_text(cls, v):
        if len(v.strip()) < 3:
            raise ValueError("Text too short")
        return v.strip()


class ChatRequest(BaseModel):
    message: str
    session_id: str


# =========================
# STARTUP
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_predictor, rag_pipeline, llm_service, symptom_extractor

    logger.info("🚀 Starting MedAI...")

    try:
        ml_predictor = MLPredictor()
        logger.success("✅ ML loaded")
    except Exception as e:
        logger.error(f"ML error: {e}")

    try:
        rag_pipeline = RAGPipeline()
        rag_pipeline.initialize()
        logger.success("✅ RAG ready")
    except Exception as e:
        logger.error(f"RAG error: {e}")

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("⚠️ OPENAI_API_KEY not found")

    try:
        llm_service = LLMService()
        logger.success(f"✅ LLM ready: {llm_service.manager.provider}")
    except Exception as e:
        logger.error(f"LLM error: {e}")

    symptoms = ml_predictor.get_all_symptoms() if ml_predictor else []
    symptom_extractor = SymptomExtractor(symptoms)

    yield
    logger.info("🛑 Shutting down...")


# =========================
# APP
# =========================
app = FastAPI(
    title="MedAI API",
    version="1.0",
    docs_url="/api/docs",
    lifespan=lifespan
)

# =========================
# STATIC
# =========================
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
else:
    logger.error(f"❌ Static folder missing: {STATIC_DIR}")


# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# GLOBAL ERROR
# =========================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal Server Error",
            "detail": str(exc)
        }
    )


# =========================
# ROOT → UI
# =========================
@app.get("/")
async def serve_ui():
    index_file = TEMPLATES_DIR / "index.html"

    if index_file.exists():
        return FileResponse(index_file)

    return JSONResponse(
        status_code=500,
        content={"error": "index.html not found"}
    )


# =========================
# HEALTH
# =========================
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "ml": ml_predictor is not None,
        "rag": rag_pipeline is not None,
        "llm": llm_service is not None,
        "provider": llm_service.manager.provider if llm_service else "none"
    }


# =========================
# ANALYZE
# =========================
@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    try:
        if not ml_predictor:
            raise Exception("ML not loaded")

        symptoms, unrecognized = symptom_extractor.extract(request.text)

        if not symptoms:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "No valid symptoms",
                    "unrecognized": unrecognized
                }
            )

        ml_result = ml_predictor.predict(symptoms)
        predictions = ml_result.get("predictions", [])

        rag_context = ""
        if rag_pipeline:
            try:
                rag_context = rag_pipeline.retrieve_for_symptoms(
                    symptoms,
                    [p["disease"] for p in predictions]
                )
            except Exception as e:
                logger.warning(f"RAG fail: {e}")

        llm_res = llm_service.analyze(symptoms, predictions, rag_context)

        # ✅ create session
        session_id = str(uuid.uuid4())
        chat_sessions[session_id] = []

        return {
            "status": "success",
            "session_id": session_id,
            "symptoms": symptoms,
            "predictions": predictions,
            "llm": {
                "explanation": llm_res.explanation,
                "follow_up_questions": llm_res.follow_up_questions,
                "recommended_actions": llm_res.recommended_actions,
                "red_flags": llm_res.red_flags,
                "disclaimer": llm_res.disclaimer
            }
        }

    except Exception as e:
        logger.error(f"Analyze error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )


# =========================
# CHAT (FIXED)
# =========================
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # ✅ always create session if missing
        if request.session_id not in chat_sessions:
            chat_sessions[request.session_id] = []

        session = chat_sessions[request.session_id]

        context = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in session[-6:]
        )

        res = llm_service.chat(request.message, context)

        response_text = getattr(res, "explanation", None) or str(res)

        session.append({"role": "user", "content": request.message})
        session.append({"role": "assistant", "content": response_text})

        return {
            "status": "success",
            "response": response_text
        }

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )