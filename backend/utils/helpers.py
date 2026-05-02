"""
============================================================
MedAI — Helper Utilities
============================================================
Shared utility functions used across the codebase:
  - Text normalization
  - Confidence formatting
  - Session ID generation
  - Response serialization
  - Input validation
  - Timing decorators
============================================================
"""

import re
import time
import uuid
import json
import hashlib
import logging
import functools
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# TEXT UTILITIES
# ══════════════════════════════════════════════════════════

def normalize_symptom(symptom: str) -> str:
    """
    Normalize a symptom string to internal format.
    Examples:
        "High Fever"    → "high_fever"
        "Sore Throat"   → "sore_throat"
        "joint-pain"    → "joint_pain"
        "  Headache  "  → "headache"
    """
    return (
        symptom
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("'", "")
        .replace(".", "")
    )


def normalize_symptom_list(symptoms: List[str]) -> List[str]:
    """Normalize a list of symptom strings."""
    seen = set()
    result = []
    for s in symptoms:
        normalized = normalize_symptom(s)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def symptom_to_display(symptom: str) -> str:
    """
    Convert internal symptom format to human-readable display.
    Example: "high_fever" → "High Fever"
    """
    return symptom.replace("_", " ").title()


def symptoms_to_display_list(symptoms: List[str]) -> List[str]:
    """Convert a list of internal symptom names to display names."""
    return [symptom_to_display(s) for s in symptoms]


def truncate_text(text: str, max_chars: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_chars, adding suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def clean_llm_json(raw: str) -> str:
    """
    Strip markdown code fences from LLM JSON responses.
    Handles: ```json ... ```, ``` ... ```, and plain JSON.
    """
    raw = raw.strip()
    # Remove ```json ... ``` or ``` ... ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# ══════════════════════════════════════════════════════════
# CONFIDENCE & SEVERITY HELPERS
# ══════════════════════════════════════════════════════════

def confidence_to_label(confidence_pct: float) -> str:
    """
    Convert numeric confidence percentage to human-readable label.
    Args:
        confidence_pct: 0–100 percentage
    Returns:
        "High" | "Moderate" | "Low" | "Very Low"
    """
    if confidence_pct >= 70:
        return "High"
    elif confidence_pct >= 40:
        return "Moderate"
    elif confidence_pct >= 20:
        return "Low"
    else:
        return "Very Low"


def severity_to_color(severity: str) -> str:
    """Map severity level to a UI color string."""
    mapping = {
        "emergency": "#ef4444",   # red
        "severe":    "#f59e0b",   # amber
        "moderate":  "#3b82f6",   # blue
        "chronic":   "#8b5cf6",   # purple
        "mild":      "#10b981",   # green
    }
    return mapping.get(severity.lower(), "#94a3b8")


def urgency_is_emergency(urgency: str) -> bool:
    """Return True if the urgency string indicates an emergency."""
    emergency_keywords = ["emergency", "immediately", "ambulance", "urgent", "911", "112"]
    return any(kw in urgency.lower() for kw in emergency_keywords)


# ══════════════════════════════════════════════════════════
# SESSION MANAGEMENT
# ══════════════════════════════════════════════════════════

def generate_session_id(prefix: str = "session") -> str:
    """Generate a unique session ID."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def hash_text(text: str) -> str:
    """Return a short hash of a text string (for caching keys)."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


# ══════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════

def validate_symptom_text(text: str) -> Dict[str, Any]:
    """
    Validate free-text symptom input.
    Returns dict with 'valid' bool and 'error' message if invalid.
    """
    if not text or not text.strip():
        return {"valid": False, "error": "Input text is empty."}

    if len(text.strip()) < 5:
        return {"valid": False, "error": "Input too short. Please describe your symptoms in more detail."}

    if len(text) > 3000:
        return {"valid": False, "error": "Input too long. Please keep description under 3000 characters."}

    # Warn if text looks like it's not symptoms
    non_medical = re.compile(r"^\d+$|^https?://|<script", re.I)
    if non_medical.match(text.strip()):
        return {"valid": False, "error": "Input does not appear to be a symptom description."}

    return {"valid": True, "error": None}


def validate_symptoms_list(symptoms: List[str]) -> Dict[str, Any]:
    """Validate a list of symptom strings."""
    if not symptoms:
        return {"valid": False, "error": "No symptoms provided."}
    if len(symptoms) > 50:
        return {"valid": False, "error": "Too many symptoms. Please provide at most 50."}
    return {"valid": True, "error": None}


# ══════════════════════════════════════════════════════════
# SERIALIZATION
# ══════════════════════════════════════════════════════════

def safe_json_serialize(obj: Any) -> Any:
    """
    Recursively convert numpy / dataclass objects to JSON-serializable types.
    Handles: numpy int/float, datetime, dataclasses.
    """
    import numpy as np
    from dataclasses import asdict, is_dataclass

    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(i) for i in obj]
    return obj


def format_prediction_for_display(prediction: Dict) -> Dict:
    """Format a raw ML prediction dict for clean API output."""
    return {
        "rank":             prediction.get("rank", 1),
        "disease":          prediction.get("disease", "Unknown"),
        "confidence":       round(float(prediction.get("confidence", 0)), 2),
        "confidence_level": prediction.get("confidence_level", "Low"),
        "severity":         prediction.get("severity", "unknown"),
        "specialist":       prediction.get("specialist", "General Physician"),
        "urgency":          prediction.get("urgency", "Consult a doctor"),
        "description":      truncate_text(prediction.get("description", ""), 300),
        "matching_symptoms": prediction.get("matching_symptoms", []),
    }


# ══════════════════════════════════════════════════════════
# DECORATORS
# ══════════════════════════════════════════════════════════

def timeit(func: Callable) -> Callable:
    """Decorator that logs the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def timeit_async(func: Callable) -> Callable:
    """Async version of timeit decorator."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def retry(max_attempts: int = 3, delay: float = 1.0, exceptions=(Exception,)):
    """
    Decorator that retries a function on failure.
    Useful for LLM API calls that may have transient errors.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_attempts:
                        logger.warning(f"{func.__name__} attempt {attempt} failed: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts.")
            raise last_exc
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════
# MISC
# ══════════════════════════════════════════════════════════

def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent.parent


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds to human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        return f"{seconds/60:.1f}min"


def build_error_response(message: str, status: str = "error", code: int = 400) -> Dict:
    """Build a standardized error response dict."""
    return {
        "status": status,
        "error": message,
        "code": code,
        "timestamp": datetime.utcnow().isoformat()
    }


def build_success_response(data: Any, message: str = "Success") -> Dict:
    """Build a standardized success response dict."""
    return {
        "status": "success",
        "message": message,
        "data": safe_json_serialize(data),
        "timestamp": datetime.utcnow().isoformat()
    }