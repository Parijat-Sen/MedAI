"""
============================================================
MedAI — LLM Integration Module
============================================================
Manages LLM calls with:
  - Advanced prompt engineering
  - ML + RAG context injection
  - Structured JSON output
  - Multi-turn chat support
  - Fallback for no API key (rule-based)
  - Provider support: OpenAI / Anthropic / Local Ollama
============================================================
"""

import os
import json
import logging
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# ── Try importing LLM libraries ───────────────────────────
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


@dataclass
class LLMResponse:
    """Structured response from the LLM layer."""
    explanation: str
    follow_up_questions: List[str]
    recommended_actions: List[str]
    red_flags: List[str]
    confidence_note: str
    disclaimer: str
    raw_text: str
    model_used: str


# ══════════════════════════════════════════════════════════
# PROMPT ENGINEERING
# ══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are MedAI, an advanced clinical decision support assistant developed for medical education and preliminary analysis. You are assisting trained healthcare students and professionals — NOT replacing them.

Your responsibilities:
1. Explain ML-predicted disease(s) in clinical but understandable language
2. Incorporate retrieved medical knowledge to ground your response
3. Highlight symptom-disease correlations
4. Ask intelligent follow-up questions to refine the differential diagnosis
5. Recommend appropriate actions and urgency level
6. Flag any red-flag symptoms that require emergency care
7. ALWAYS remind users this is not a replacement for professional diagnosis

Your tone: Professional, empathetic, evidence-based, and clear.
Your format: Always respond in valid JSON with the structure specified.

CRITICAL SAFETY RULE: You must NEVER provide specific drug dosages, recommend medications, or make definitive diagnoses. You provide preliminary analysis only."""


def build_prediction_prompt(
    symptoms: List[str],
    ml_predictions: List[Dict],
    rag_context: str,
    chat_history: List[Dict] = None
) -> str:
    """
    Build the main prediction prompt combining ML results + RAG context.
    This is the core of the prompt engineering strategy.

    Args:
        symptoms: User-provided symptoms
        ml_predictions: List of ML model predictions with confidence scores
        rag_context: Retrieved medical knowledge from FAISS
        chat_history: Previous conversation turns

    Returns:
        Formatted prompt string
    """
    # Format symptom list
    symptom_list = "\n".join(f"  • {s.replace('_', ' ').title()}" for s in symptoms)

    # Format ML predictions
    pred_lines = []
    for p in ml_predictions[:3]:
        pred_lines.append(
            f"  {p['rank']}. {p['disease']} "
            f"(Confidence: {p['confidence']:.1f}% — {p['confidence_level']}) "
            f"[Severity: {p.get('severity', 'unknown')}]"
        )
    predictions_text = "\n".join(pred_lines)

    # Format chat history (for multi-turn)
    history_text = ""
    if chat_history:
        recent = chat_history[-4:]  # Last 4 exchanges
        history_text = "\n\nPREVIOUS CONVERSATION:\n" + "\n".join(
            f"{'Patient' if m['role'] == 'user' else 'MedAI'}: {m['content']}"
            for m in recent
        )

    prompt = f"""PATIENT SYMPTOM ANALYSIS REQUEST

════════════════════════════════════════
REPORTED SYMPTOMS ({len(symptoms)} total):
{symptom_list}

════════════════════════════════════════
ML MODEL PREDICTIONS (Random Forest + XGBoost Ensemble):
{predictions_text}

════════════════════════════════════════
RETRIEVED MEDICAL KNOWLEDGE (from medical database):
{rag_context}
{history_text}
════════════════════════════════════════

TASK: Based on the symptoms, ML predictions, and medical knowledge above, provide a comprehensive clinical analysis.

Respond ONLY with a valid JSON object in this exact structure:
{{
  "explanation": "Detailed 3-4 paragraph clinical explanation of the predicted conditions, connecting reported symptoms to the ML predictions. Reference specific symptoms. Explain the pathophysiology briefly. Use medical terminology but explain it.",
  
  "follow_up_questions": [
    "Question 1 — specific, clinically relevant question to help narrow diagnosis",
    "Question 2 — ask about symptom duration, onset, or severity",
    "Question 3 — ask about relevant medical history or risk factors",
    "Question 4 — ask about associated symptoms not yet mentioned"
  ],
  
  "recommended_actions": [
    "Immediate action item 1",
    "Diagnostic test or examination to request",
    "Lifestyle or monitoring recommendation",
    "When to escalate to emergency care"
  ],
  
  "red_flags": [
    "List any symptoms from the input that are concerning and require urgent attention",
    "Only include if genuinely alarming — leave empty array [] if no red flags"
  ],
  
  "confidence_note": "One sentence explaining the ML model's confidence level and what it means clinically.",
  
  "disclaimer": "This analysis is generated by an AI system for educational and preliminary screening purposes only. It does NOT constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional."
}}"""

    return prompt


def build_followup_prompt(user_message: str, context: str) -> str:
    """Build prompt for follow-up questions in the chat."""
    return f"""FOLLOW-UP CONVERSATION

Medical Context from previous analysis:
{context}

Patient's response / new information:
{user_message}

Based on the new information, provide an updated analysis or answer the patient's question.
Maintain clinical focus and always encourage professional consultation.

Respond in the same JSON format as before."""


# ══════════════════════════════════════════════════════════
# LLM CLIENT MANAGER
# ══════════════════════════════════════════════════════════

class LLMManager:
    """
    Manages LLM provider selection and API calls.
    Falls back gracefully if no API key is configured.
    """

    def __init__(self):
        self.provider = self._detect_provider()
        self.client = self._init_client()
        logger.info(f"LLM Provider: {self.provider}")

    def _detect_provider(self) -> str:
        """Detect which LLM provider to use based on available API keys."""
        if os.getenv("OPENAI_API_KEY") and OPENAI_AVAILABLE:
            return "openai"
        elif os.getenv("ANTHROPIC_API_KEY") and ANTHROPIC_AVAILABLE:
            return "anthropic"
        elif os.getenv("USE_LOCAL_LLM", "false").lower() == "true":
            return "local"
        else:
            logger.warning("⚠️  No LLM API key found. Using rule-based fallback.")
            return "fallback"

    def _init_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == "openai":
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        return None

    def generate(self, prompt: str, system_prompt: str = SYSTEM_PROMPT,
                 temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """
        Send prompt to LLM and get response.

        Args:
            prompt: The main prompt
            system_prompt: System-level instructions
            temperature: Creativity (lower = more deterministic)
            max_tokens: Max response length

        Returns:
            Raw response text from LLM
        """
        if self.provider == "openai":
            return self._call_openai(prompt, system_prompt, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, system_prompt, temperature, max_tokens)
        elif self.provider == "local":
            return self._call_local(prompt, system_prompt)
        else:
            return self._rule_based_fallback(prompt)

    def _call_openai(self, prompt: str, system_prompt: str,
                     temperature: float, max_tokens: int) -> str:
        """Call OpenAI API."""
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    def _call_anthropic(self, prompt: str, system_prompt: str,
                        temperature: float, max_tokens: int) -> str:
        """Call Anthropic Claude API."""
        response = self.client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.content[0].text

    def _call_local(self, prompt: str, system_prompt: str) -> str:
        """Call local Ollama model."""
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": os.getenv("LOCAL_LLM_MODEL", "mistral"),
                "prompt": f"{system_prompt}\n\n{prompt}",
                "stream": False,
                "format": "json"
            },
            timeout=120
        )
        return response.json()["response"]

    def _rule_based_fallback(self, prompt: str) -> str:
        """
        Rule-based fallback when no LLM is available.
        Generates a structured response from ML predictions.
        """
        # Extract key info from prompt using regex
        diseases = re.findall(r'\d\. ([^(]+) \(Confidence', prompt)
        symptoms = re.findall(r'• ([^\n]+)', prompt)

        top_disease = diseases[0].strip() if diseases else "Unknown condition"
        symptom_list = ", ".join(symptoms[:5]) if symptoms else "the reported symptoms"

        fallback = {
            "explanation": (
                f"Based on the reported symptoms including {symptom_list}, "
                f"the ML model's primary prediction is {top_disease}. "
                f"This prediction is based on pattern matching against a trained dataset of "
                f"20 diseases and {len(symptoms)} symptom categories. "
                f"The confidence scores reflect the probability distribution across known conditions. "
                f"Note: This is a rule-based fallback response. Configure an OpenAI API key for "
                f"full LLM-powered explanations."
            ),
            "follow_up_questions": [
                "How long have you been experiencing these symptoms?",
                "Have you had any fever? If yes, what is the temperature?",
                "Do you have any known medical conditions or allergies?",
                "Have you recently travelled to any region with endemic diseases?"
            ],
            "recommended_actions": [
                "Consult a qualified healthcare professional for proper diagnosis",
                "Monitor your symptoms and note any changes",
                "Stay hydrated and rest if experiencing fever or fatigue",
                "Seek emergency care immediately if symptoms worsen rapidly"
            ],
            "red_flags": [
                "Chest pain or difficulty breathing — seek emergency care immediately",
                "High fever (above 39°C / 102°F) lasting more than 3 days",
                "Severe abdominal pain or blood in urine/stool"
            ],
            "confidence_note": (
                f"The ML model is most confident about {top_disease}. "
                "Lower confidence predictions should be considered as differential diagnoses."
            ),
            "disclaimer": (
                "This analysis is generated by an AI system for educational and preliminary screening "
                "purposes only. It does NOT constitute medical advice, diagnosis, or treatment. "
                "Always consult a qualified healthcare professional."
            )
        }

        return json.dumps(fallback)


# ══════════════════════════════════════════════════════════
# RESPONSE PARSER
# ══════════════════════════════════════════════════════════

def parse_llm_response(raw_text: str, model_used: str) -> LLMResponse:
    """
    Parse LLM JSON response into structured LLMResponse.
    Handles partial/malformed JSON gracefully.
    """
    try:
        # Clean response (remove markdown code blocks if present)
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"```(?:json)?", "", cleaned).strip().rstrip("```").strip()

        data = json.loads(cleaned)

        return LLMResponse(
            explanation=data.get("explanation", "Analysis not available."),
            follow_up_questions=data.get("follow_up_questions", []),
            recommended_actions=data.get("recommended_actions", []),
            red_flags=data.get("red_flags", []),
            confidence_note=data.get("confidence_note", ""),
            disclaimer=data.get("disclaimer", "This is not medical advice."),
            raw_text=raw_text,
            model_used=model_used
        )

    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON. Using raw text.")
        return LLMResponse(
            explanation=raw_text[:2000] if raw_text else "Analysis unavailable.",
            follow_up_questions=["Could you describe when your symptoms started?"],
            recommended_actions=["Please consult a qualified healthcare professional."],
            red_flags=[],
            confidence_note="",
            disclaimer="This is not medical advice. Consult a doctor.",
            raw_text=raw_text,
            model_used=model_used
        )


# ══════════════════════════════════════════════════════════
# HIGH-LEVEL LLM SERVICE
# ══════════════════════════════════════════════════════════

class LLMService:
    """
    Production LLM service called by FastAPI endpoints.
    Combines ML predictions + RAG context → LLM response.
    """

    def __init__(self):
        self.manager = LLMManager()

    def analyze(
        self,
        symptoms: List[str],
        ml_predictions: List[Dict],
        rag_context: str,
        chat_history: List[Dict] = None
    ) -> LLMResponse:
        """
        Full analysis pipeline: build prompt → call LLM → parse response.

        Args:
            symptoms: List of symptom strings
            ml_predictions: Output from MLPredictor.predict()
            rag_context: Retrieved medical context from RAGPipeline
            chat_history: Optional previous conversation turns

        Returns:
            Structured LLMResponse
        """
        logger.info(f"LLM analysis: {len(symptoms)} symptoms, {len(ml_predictions)} predictions")

        prompt = build_prediction_prompt(
            symptoms=symptoms,
            ml_predictions=ml_predictions,
            rag_context=rag_context,
            chat_history=chat_history
        )

        raw_response = self.manager.generate(prompt)
        return parse_llm_response(raw_response, self.manager.provider)

    def chat(self, user_message: str, context: str) -> LLMResponse:
        """Handle a follow-up chat message."""
        prompt = build_followup_prompt(user_message, context)
        raw_response = self.manager.generate(prompt)
        return parse_llm_response(raw_response, self.manager.provider)