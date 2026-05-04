"""
MedAI — LLM Integration Module (FINAL STABLE VERSION)
"""

import os
import json
import logging
import re
from typing import List, Dict
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# =========================
# IMPORT OPENAI SDK
# =========================
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ✅ DEBUG
print("OPENAI AVAILABLE:", OPENAI_AVAILABLE)


# =========================
# RESPONSE STRUCTURE
# =========================
@dataclass
class LLMResponse:
    explanation: str
    follow_up_questions: List[str]
    recommended_actions: List[str]
    red_flags: List[str]
    confidence_note: str
    disclaimer: str
    raw_text: str
    model_used: str


# =========================
# SYSTEM PROMPT
# =========================
SYSTEM_PROMPT = """You are a medical assistant AI.

Rules:
- Do NOT give exact diagnosis
- Do NOT give medicine/dosage
- Explain clearly
- Ask follow-up questions
- Always include disclaimer

Respond ONLY in valid JSON format:
{
 "explanation": "...",
 "follow_up_questions": [],
 "recommended_actions": [],
 "red_flags": [],
 "confidence_note": "...",
 "disclaimer": "..."
}
"""


# =========================
# LLM MANAGER
# =========================
class LLMManager:

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")

        print("DEBUG LLM KEY:", self.api_key)

        self.provider = self._detect_provider()
        self.client = self._init_client()

        logger.info(f"LLM Provider: {self.provider}")

    def _detect_provider(self):
        if self.api_key and OPENAI_AVAILABLE:
            logger.info("✅ OpenRouter API key detected")
            return "openai"
        else:
            logger.warning("⚠️ No API key → fallback mode")
            return "fallback"

    def _init_client(self):
        if self.provider == "openai":
            return OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1",
                default_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "MedAI Project"
                }
            )
        return None

    def generate(self, prompt: str):
        if self.provider == "openai":
            return self._call_openai(prompt)
        else:
            return self._fallback()

    def _call_openai(self, prompt: str):
        try:
            print("CALLING MODEL:", self.model)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            content = response.choices[0].message.content

            if not content:
                raise ValueError("Empty response")

            return content

        except Exception as e:
            logger.error(f"❌ OpenRouter Error: {e}")
            return self._fallback()

    def _fallback(self):
        return json.dumps({
            "explanation": "Basic analysis based on symptoms.",
            "follow_up_questions": ["How long symptoms?", "Any fever?"],
            "recommended_actions": ["Consult doctor"],
            "red_flags": [],
            "confidence_note": "Fallback response",
            "disclaimer": "This is not medical advice"
        })


# =========================
# PARSER
# =========================
def parse_llm_response(raw_text: str, model_used: str) -> LLMResponse:
    try:
        cleaned = raw_text.strip()

        if cleaned.startswith("```"):
            cleaned = re.sub(r"```(?:json)?", "", cleaned).strip().rstrip("```").strip()

        data = json.loads(cleaned)

        return LLMResponse(
            explanation=data.get("explanation", ""),
            follow_up_questions=data.get("follow_up_questions", []),
            recommended_actions=data.get("recommended_actions", []),
            red_flags=data.get("red_flags", []),
            confidence_note=data.get("confidence_note", ""),
            disclaimer=data.get("disclaimer", ""),
            raw_text=raw_text,
            model_used=model_used
        )

    except Exception as e:
        logger.error(f"❌ JSON Parse Error: {e}")
        logger.error(f"RAW RESPONSE: {raw_text}")

        return LLMResponse(
            explanation="Error generating response.",
            follow_up_questions=[],
            recommended_actions=[],
            red_flags=[],
            confidence_note="",
            disclaimer="Not medical advice",
            raw_text=raw_text,
            model_used=model_used
        )


# =========================
# SERVICE
# =========================
class LLMService:

    def __init__(self):
        self.manager = LLMManager()

    def analyze(self, symptoms: List[str], ml_predictions: List[Dict], rag_context: str):

        rag_context = rag_context or "No additional context"

        prompt = f"""
Symptoms: {symptoms}

Predictions: {ml_predictions}

Context: {rag_context}

Explain clearly, ask follow-up questions, suggest actions.
"""

        raw = self.manager.generate(prompt)
        return parse_llm_response(raw, self.manager.provider)

    def chat(self, message: str, context: str):

        context = context or "No previous context"

        prompt = f"""
Context: {context}

User: {message}

Provide updated analysis.
"""

        raw = self.manager.generate(prompt)
        return parse_llm_response(raw, self.manager.provider)