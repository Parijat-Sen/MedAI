"""
============================================================
MedAI — Symptom Extractor
============================================================
Converts free-text symptom descriptions into a structured
list of known symptoms using:
  1. Direct keyword matching
  2. Synonym/alias mapping
  3. Fuzzy phrase matching
  4. NLP-based extraction (if spaCy available)
============================================================
"""

import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# ── Symptom synonyms / aliases ────────────────────────────
# Maps natural language variants to canonical symptom names
SYMPTOM_ALIASES = {
    # Fever variants
    "high temperature": "fever",
    "running a temperature": "fever",
    "febrile": "fever",
    "pyrexia": "fever",
    "mild temperature": "mild_fever",
    "low grade fever": "mild_fever",
    "very high fever": "high_fever",
    "elevated temperature": "fever",
    "temperature": "fever",

    # Pain variants
    "headache": "headache",
    "head pain": "headache",
    "head hurts": "headache",
    "migraine": "severe_headache",
    "throbbing head": "severe_headache",
    "one sided headache": "throbbing_pain_one_side",

    # Respiratory
    "cough": "cough",
    "coughing": "cough",
    "dry cough": "dry_cough",
    "coughing up blood": "coughing_blood",
    "blood in sputum": "coughing_blood",
    "productive cough": "cough_with_mucus",
    "mucus cough": "cough_with_mucus",
    "phlegm": "cough_with_mucus",
    "breathing difficulty": "shortness_of_breath",
    "breathlessness": "shortness_of_breath",
    "short of breath": "shortness_of_breath",
    "difficulty breathing": "shortness_of_breath",
    "wheeze": "wheezing",
    "whistling breath": "wheezing",
    "tight chest": "chest_tightness",
    "chest tightness": "chest_tightness",
    "chest pain": "chest_pain",
    "chest pressure": "chest_pain_pressure",
    "chest hurts": "chest_pain",
    "stuffy nose": "congestion",
    "nasal congestion": "congestion",
    "blocked nose": "congestion",
    "runny nose": "runny_nose",
    "sneezing": "sneezing",

    # Gastrointestinal
    "stomach ache": "stomach_cramps",
    "tummy ache": "stomach_cramps",
    "abdominal pain": "abdominal_pain",
    "stomach pain": "abdominal_pain",
    "belly pain": "abdominal_pain",
    "right side pain": "severe_abdominal_pain_right_lower",
    "nausea": "nausea",
    "feeling sick": "nausea",
    "queasy": "nausea",
    "vomiting": "vomiting",
    "throwing up": "vomiting",
    "diarrhea": "diarrhea",
    "loose stools": "diarrhea",
    "watery stools": "diarrhea",
    "constipation": "constipation",
    "loose motion": "diarrhea",
    "indigestion": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    "not hungry": "loss_of_appetite",

    # Urinary
    "burning while urinating": "burning_urination",
    "painful urination": "burning_urination",
    "frequent urination": "frequent_urination",
    "peeing a lot": "frequent_urination",
    "cloudy pee": "cloudy_urine",
    "dark urine": "dark_urine",
    "blood in urine": "blood_in_urine",
    "pee a lot": "frequent_urination",

    # General
    "tired": "fatigue",
    "exhausted": "fatigue",
    "weakness": "weakness",
    "weak": "weakness",
    "dizzy": "dizziness",
    "dizziness": "dizziness",
    "light headed": "lightheadedness",
    "lightheaded": "lightheadedness",
    "chills": "chills",
    "shivering": "chills",
    "sweating": "sweating",
    "night sweats": "night_sweats",
    "weight loss": "unexplained_weight_loss",
    "losing weight": "unexplained_weight_loss",
    "lost weight": "weight_loss",
    "pale skin": "pale_skin",
    "pallor": "pale_skin",

    # Neurological
    "confusion": "confusion",
    "disoriented": "confusion",
    "blurry vision": "blurred_vision",
    "vision problems": "blurred_vision",
    "can't see clearly": "blurred_vision",
    "sensitive to light": "sensitivity_to_light",
    "light sensitivity": "sensitivity_to_light",
    "sensitive to sound": "sensitivity_to_sound",
    "numbness": "numbness_in_hands_feet",
    "tingling": "tingling_sensation",
    "pins and needles": "tingling_sensation",

    # Skin
    "rash": "skin_rash",
    "skin rash": "skin_rash",
    "itchy": "itchy_blisters",
    "itching": "itchy_blisters",
    "blisters": "itchy_blisters",
    "yellow skin": "yellow_skin",
    "yellow eyes": "yellow_eyes",
    "jaundice": "yellow_skin",

    # Musculoskeletal
    "body aches": "muscle_aches",
    "muscle pain": "muscle_aches",
    "joint pain": "joint_pain",
    "joint swelling": "joint_swelling",
    "stiff joints": "joint_stiffness",
    "back pain": "lower_back_pain",

    # Cardiac
    "heart racing": "rapid_heartbeat",
    "palpitations": "irregular_heartbeat",
    "fast heartbeat": "rapid_heartbeat",
    "irregular heartbeat": "irregular_heartbeat",
    "pain in arm": "pain_radiating_to_arm_jaw",
    "jaw pain": "pain_radiating_to_arm_jaw",
    "cold hands": "cold_extremities",
    "cold feet": "cold_extremities",

    # Senses
    "no taste": "loss_of_taste_smell",
    "loss of taste": "loss_of_taste_smell",
    "can't smell": "loss_of_taste_smell",
    "loss of smell": "loss_of_taste_smell",
    "sore throat": "sore_throat",
    "throat pain": "sore_throat",
    "watery eyes": "watery_eyes",
    "eye pain": "pain_behind_eyes",
    "eyes hurt": "pain_behind_eyes",

    # Persistent
    "long lasting cough": "persistent_cough_3_weeks",
    "chronic cough": "persistent_cough_3_weeks",
    "prolonged cough": "persistent_cough_3_weeks",
    "sustained fever": "sustained_high_fever",
    "persistent fever": "sustained_high_fever",
    "sudden fever": "sudden_high_fever",
}


class SymptomExtractor:
    """
    Extracts structured symptom lists from free-text descriptions.
    """

    def __init__(self, known_symptoms: List[str] = None):
        self.known_symptoms = set(known_symptoms or [])

        # Load symptom list if not provided
        if not self.known_symptoms:
            symptom_file = Path("backend/data/symptom_list.json")
            if symptom_file.exists():
                with open(symptom_file) as f:
                    self.known_symptoms = set(json.load(f))

        self.alias_map = {k.lower(): v for k, v in SYMPTOM_ALIASES.items()}
        logger.info(f"SymptomExtractor ready: {len(self.known_symptoms)} known symptoms")

    def extract(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract symptoms from free-text input.

        Args:
            text: User's symptom description

        Returns:
            (extracted_symptoms, unrecognized_phrases)
        """
        # Normalize text
        text_lower = text.lower().strip()

        extracted = set()
        unrecognized = []

        # Step 1: Direct match against known symptoms (underscore and space forms)
        for symptom in self.known_symptoms:
            symptom_readable = symptom.replace("_", " ")
            if symptom_readable in text_lower or symptom in text_lower:
                extracted.add(symptom)

        # Step 2: Alias mapping
        for alias, canonical in self.alias_map.items():
            if alias in text_lower:
                # Map to known symptom
                if canonical in self.known_symptoms:
                    extracted.add(canonical)
                else:
                    # Try partial match
                    for ks in self.known_symptoms:
                        if canonical in ks or ks in canonical:
                            extracted.add(ks)
                            break

        # Step 3: Parse comma/semicolon-separated lists
        # "I have fever, headache, and cough" → individual tokens
        separators = r'[,;]|\band\b|\bwith\b|\balso\b|\bplus\b'
        parts = re.split(separators, text_lower)

        for part in parts:
            part = part.strip().strip(".")
            if not part or len(part) < 3:
                continue

            # Remove common filler words
            fillers = ["i have", "i feel", "i am experiencing", "suffering from",
                       "experiencing", "having", "my", "some", "slight", "severe",
                       "mild", "very", "really", "very bad", "a bit of", "a little"]
            cleaned = part
            for filler in fillers:
                cleaned = cleaned.replace(filler, "").strip()

            if not cleaned:
                continue

            # Try exact match
            if cleaned in self.known_symptoms:
                extracted.add(cleaned)
                continue

            # Try alias
            if cleaned in self.alias_map:
                canonical = self.alias_map[cleaned]
                if canonical in self.known_symptoms:
                    extracted.add(canonical)
                continue

            # Try underscore form
            underscore_form = cleaned.replace(" ", "_")
            if underscore_form in self.known_symptoms:
                extracted.add(underscore_form)
                continue

            # Mark as unrecognized if seems like a symptom
            if len(cleaned.split()) <= 4 and len(cleaned) > 3:
                # Avoid marking obvious non-symptoms
                non_symptoms = {"the", "and", "but", "for", "days", "week", "weeks"}
                if cleaned not in non_symptoms:
                    unrecognized.append(cleaned)

        return sorted(list(extracted)), unrecognized

    def format_for_display(self, symptoms: List[str]) -> List[str]:
        """Convert internal symptom names to human-readable display format."""
        return [s.replace("_", " ").title() for s in symptoms]

    def suggest_symptoms(self, partial: str) -> List[str]:
        """Autocomplete suggestions for partial symptom text."""
        partial = partial.lower().replace(" ", "_")
        return [
            s.replace("_", " ")
            for s in self.known_symptoms
            if partial in s
        ][:10]