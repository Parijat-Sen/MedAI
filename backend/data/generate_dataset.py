"""
============================================================
MedAI — Dataset Generator
============================================================
Generates a comprehensive symptom-disease dataset for training.
In production, replace with the Kaggle dataset:
  https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

This script creates:
  - training_data.csv  : symptom-disease pairs
  - symptom_list.json  : all unique symptoms
  - disease_info.json  : disease descriptions for RAG
============================================================
"""

import json
import random
import pandas as pd
import numpy as np
from pathlib import Path

# ── Seed for reproducibility ──────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Complete symptom-disease mapping ──────────────────────
DISEASE_SYMPTOM_MAP = {
    "Influenza (Flu)": {
        "core": ["fever", "chills", "muscle_aches", "fatigue", "headache"],
        "common": ["sore_throat", "runny_nose", "cough", "loss_of_appetite"],
        "rare": ["vomiting", "diarrhea", "chest_pain"],
        "description": "Influenza is a contagious respiratory illness caused by influenza viruses. It can cause mild to severe illness, and in some cases can lead to hospitalization or death.",
        "severity": "moderate",
        "specialist": "General Physician",
        "urgency": "See doctor within 24-48 hours if symptoms are severe"
    },
    "Common Cold": {
        "core": ["runny_nose", "sneezing", "sore_throat", "congestion"],
        "common": ["mild_fever", "cough", "watery_eyes", "fatigue"],
        "rare": ["headache", "muscle_aches", "loss_of_appetite"],
        "description": "The common cold is a viral infection of the upper respiratory tract. It is usually harmless and resolves on its own within 7-10 days.",
        "severity": "mild",
        "specialist": "General Physician",
        "urgency": "Self-care at home; see doctor if symptoms worsen after 10 days"
    },
    "Pneumonia": {
        "core": ["high_fever", "chest_pain", "cough_with_mucus", "shortness_of_breath"],
        "common": ["fatigue", "chills", "nausea", "vomiting"],
        "rare": ["confusion", "sweating", "rapid_heartbeat"],
        "description": "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough, fever, chills, and difficulty breathing.",
        "severity": "severe",
        "specialist": "Pulmonologist",
        "urgency": "Seek immediate medical attention"
    },
    "Diabetes Mellitus Type 2": {
        "core": ["frequent_urination", "excessive_thirst", "increased_hunger", "unexplained_weight_loss"],
        "common": ["fatigue", "blurred_vision", "slow_healing_wounds", "frequent_infections"],
        "rare": ["numbness_in_hands_feet", "tingling_sensation", "dark_skin_patches"],
        "description": "Type 2 diabetes is a chronic condition that affects the way the body metabolizes sugar (glucose). With type 2 diabetes, the body either doesn't produce enough insulin, or it doesn't use insulin properly.",
        "severity": "chronic",
        "specialist": "Endocrinologist",
        "urgency": "Schedule appointment with doctor within a week"
    },
    "Hypertension": {
        "core": ["headache", "dizziness", "blurred_vision", "chest_pain"],
        "common": ["shortness_of_breath", "nosebleeds", "fatigue", "irregular_heartbeat"],
        "rare": ["blood_in_urine", "pounding_in_chest", "vision_problems"],
        "description": "Hypertension (high blood pressure) is a common condition in which the long-term force of the blood against artery walls is high enough that it may eventually cause health problems.",
        "severity": "moderate",
        "specialist": "Cardiologist",
        "urgency": "See doctor within 1-2 days if BP is very high"
    },
    "Malaria": {
        "core": ["cyclical_fever", "chills", "sweating", "headache"],
        "common": ["nausea", "vomiting", "muscle_pain", "fatigue"],
        "rare": ["jaundice", "confusion", "anemia", "rapid_breathing"],
        "description": "Malaria is a serious and sometimes fatal disease caused by a parasite that commonly infects a certain type of mosquito that feeds on humans.",
        "severity": "severe",
        "specialist": "Infectious Disease Specialist",
        "urgency": "Seek immediate medical attention"
    },
    "Typhoid Fever": {
        "core": ["sustained_high_fever", "abdominal_pain", "headache", "weakness"],
        "common": ["constipation", "diarrhea", "loss_of_appetite", "rose_spots"],
        "rare": ["confusion", "agitation", "hallucinations", "relative_bradycardia"],
        "description": "Typhoid fever is a bacterial infection caused by Salmonella typhi. It spreads through contaminated food and water and causes high fever, abdominal pain, and other symptoms.",
        "severity": "severe",
        "specialist": "Infectious Disease Specialist",
        "urgency": "Seek medical attention immediately"
    },
    "Dengue Fever": {
        "core": ["sudden_high_fever", "severe_headache", "pain_behind_eyes", "joint_muscle_pain"],
        "common": ["skin_rash", "mild_bleeding", "nausea", "vomiting"],
        "rare": ["severe_abdominal_pain", "rapid_breathing", "bleeding_gums", "fatigue"],
        "description": "Dengue fever is a mosquito-borne tropical disease caused by the dengue virus. It causes flu-like symptoms including fever, headache, muscle and joint pains, and a characteristic rash.",
        "severity": "severe",
        "specialist": "Infectious Disease Specialist",
        "urgency": "Seek immediate medical attention"
    },
    "Migraine": {
        "core": ["severe_headache", "throbbing_pain_one_side", "nausea", "sensitivity_to_light"],
        "common": ["sensitivity_to_sound", "vomiting", "visual_disturbances", "aura"],
        "rare": ["neck_stiffness", "confusion", "tingling_in_face", "weakness"],
        "description": "A migraine is a powerful headache that often happens with nausea, vomiting, and extreme sensitivity to light and sound. Migraine attacks can last for hours to days.",
        "severity": "moderate",
        "specialist": "Neurologist",
        "urgency": "See doctor if migraines are frequent or worsening"
    },
    "Gastroenteritis": {
        "core": ["diarrhea", "nausea", "vomiting", "stomach_cramps"],
        "common": ["fever", "loss_of_appetite", "dehydration", "headache"],
        "rare": ["muscle_aches", "blood_in_stool", "severe_abdominal_pain"],
        "description": "Gastroenteritis is inflammation of the stomach and intestines, typically caused by a viral or bacterial infection. It causes diarrhea, vomiting, and stomach pain.",
        "severity": "mild",
        "specialist": "Gastroenterologist",
        "urgency": "Self-care; see doctor if symptoms last more than 3 days"
    },
    "Asthma": {
        "core": ["shortness_of_breath", "wheezing", "chest_tightness", "coughing"],
        "common": ["difficulty_sleeping", "shortness_of_breath_on_exercise", "fatigue"],
        "rare": ["rapid_breathing", "bluish_lips", "anxiety", "sweating"],
        "description": "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out and shortness of breath.",
        "severity": "moderate",
        "specialist": "Pulmonologist / Allergist",
        "urgency": "Seek emergency help if breathing is severely impaired"
    },
    "Urinary Tract Infection (UTI)": {
        "core": ["burning_urination", "frequent_urination", "cloudy_urine", "pelvic_pain"],
        "common": ["strong_smelling_urine", "blood_in_urine", "lower_back_pain", "fever"],
        "rare": ["nausea", "vomiting", "shaking_chills", "confusion"],
        "description": "A urinary tract infection (UTI) is an infection in any part of the urinary system. Most infections involve the lower urinary tract — the bladder and the urethra.",
        "severity": "mild",
        "specialist": "Urologist",
        "urgency": "See doctor within 1-2 days"
    },
    "Anemia": {
        "core": ["fatigue", "weakness", "pale_skin", "shortness_of_breath"],
        "common": ["dizziness", "irregular_heartbeat", "chest_pain", "cold_extremities"],
        "rare": ["headache", "cognitive_problems", "brittle_nails", "unusual_cravings"],
        "description": "Anemia is a condition in which you lack enough healthy red blood cells to carry adequate oxygen to your body's tissues. Having anemia, also referred to as low hemoglobin, can make you feel tired and weak.",
        "severity": "moderate",
        "specialist": "Hematologist",
        "urgency": "See doctor within a few days"
    },
    "Appendicitis": {
        "core": ["severe_abdominal_pain_right_lower", "nausea", "vomiting", "fever"],
        "common": ["loss_of_appetite", "abdominal_bloating", "inability_to_pass_gas", "diarrhea"],
        "rare": ["constipation", "inability_to_stand_straight", "back_pain"],
        "description": "Appendicitis is an inflammation of the appendix. It's a medical emergency that almost always requires prompt surgery to remove the appendix.",
        "severity": "emergency",
        "specialist": "Surgeon",
        "urgency": "EMERGENCY - Seek immediate medical attention"
    },
    "Tuberculosis (TB)": {
        "core": ["persistent_cough_3_weeks", "coughing_blood", "chest_pain", "weight_loss"],
        "common": ["fatigue", "fever", "night_sweats", "chills"],
        "rare": ["loss_of_appetite", "swollen_lymph_nodes", "joint_pain"],
        "description": "Tuberculosis (TB) is a potentially serious infectious disease that mainly affects the lungs. The bacteria that cause tuberculosis are spread from person to person through tiny droplets released into the air.",
        "severity": "severe",
        "specialist": "Pulmonologist / Infectious Disease Specialist",
        "urgency": "See doctor immediately"
    },
    "COVID-19": {
        "core": ["fever", "dry_cough", "fatigue", "loss_of_taste_smell"],
        "common": ["sore_throat", "headache", "body_aches", "shortness_of_breath"],
        "rare": ["chest_pain", "confusion", "difficulty_speaking", "skin_rash"],
        "description": "COVID-19 is a respiratory illness caused by the SARS-CoV-2 virus. Symptoms range from mild to severe and can include fever, cough, and difficulty breathing.",
        "severity": "moderate",
        "specialist": "General Physician / Pulmonologist",
        "urgency": "Isolate and consult doctor; seek emergency care if breathing is difficult"
    },
    "Chicken Pox": {
        "core": ["itchy_blisters", "fever", "tiredness", "loss_of_appetite"],
        "common": ["headache", "stomach_ache", "skin_rash_spreading"],
        "rare": ["secondary_bacterial_infection", "pneumonia", "brain_inflammation"],
        "description": "Chickenpox is a highly contagious viral infection causing an itchy, blister-like rash on the skin. It is caused by the varicella-zoster virus.",
        "severity": "mild",
        "specialist": "General Physician / Dermatologist",
        "urgency": "Consult doctor for antiviral treatment if needed"
    },
    "Jaundice": {
        "core": ["yellow_skin", "yellow_eyes", "dark_urine", "fatigue"],
        "common": ["abdominal_pain", "pale_stool", "itching", "weight_loss"],
        "rare": ["fever", "vomiting", "loss_of_appetite", "confusion"],
        "description": "Jaundice is a condition in which the skin, whites of the eyes and mucous membranes turn yellow because of a high level of bilirubin, a yellow-orange bile pigment.",
        "severity": "moderate",
        "specialist": "Gastroenterologist / Hepatologist",
        "urgency": "See doctor within 1-2 days"
    },
    "Arthritis": {
        "core": ["joint_pain", "joint_stiffness", "joint_swelling", "reduced_range_of_motion"],
        "common": ["fatigue", "warmth_around_joints", "redness_around_joints"],
        "rare": ["fever", "weight_loss", "anemia", "skin_nodules"],
        "description": "Arthritis is inflammation of one or more joints, causing pain and stiffness that can worsen with age. The most common types are osteoarthritis and rheumatoid arthritis.",
        "severity": "chronic",
        "specialist": "Rheumatologist",
        "urgency": "Schedule appointment with specialist"
    },
    "Heart Attack (Myocardial Infarction)": {
        "core": ["chest_pain_pressure", "pain_radiating_to_arm_jaw", "shortness_of_breath", "sweating"],
        "common": ["nausea", "lightheadedness", "fatigue", "rapid_heartbeat"],
        "rare": ["indigestion_like_pain", "upper_back_pain", "anxiety", "cold_sweat"],
        "description": "A heart attack occurs when the flow of blood to the heart is severely reduced or blocked. The blockage is usually due to a buildup of fat, cholesterol and other substances in the heart arteries.",
        "severity": "emergency",
        "specialist": "Cardiologist",
        "urgency": "EMERGENCY - Call ambulance immediately (dial 112)"
    }
}

# ── All unique symptoms ────────────────────────────────────
ALL_SYMPTOMS = sorted(set(
    symptom
    for disease_data in DISEASE_SYMPTOM_MAP.values()
    for category in ["core", "common", "rare"]
    for symptom in disease_data[category]
))


def generate_patient_record(disease_name: str, disease_data: dict) -> dict:
    """Generate a single synthetic patient record."""
    record = {symptom: 0 for symptom in ALL_SYMPTOMS}

    # Always include core symptoms (high probability)
    for symptom in disease_data["core"]:
        if symptom in record:
            record[symptom] = 1

    # Include common symptoms with 70% probability
    for symptom in disease_data["common"]:
        if symptom in record and random.random() < 0.70:
            record[symptom] = 1

    # Include rare symptoms with 25% probability
    for symptom in disease_data["rare"]:
        if symptom in record and random.random() < 0.25:
            record[symptom] = 1

    # Add noise: randomly toggle 1-2 non-core symptoms
    non_core = [s for s in ALL_SYMPTOMS if s not in disease_data["core"]]
    noise_symptoms = random.sample(non_core, min(2, len(non_core)))
    for symptom in noise_symptoms:
        if random.random() < 0.10:
            record[symptom] = 1 - record[symptom]

    record["disease"] = disease_name
    return record


def generate_dataset(samples_per_disease: int = 150) -> pd.DataFrame:
    """Generate the full training dataset."""
    all_records = []

    print("📊 Generating synthetic symptom-disease dataset...")
    for disease, data in DISEASE_SYMPTOM_MAP.items():
        print(f"   → Generating {samples_per_disease} samples for: {disease}")
        for _ in range(samples_per_disease):
            record = generate_patient_record(disease, data)
            all_records.append(record)

    df = pd.DataFrame(all_records)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n✅ Dataset generated: {len(df)} total samples, {df['disease'].nunique()} diseases")
    print(f"   Symptoms: {len(ALL_SYMPTOMS)}")
    return df


def save_disease_info():
    """Save disease information for RAG pipeline."""
    disease_info = {}
    for disease, data in DISEASE_SYMPTOM_MAP.items():
        disease_info[disease] = {
            "description": data["description"],
            "severity": data["severity"],
            "specialist": data["specialist"],
            "urgency": data["urgency"],
            "all_symptoms": data["core"] + data["common"] + data["rare"]
        }
    return disease_info


def generate_medical_documents():
    """Generate medical knowledge documents for RAG."""
    documents = []

    for disease, data in DISEASE_SYMPTOM_MAP.items():
        doc = f"""
DISEASE: {disease}
SEVERITY: {data['severity'].upper()}

DESCRIPTION:
{data['description']}

SYMPTOMS:
Primary Symptoms: {', '.join(data['core'])}
Common Symptoms: {', '.join(data['common'])}
Rare Symptoms: {', '.join(data['rare'])}

SPECIALIST: {data['specialist']}
URGENCY: {data['urgency']}

TREATMENT OVERVIEW:
Treatment for {disease} typically involves consultation with a {data['specialist']}.
{'Emergency medical care is required immediately.' if data['severity'] == 'emergency' else
 'Prompt medical evaluation is recommended.' if data['severity'] == 'severe' else
 'Medical consultation is advised.' if data['severity'] == 'moderate' else
 'Ongoing management and monitoring is required.' if data['severity'] == 'chronic' else
 'Most cases can be managed with appropriate rest and medication.'}

DISCLAIMER:
This information is for educational purposes only and should not replace professional medical advice.
Always consult a qualified healthcare provider for diagnosis and treatment.
"""
        documents.append({"title": disease, "content": doc.strip()})

    return documents


if __name__ == "__main__":
    # Create data directory
    data_dir = Path("backend/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate and save training dataset
    df = generate_dataset(samples_per_disease=150)
    df.to_csv(data_dir / "training_data.csv", index=False)
    print(f"💾 Saved: {data_dir}/training_data.csv")

    # Save symptom list
    with open(data_dir / "symptom_list.json", "w") as f:
        json.dump(ALL_SYMPTOMS, f, indent=2)
    print(f"💾 Saved: {data_dir}/symptom_list.json")

    # Save disease info
    disease_info = save_disease_info()
    with open(data_dir / "disease_info.json", "w") as f:
        json.dump(disease_info, f, indent=2)
    print(f"💾 Saved: {data_dir}/disease_info.json")

    # Save medical documents for RAG
    documents = generate_medical_documents()
    with open(data_dir / "medical_documents.json", "w") as f:
        json.dump(documents, f, indent=2)
    print(f"💾 Saved: {data_dir}/medical_documents.json")

    print("\n✅ All data files generated successfully!")
    print(f"   Total diseases: {len(DISEASE_SYMPTOM_MAP)}")
    print(f"   Total symptoms: {len(ALL_SYMPTOMS)}")
    print(f"   Dataset shape: {df.shape}")