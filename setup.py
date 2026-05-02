"""
============================================================
MedAI — One-Click Setup Script
============================================================
Run this ONCE before starting the server.
It will:
  1. Generate the dataset
  2. Train all ML models
  3. Build the FAISS RAG index
  4. Verify everything works
============================================================
Usage:
    python setup.py
============================================================
"""

import sys
import json
import time
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def banner(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def check_imports():
    banner("Step 0: Checking Dependencies")
    required = [
        "sklearn", "xgboost", "pandas", "numpy",
        "faiss", "sentence_transformers", "fastapi", "uvicorn"
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} — MISSING")
            missing.append(pkg)

    if missing:
        print(f"\n⚠️  Missing packages: {missing}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    print("\n✅ All dependencies found.")


def step1_generate_data():
    banner("Step 1: Generating Dataset")
    sys.path.insert(0, ".")
    from backend.data.generate_dataset import (
        generate_dataset, save_disease_info,
        generate_medical_documents, ALL_SYMPTOMS
    )
    import json

    data_dir = Path("backend/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    df = generate_dataset(samples_per_disease=150)
    df.to_csv(data_dir / "training_data.csv", index=False)

    with open(data_dir / "symptom_list.json", "w") as f:
        json.dump(ALL_SYMPTOMS, f, indent=2)

    disease_info = save_disease_info()
    with open(data_dir / "disease_info.json", "w") as f:
        json.dump(disease_info, f, indent=2)

    documents = generate_medical_documents()
    with open(data_dir / "medical_documents.json", "w") as f:
        json.dump(documents, f, indent=2)

    print(f"\n✅ Dataset ready: {len(df)} samples, {df['disease'].nunique()} diseases")


def step2_train_models():
    banner("Step 2: Training ML Models")
    from backend.ml.train_model import train_all_models
    train_all_models()
    print("\n✅ ML models trained and saved.")


def step3_build_rag():
    banner("Step 3: Building RAG Index (FAISS)")
    from backend.rag.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    pipeline._build_from_documents()

    # Quick retrieval test
    results = pipeline.retrieve("fever headache malaria", top_k=2)
    print(f"\n  RAG test query: 'fever headache malaria'")
    for r in results:
        print(f"  [{r.score:.3f}] {r.source}: {r.content[:80]}...")

    print("\n✅ FAISS index built and tested.")


def step4_verify():
    banner("Step 4: Final Verification")

    # Check all required files exist
    required_files = [
        "backend/data/training_data.csv",
        "backend/data/symptom_list.json",
        "backend/data/disease_info.json",
        "backend/data/medical_documents.json",
        "backend/ml/saved_models/best_model.pkl",
        "backend/ml/saved_models/label_encoder.pkl",
        "backend/ml/saved_models/model_metadata.json",
        "backend/rag/faiss_index/medical.index",
        "backend/rag/faiss_index/chunks.pkl",
    ]

    all_ok = True
    for filepath in required_files:
        exists = Path(filepath).exists()
        status = "✅" if exists else "❌"
        size = Path(filepath).stat().st_size if exists else 0
        print(f"  {status} {filepath} ({size:,} bytes)")
        if not exists:
            all_ok = False

    # Test ML predictor
    print("\n  Testing ML predictor...")
    from backend.ml.predictor import MLPredictor
    predictor = MLPredictor()
    result = predictor.predict(["fever", "headache", "chills", "muscle_aches"])
    preds = result.get("predictions", [])
    print(f"  Test prediction: {preds[0]['disease']} ({preds[0]['confidence']:.1f}%)" if preds else "  No predictions")

    if all_ok:
        print("\n" + "="*60)
        print("  ✅ SETUP COMPLETE — All systems ready!")
        print("="*60)
        print("\n  Start the server with:")
        print("  → python run.py")
        print("  → Then open: http://localhost:8000")
    else:
        print("\n❌ Some files are missing. Check errors above.")


if __name__ == "__main__":
    print("\n" + "█"*60)
    print("  MedAI — Symptom-Based Disease Prediction System")
    print("  B.Tech Final Year Project — Setup Script")
    print("█"*60)

    start = time.time()

    try:
        check_imports()
        step1_generate_data()
        step2_train_models()
        step3_build_rag()
        step4_verify()
    except Exception as e:
        logger.error(f"\n❌ Setup failed at: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start
    print(f"\n⏱️  Total setup time: {elapsed:.1f} seconds\n")