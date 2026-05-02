"""
============================================================
MedAI — ML Model Training Module
============================================================
Trains multiple ML models on the symptom-disease dataset:
  - Random Forest (primary)
  - XGBoost (secondary)
  - Gradient Boosting (tertiary)

Includes:
  - Data preprocessing pipeline
  - SMOTE for class balancing
  - Cross-validation
  - Comprehensive evaluation metrics
  - Model persistence
============================================================
"""

import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── Logging setup ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────
DATA_PATH = Path("backend/data/training_data.csv")
SYMPTOM_LIST_PATH = Path("backend/data/symptom_list.json")
MODEL_SAVE_PATH = Path("backend/ml/saved_models")
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════

def load_data() -> Tuple[pd.DataFrame, pd.Series, list, list]:
    """
    Load and preprocess the dataset.
    Returns: (X, y, feature_names, class_names)
    """
    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Diseases: {df['disease'].nunique()}")

    # Load symptom list for consistent feature ordering
    with open(SYMPTOM_LIST_PATH) as f:
        symptom_list = json.load(f)

    # Ensure all symptom columns exist (some may be missing if dataset differs)
    for symptom in symptom_list:
        if symptom not in df.columns:
            df[symptom] = 0

    X = df[symptom_list]
    y = df["disease"]
    feature_names = symptom_list
    class_names = sorted(y.unique().tolist())

    logger.info(f"Features: {len(feature_names)} symptoms")
    logger.info(f"Classes: {len(class_names)} diseases")
    logger.info(f"Class distribution:\n{y.value_counts()}")

    return X, y, feature_names, class_names


def preprocess_data(X: pd.DataFrame, y: pd.Series):
    """
    Encode labels, apply SMOTE for class balancing.
    Returns: (X_resampled, y_resampled, label_encoder)
    """
    # Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Apply SMOTE to handle class imbalance
    logger.info("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X, y_encoded)
    logger.info(f"After SMOTE: {X_resampled.shape[0]} samples")

    return X_resampled, y_resampled, le


# ══════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════

def get_models() -> Dict[str, Any]:
    """Return dictionary of models to train and evaluate."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=0
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=42,
            verbosity=0
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
    }


# ══════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   class_names: list, model_name: str) -> Dict[str, Any]:
    """Run comprehensive evaluation on test set."""
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision_macro": round(precision_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro", zero_division=0), 4),
        "precision_weighted": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"MODEL: {model_name}")
    logger.info(f"{'='*60}")
    for k, v in metrics.items():
        logger.info(f"  {k:<25} : {v:.4f}")

    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    logger.info(f"\nClassification Report:\n{report}")

    return metrics


def cross_validate_model(model, X: np.ndarray, y: np.ndarray, model_name: str) -> float:
    """Run stratified k-fold cross-validation."""
    logger.info(f"\nCross-validating {model_name}...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    logger.info(f"  CV F1 Macro: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()


# ══════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════

def get_top_features(model, feature_names: list, top_n: int = 20) -> list:
    """Extract top N most important features from the model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [
            {"symptom": feature_names[i], "importance": round(float(importances[i]), 6)}
            for i in indices
        ]
        return top_features
    return []


# ══════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════

def train_all_models():
    """Main training pipeline — trains, evaluates, and saves all models."""
    logger.info("\n" + "="*60)
    logger.info("   MedAI — ML Model Training Pipeline")
    logger.info("="*60)

    # 1. Load data
    X, y, feature_names, class_names = load_data()

    # 2. Preprocess
    X_processed, y_processed, label_encoder = preprocess_data(X, y)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed,
        test_size=0.20,
        random_state=42,
        stratify=y_processed
    )
    logger.info(f"\nSplit: Train={len(X_train)}, Test={len(X_test)}")

    # 4. Train all models
    models = get_models()
    all_metrics = {}
    best_model_name = None
    best_f1 = -1

    for model_name, model in models.items():
        logger.info(f"\n🏋️  Training {model_name}...")
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(model, X_test, y_test, class_names, model_name)
        cv_score = cross_validate_model(model, X_processed, y_processed, model_name)
        metrics["cv_f1_macro"] = round(cv_score, 4)
        all_metrics[model_name] = metrics

        # Track best model
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_model_name = model_name

        # Save individual model
        model_path = MODEL_SAVE_PATH / f"{model_name.lower()}_model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"💾 Saved: {model_path}")

    # 5. Save best model as primary
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    # Re-train best model on full data for production
    logger.info(f"\n🏆 Best Model: {best_model_name} (F1={best_f1:.4f})")
    best_model.fit(X_processed, y_processed)

    joblib.dump(best_model, MODEL_SAVE_PATH / "best_model.pkl")
    joblib.dump(label_encoder, MODEL_SAVE_PATH / "label_encoder.pkl")
    logger.info("💾 Saved: best_model.pkl + label_encoder.pkl")

    # 6. Save metadata
    top_features = get_top_features(best_model, feature_names)
    metadata = {
        "best_model": best_model_name,
        "class_names": class_names,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "num_classes": len(class_names),
        "metrics": all_metrics,
        "top_features": top_features,
        "training_samples": int(len(X_processed)),
    }
    with open(MODEL_SAVE_PATH / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("💾 Saved: model_metadata.json")

    # 7. Print comparison table
    logger.info("\n" + "="*60)
    logger.info("  MODEL COMPARISON TABLE")
    logger.info("="*60)
    logger.info(f"{'Model':<25} {'Accuracy':>10} {'F1 Macro':>10} {'CV F1':>10}")
    logger.info("-"*60)
    for name, m in all_metrics.items():
        marker = " ★" if name == best_model_name else ""
        logger.info(f"{name+marker:<25} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} {m['cv_f1_macro']:>10.4f}")
    logger.info("="*60)
    logger.info("\n✅ Training complete!")

    return best_model, label_encoder, feature_names, class_names, all_metrics


if __name__ == "__main__":
    # First generate data if not present
    if not DATA_PATH.exists():
        logger.info("Dataset not found. Generating...")
        import sys
        sys.path.insert(0, ".")
        from backend.data.generate_dataset import generate_dataset, save_disease_info
        import json
        df = generate_dataset(150)
        df.to_csv(DATA_PATH, index=False)

    train_all_models()