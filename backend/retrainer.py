"""
VitalSense — Model Retraining Pipeline
========================================
Periodically retrains the XGBoost model on accumulated patient data.
Merges original training dataset with new real-world readings.
Evaluates new model before promoting it.
"""

import pickle
import logging
import warnings
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("vitalsense.retrainer")

BASE_DIR    = Path(__file__).parent.parent
DATA_DIR    = BASE_DIR / "data"
MODEL_DIR   = DATA_DIR / "models"
LOG_DIR     = DATA_DIR / "logs"

ORIGINAL_DATASET = BASE_DIR.parent / "data" / "vitalsense_unified_dataset.csv"
FALLBACK_DATASET = BASE_DIR / "data" / "processed" / "vitalsense_unified_dataset.csv"

FEATURE_COLS = [
    'age', 'gender', 'systolic_bp', 'cholesterol', 'max_heart_rate',
    'blood_sugar', 'risk_score_raw', 'oldpeak', 'ca',
    'lifestyle_score', 'lab_risk', 'history_risk', 'symptom_risk',
    'med_count', 'total_risk_score',
    'pitch', 'energy', 'stress_score', 'fatigue_score',
    'stress_level_enc', 'fatigue_level_enc'
]
LABEL_COL = 'risk_label'

MIN_NEW_SAMPLES = 10  # minimum new samples before retraining kicks in


async def schedule_retraining():
    """Background task: check if enough new data, then retrain."""
    logger.info("🔄 Retraining check started...")
    try:
        from .database import db
        new_data = db.get_all_vitals_for_training()
        if len(new_data) < MIN_NEW_SAMPLES:
            logger.info(f"Only {len(new_data)} new samples — skipping retrain (need {MIN_NEW_SAMPLES})")
            _log_event("skipped", {"reason": "insufficient_data", "samples": len(new_data)})
            return

        result = retrain(new_data)
        _log_event("completed", result)
        logger.info(f"✅ Retraining complete: {result}")

    except Exception as e:
        logger.error(f"Retraining failed: {e}")
        _log_event("failed", {"error": str(e)})


def retrain(new_records: list) -> dict:
    """
    Full retrain pipeline:
      1. Load original dataset
      2. Append new real-world records
      3. Preprocess + encode
      4. Train new XGBoost model
      5. Evaluate on held-out split
      6. Promote if better than current
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load base dataset ──────────────────────────────────────────────────
    base_df = _load_base_dataset()
    logger.info(f"Base dataset: {len(base_df)} rows")

    # ── 2. Build new-data DataFrame ───────────────────────────────────────────
    new_df = pd.DataFrame(new_records)
    new_df = new_df.rename(columns={"risk_label": LABEL_COL})
    # Keep only columns we have
    available = [c for c in FEATURE_COLS + [LABEL_COL] if c in new_df.columns]
    new_df = new_df[available]
    logger.info(f"New records: {len(new_df)} rows")

    # ── 3. Combine ────────────────────────────────────────────────────────────
    combined = pd.concat([base_df, new_df], ignore_index=True)
    combined = combined.dropna(subset=[LABEL_COL])
    combined[FEATURE_COLS] = combined[FEATURE_COLS].fillna(0)
    logger.info(f"Combined dataset: {len(combined)} rows | Label dist: {combined[LABEL_COL].value_counts().to_dict()}")

    # ── 4. Encode labels ──────────────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        import xgboost as xgb

        le = LabelEncoder()
        y = le.fit_transform(combined[LABEL_COL])
        X = combined[FEATURE_COLS].astype(float)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

        # ── 5. Train ──────────────────────────────────────────────────────────
        new_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        new_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # ── 6. Evaluate ───────────────────────────────────────────────────────
        y_pred = new_model.predict(X_test)
        new_acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        logger.info(f"New model accuracy: {new_acc:.4f}")

        # Compare with current model
        current_acc = _evaluate_current_model(X_test, y_test)
        logger.info(f"Current model accuracy: {current_acc:.4f}")

        should_promote = new_acc >= current_acc - 0.02  # allow 2% slack

        if should_promote:
            _backup_current_model()
            _save_model(new_model, le, FEATURE_COLS)
            # Hot-reload the predictor
            from .predictor import predictor
            predictor.reload()
            status = "promoted"
            logger.info("✅ New model PROMOTED and loaded")
        else:
            status = "rejected"
            logger.warning(f"⚠️ New model rejected (acc {new_acc:.3f} < current {current_acc:.3f})")

    return {
        "status": status,
        "new_accuracy": round(new_acc, 4),
        "current_accuracy": round(current_acc, 4),
        "total_samples": len(combined),
        "new_samples": len(new_df),
        "timestamp": datetime.utcnow().isoformat(),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 3),
                "recall": round(report[cls]["recall"], 3),
                "f1": round(report[cls]["f1-score"], 3),
            }
            for cls in le.classes_ if cls in report
        }
    }


def _load_base_dataset() -> pd.DataFrame:
    """Try multiple paths to find the base training dataset."""
    for path in [ORIGINAL_DATASET, FALLBACK_DATASET,
                 BASE_DIR / "vitalsense_unified_dataset.csv"]:
        if path.exists():
            df = pd.read_csv(path)
            available = [c for c in FEATURE_COLS + [LABEL_COL] if c in df.columns]
            return df[available]
    logger.warning("Base dataset not found — training on new data only")
    return pd.DataFrame(columns=FEATURE_COLS + [LABEL_COL])


def _evaluate_current_model(X_test, y_test) -> float:
    """Score the currently deployed model on the test split."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sklearn.metrics import accuracy_score
            model_path = _find_current_model()
            if not model_path or not model_path.exists():
                return 0.0
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            y_pred = model.predict(X_test)
            return accuracy_score(y_test, y_pred)
    except Exception:
        return 0.0


def _find_current_model():
    for p in [MODEL_DIR / "vitalsense_xgboost_model.pkl",
              BASE_DIR / "vitalsense_xgboost_model.pkl",
              BASE_DIR.parent / "vitalsense_xgboost_model.pkl"]:
        if p.exists():
            return p
    return None


def _backup_current_model():
    current = _find_current_model()
    if current and current.exists():
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup = MODEL_DIR / f"model_backup_{ts}.pkl"
        shutil.copy2(current, backup)
        logger.info(f"Current model backed up to {backup}")


def _save_model(model, le, feature_names):
    """Persist the newly trained model + encoders."""
    with open(MODEL_DIR / "vitalsense_xgboost_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(MODEL_DIR / "vitalsense_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(MODEL_DIR / "vitalsense_feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    logger.info(f"New model saved to {MODEL_DIR}")


def _log_event(event_type: str, data: dict):
    """Append retraining event to log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "retraining_log.jsonl"
    import json
    entry = {"event": event_type, "data": data, "ts": datetime.utcnow().isoformat()}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
