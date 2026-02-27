"""
VitalSense — ML Predictor
==========================
Loads the XGBoost model + encoders and runs inference.
Builds the 21-feature vector from user profile + wearable vitals + voice.
"""

import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger("vitalsense.predictor")

MODEL_DIR   = Path(__file__).parent.parent / "data" / "models"
BASE_DIR    = Path(__file__).parent.parent

# Try data/models first (retrained), fall back to project root (original)
def _find_pkl(name):
    candidates = [
        MODEL_DIR / name,
        BASE_DIR / name,
        BASE_DIR.parent / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


class VitalSensePredictor:
    FEATURE_ORDER = [
        'age', 'gender', 'systolic_bp', 'cholesterol', 'max_heart_rate',
        'blood_sugar', 'risk_score_raw', 'oldpeak', 'ca',
        'lifestyle_score', 'lab_risk', 'history_risk', 'symptom_risk',
        'med_count', 'total_risk_score',
        'pitch', 'energy', 'stress_score', 'fatigue_score',
        'stress_level_enc', 'fatigue_level_enc'
    ]

    RECOMMENDATIONS = {
        "High": [
            "🏥 Consult a cardiologist or physician immediately.",
            "💊 Review medications — statins or antihypertensives may be indicated.",
            "📉 Prioritize reducing blood pressure and cholesterol.",
            "🚫 Avoid strenuous physical activity until cleared by a doctor.",
        ],
        "Moderate": [
            "🩺 Schedule a health checkup within 1–2 months.",
            "🥗 Adopt a heart-healthy diet: reduce sodium and saturated fats.",
            "🚶 Aim for 150+ minutes of moderate exercise per week.",
            "😴 Ensure 7–9 hours of quality sleep nightly.",
        ],
        "Low": [
            "✅ Current indicators are favorable — maintain healthy habits.",
            "📅 Continue annual health screenings.",
            "🏃 Sustain regular physical activity and balanced nutrition.",
            "💧 Stay hydrated and manage daily stress levels.",
        ],
    }

    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.model_version = "original"
        self.loaded_at = None
        self._load()

    def _load(self):
        try:
            model_path = _find_pkl("vitalsense_xgboost_model.pkl")
            enc_path   = _find_pkl("vitalsense_label_encoder.pkl")
            feat_path  = _find_pkl("vitalsense_feature_names.pkl")

            if not all([model_path, enc_path, feat_path]):
                logger.warning("Model files not found — using heuristic fallback")
                return

            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                with open(enc_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
                with open(feat_path, "rb") as f:
                    self.feature_names = pickle.load(f)

            self.loaded_at = datetime.utcnow().isoformat()
            logger.info(f"✅ Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")

    def reload(self):
        """Hot-reload after retraining."""
        self._load()
        logger.info("Model reloaded after retraining")

    def build_feature_vector(self, user: dict, vitals: dict) -> dict:
        """
        Combine user profile (age, gender) + wearable vitals + voice
        into the 21-feature vector the model expects.

        Derived features computed here:
          - risk_score_raw       from BP + cholesterol + HR
          - lifestyle_score      from steps + sleep
          - lab_risk             from cholesterol + blood_sugar
          - history_risk         from user medical history (default 1)
          - symptom_risk         from HR anomalies + stress
          - med_count            default 1
          - total_risk_score     weighted composite
        """
        age    = user.get("age", 50)
        gender = user.get("gender", 1)

        sbp        = vitals.get("systolic_bp", 120)
        chol       = vitals.get("cholesterol", 200)
        max_hr     = vitals.get("max_heart_rate") or vitals.get("heart_rate", 75)
        bs         = 1 if vitals.get("blood_sugar", 0) and vitals.get("blood_sugar", 0) > 120 else 0
        oldpeak    = vitals.get("oldpeak", 0.0)
        ca         = vitals.get("ca", 0)
        sleep_hrs  = vitals.get("sleep_hours", 7)
        steps      = vitals.get("steps", 5000)
        hrv        = vitals.get("hrv", 50)
        spo2       = vitals.get("spo2", 98)

        # Derived composite scores
        risk_score_raw = (
            max(0, sbp - 120) * 0.3 +
            max(0, chol - 200) * 0.05 +
            max(0, 180 - max_hr) * 0.1 +
            oldpeak * 2.0
        )

        lifestyle_score = min(3, int(
            (1 if steps >= 7500 else 0) +
            (1 if 7 <= sleep_hrs <= 9 else 0) +
            (1 if hrv >= 50 else 0)
        ))

        lab_risk = min(4, int(
            (1 if chol > 200 else 0) +
            (1 if chol > 240 else 0) +
            (1 if bs == 1 else 0) +
            (1 if spo2 < 95 else 0)
        ))

        history_risk = vitals.get("history_risk", 1)
        med_count    = vitals.get("med_count", 1)

        # Symptom risk from HR + stress device reading
        hr_anom = max(0, vitals.get("heart_rate", 75) - 100) / 20
        dev_stress = vitals.get("stress_reading", 25) / 100
        symptom_risk = min(6, round(hr_anom * 3 + dev_stress * 3, 1))

        total_risk_score = (
            risk_score_raw * 0.35 +
            (lab_risk / 4) * 6 +
            (lifestyle_score / 3) * 4 +
            symptom_risk * 0.8 +
            history_risk * 0.5
        )

        return {
            "age": age,
            "gender": gender,
            "systolic_bp": sbp,
            "cholesterol": chol,
            "max_heart_rate": max_hr,
            "blood_sugar": bs,
            "risk_score_raw": round(risk_score_raw, 3),
            "oldpeak": oldpeak,
            "ca": ca,
            "lifestyle_score": lifestyle_score,
            "lab_risk": lab_risk,
            "history_risk": history_risk,
            "symptom_risk": symptom_risk,
            "med_count": med_count,
            "total_risk_score": round(total_risk_score, 3),
            "pitch": vitals.get("pitch", 0),
            "energy": vitals.get("energy", 0),
            "stress_score": vitals.get("stress_score", 0),
            "fatigue_score": vitals.get("fatigue_score", 0),
            "stress_level_enc": vitals.get("stress_level_enc", 0),
            "fatigue_level_enc": vitals.get("fatigue_level_enc", 0),
        }

    def predict(self, features: dict) -> dict:
        """Run XGBoost inference. Falls back to heuristic if model not loaded."""
        if self.model and self.label_encoder:
            return self._xgboost_predict(features)
        return self._heuristic_predict(features)

    def _xgboost_predict(self, features: dict) -> dict:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = {f: features.get(f, 0.0) for f in self.FEATURE_ORDER}
            X = pd.DataFrame([row])
            enc_pred = self.model.predict(X)[0]
            proba    = self.model.predict_proba(X)[0]
            label    = self.label_encoder.inverse_transform([enc_pred])[0]
            classes  = list(self.label_encoder.classes_)
            prob_dict = {c: round(float(p), 4) for c, p in zip(classes, proba)}
            confidence = float(max(proba))
            risk_score = self._risk_score_from_proba(prob_dict)
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "risk_score": round(risk_score, 1),
            "probabilities": prob_dict,
            "recommendations": self.RECOMMENDATIONS[label],
            "engine": "xgboost",
        }

    def _heuristic_predict(self, f: dict) -> dict:
        """Calibrated fallback when XGBoost model is unavailable."""
        def norm(v, mn, mx): return max(0, min(1, (v - mn) / (mx - mn + 1e-6)))
        score = (
            norm(f.get("total_risk_score", 0), 0, 25.9) * 18.4 +
            norm(f.get("risk_score_raw", 0), 0, 38.9) * 15.2 +
            norm(f.get("symptom_risk", 0), 0, 6) * 12.1 +
            norm(f.get("cholesterol", 0), 150, 564) * 9.8 +
            (1 - norm(f.get("max_heart_rate", 75), 60, 202)) * 8.3 +
            norm(f.get("systolic_bp", 0), 90, 200) * 7.6 +
            norm(f.get("age", 50), 30, 80) * 6.9 +
            norm(f.get("stress_score", 0), 0, 100) * 5.4 +
            norm(f.get("fatigue_score", 0), 0, 100) * 4.8
        )
        score = min(100, max(0, score))
        if score >= 62:
            label, conf = "High", min(0.97, 0.6 + (score - 62) / 100)
        elif score >= 35:
            label, conf = "Moderate", 0.55
        else:
            label, conf = "Low", min(0.97, 0.6 + (35 - score) / 100)
        prob_dict = {
            "High": conf if label == "High" else round((1 - conf) * 0.3, 3),
            "Moderate": conf if label == "Moderate" else 0.15,
            "Low": conf if label == "Low" else round((1 - conf) * 0.5, 3),
        }
        return {
            "label": label,
            "confidence": round(conf, 4),
            "risk_score": round(score, 1),
            "probabilities": prob_dict,
            "recommendations": self.RECOMMENDATIONS[label],
            "engine": "heuristic",
        }

    @staticmethod
    def _risk_score_from_proba(prob_dict: dict) -> float:
        return (
            prob_dict.get("Low", 0) * 15 +
            prob_dict.get("Moderate", 0) * 50 +
            prob_dict.get("High", 0) * 85
        )

    def get_model_info(self) -> dict:
        return {
            "loaded": self.model is not None,
            "version": self.model_version,
            "loaded_at": self.loaded_at,
            "engine": "xgboost" if self.model else "heuristic",
            "feature_count": len(self.FEATURE_ORDER),
        }


predictor = VitalSensePredictor()
