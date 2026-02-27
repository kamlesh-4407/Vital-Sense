"""
VitalSense — Pydantic Data Models
All request/response schemas for the API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ── USER ──────────────────────────────────────────────────────────────────────
class UserOnboard(BaseModel):
    name: str = Field(..., example="Priya Sharma")
    age: int = Field(..., ge=0, le=120, example=34)
    gender: int = Field(..., ge=0, le=1, description="0=Female, 1=Male", example=0)


# ── VOICE ─────────────────────────────────────────────────────────────────────
class VoicePayload(BaseModel):
    user_id: str
    pitch: float = 0.0
    energy: float = 0.0
    stress_score: float = 0.0
    fatigue_score: float = 0.0
    stress_level_enc: int = 0       # 0=Low, 1=Medium, 2=High
    fatigue_level_enc: int = 0
    recorded_at: Optional[str] = None


# ── WEARABLE — FITBIT ─────────────────────────────────────────────────────────
class FitbitNotification(BaseModel):
    collectionType: str
    date: str
    ownerId: str
    ownerType: str
    subscriptionId: str

class FitbitPayload(BaseModel):
    updates: List[FitbitNotification]


# ── WEARABLE — GOOGLE FIT ─────────────────────────────────────────────────────
class DataPoint(BaseModel):
    dataTypeName: str
    value: Any
    startTimeNanos: Optional[str] = None
    endTimeNanos: Optional[str] = None

class GoogleFitPayload(BaseModel):
    user_id: str
    data_points: List[DataPoint]


# ── WEARABLE — GENERIC / NORMALIZED ──────────────────────────────────────────
class WearableWebhook(BaseModel):
    """Normalized wearable reading after source-specific parsing."""
    user_id: str
    source: str                         # "fitbit" | "googlefit" | "manual"
    timestamp: str

    # Vitals (auto-populated from wearable)
    heart_rate: Optional[float] = None           # bpm
    resting_heart_rate: Optional[float] = None   # bpm
    systolic_bp: Optional[float] = None          # mmHg
    diastolic_bp: Optional[float] = None
    spo2: Optional[float] = None                 # %
    steps: Optional[int] = None
    calories: Optional[float] = None
    sleep_hours: Optional[float] = None
    sleep_score: Optional[float] = None          # 0-100
    hrv: Optional[float] = None                  # ms
    stress_reading: Optional[float] = None       # device stress (Fitbit)
    cholesterol: Optional[float] = None
    blood_sugar: Optional[float] = None
    weight_kg: Optional[float] = None

    # Derived
    max_heart_rate: Optional[float] = None
    lifestyle_score: Optional[float] = None


# ── PREDICTION ────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    user_id: str
    label: str                    # "Low" | "Moderate" | "High"
    confidence: float
    risk_score: float             # 0–100
    probabilities: Dict[str, float]
    timestamp: str
    vitals: Optional[Dict] = None
    recommendations: Optional[List[str]] = None
