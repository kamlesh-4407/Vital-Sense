"""
VitalSense Backend — FastAPI Server
====================================
Handles:
  - User onboarding (age + gender only)
  - Wearable data ingestion (Google Fit / Fitbit webhooks)
  - Voice biomarker submission
  - ML prediction endpoint
  - Periodic model retraining trigger
  - WebSocket for real-time vitals feed
  - Risk alert broadcasting
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import (
    UserOnboard, VoicePayload, WearableWebhook,
    FitbitPayload, GoogleFitPayload, PredictionResponse
)
from .database import db
from .predictor import predictor
from .retrainer import schedule_retraining
from .wearable_sync import fitbit_sync, googlefit_sync
from .ws_manager import ConnectionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vitalsense")

app = FastAPI(title="VitalSense API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = ConnectionManager()

# ── Mount static frontend ─────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def serve_dashboard():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


# ── ONBOARDING ────────────────────────────────────────────────────────────────
@app.post("/api/users/onboard")
async def onboard_user(payload: UserOnboard):
    """Register a new user with only age + gender. All other data comes from wearables."""
    user = db.create_user(payload.age, payload.gender, payload.name)
    logger.info(f"New user onboarded: {user['user_id']}")
    return {"user_id": user["user_id"], "message": "Onboarding complete. Connect your wearable to begin."}


@app.get("/api/users/{user_id}")
async def get_user(user_id: str):
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user


# ── WEARABLE INGESTION ────────────────────────────────────────────────────────
@app.post("/api/wearable/fitbit/webhook")
async def fitbit_webhook(payload: FitbitPayload, background_tasks: BackgroundTasks):
    """Fitbit pushes data here on subscription updates."""
    for notification in payload.updates:
        raw = await fitbit_sync.fetch_and_normalize(notification)
        if raw:
            db.store_wearable_reading(raw)
            background_tasks.add_task(run_prediction_pipeline, raw["user_id"])
    return {"received": len(payload.updates)}


@app.post("/api/wearable/googlefit/sync")
async def googlefit_sync_endpoint(payload: GoogleFitPayload, background_tasks: BackgroundTasks):
    """Google Fit data sync — called on schedule or OAuth callback."""
    raw = googlefit_sync.normalize(payload)
    db.store_wearable_reading(raw)
    background_tasks.add_task(run_prediction_pipeline, raw["user_id"])
    return {"status": "synced", "readings": len(raw.get("data_points", []))}


@app.post("/api/wearable/manual")
async def manual_wearable_input(data: dict, background_tasks: BackgroundTasks):
    """Fallback: manually push wearable readings (for testing / demo)."""
    db.store_wearable_reading(data)
    background_tasks.add_task(run_prediction_pipeline, data["user_id"])
    return {"status": "ok"}


# ── VOICE ─────────────────────────────────────────────────────────────────────
@app.post("/api/voice/submit")
async def submit_voice(payload: VoicePayload, background_tasks: BackgroundTasks):
    """Accept extracted voice features from the frontend voice recorder."""
    db.store_voice_reading(payload.dict())
    background_tasks.add_task(run_prediction_pipeline, payload.user_id)
    return {"status": "voice features stored"}


# ── PREDICTION ────────────────────────────────────────────────────────────────
@app.get("/api/predict/{user_id}")
async def predict(user_id: str) -> PredictionResponse:
    """Run prediction for a user using their latest wearable + voice data."""
    result = await run_prediction_pipeline(user_id, broadcast=False)
    if not result:
        raise HTTPException(400, "Insufficient data for prediction")
    return result


async def run_prediction_pipeline(user_id: str, broadcast: bool = True):
    """Core pipeline: fetch latest data → predict → store → broadcast."""
    user = db.get_user(user_id)
    if not user:
        return None

    latest = db.get_latest_vitals(user_id)
    if not latest:
        return None

    features = predictor.build_feature_vector(user, latest)
    result = predictor.predict(features)
    result["user_id"] = user_id
    result["timestamp"] = datetime.utcnow().isoformat()
    result["vitals"] = latest

    db.store_prediction(result)

    if broadcast:
        await manager.broadcast(user_id, {"type": "prediction", "data": result})
        if result["label"] == "High":
            await manager.broadcast(user_id, {
                "type": "alert",
                "level": "high",
                "message": f"⚠️ High risk detected! Confidence: {result['confidence']*100:.1f}%",
                "timestamp": result["timestamp"]
            })

    return result


# ── HISTORY & TRENDS ─────────────────────────────────────────────────────────
@app.get("/api/history/{user_id}")
async def get_history(user_id: str, limit: int = 50):
    return db.get_prediction_history(user_id, limit)


@app.get("/api/vitals/stream/{user_id}")
async def get_vitals_stream(user_id: str, limit: int = 20):
    return db.get_vitals_history(user_id, limit)


# ── MODEL RETRAINING ─────────────────────────────────────────────────────────
@app.post("/api/model/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks, secret: str = ""):
    """Manually trigger model retraining. Protected endpoint."""
    if secret != "vitalsense-retrain-2024":
        raise HTTPException(403, "Invalid secret")
    background_tasks.add_task(schedule_retraining)
    return {"status": "Retraining scheduled in background"}


@app.get("/api/model/status")
async def model_status():
    return predictor.get_model_info()


# ── WEBSOCKET ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        # Push current state immediately on connect
        latest = db.get_latest_vitals(user_id)
        if latest:
            await websocket.send_json({"type": "vitals", "data": latest})

        # Keep alive + handle incoming messages
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                data = json.loads(msg)
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
