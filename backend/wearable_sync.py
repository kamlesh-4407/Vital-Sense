"""
VitalSense — Wearable Sync Modules
=====================================
Handles OAuth flows and data normalization for:
  - Fitbit Web API
  - Google Fit REST API

In production, store OAuth tokens in a secure DB (not flat files).
"""

import os
import json
import logging
import httpx
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vitalsense.wearable")

TOKENS_FILE = Path(__file__).parent.parent / "data" / "raw" / "oauth_tokens.json"


def _load_tokens() -> dict:
    if TOKENS_FILE.exists():
        try:
            return json.loads(TOKENS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_tokens(tokens: dict):
    TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKENS_FILE.write_text(json.dumps(tokens, indent=2))


# ── FITBIT ────────────────────────────────────────────────────────────────────
class FitbitSync:
    """
    Fitbit Web API v1 integration.

    OAuth2 Setup:
      1. Register app at https://dev.fitbit.com
      2. Set FITBIT_CLIENT_ID, FITBIT_CLIENT_SECRET env vars
      3. Redirect URI: https://your-domain/api/wearable/fitbit/callback
      4. Required scopes: heartrate activity sleep profile

    Webhook:
      - Subscribe via POST /1/user/-/apiSubscriptions/vitalsense.json
      - Fitbit pushes to /api/wearable/fitbit/webhook on new data
    """

    BASE_URL    = "https://api.fitbit.com/1"
    AUTH_URL    = "https://www.fitbit.com/oauth2/authorize"
    TOKEN_URL   = "https://api.fitbit.com/oauth2/token"

    def __init__(self):
        self.client_id     = os.getenv("FITBIT_CLIENT_ID", "YOUR_CLIENT_ID")
        self.client_secret = os.getenv("FITBIT_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
        self.tokens        = _load_tokens().get("fitbit", {})

    def get_auth_url(self, user_id: str) -> str:
        """Step 1: Redirect user to Fitbit authorization page."""
        scopes = "heartrate activity sleep profile cardio_fitness"
        return (
            f"{self.AUTH_URL}"
            f"?response_type=code"
            f"&client_id={self.client_id}"
            f"&scope={scopes.replace(' ', '%20')}"
            f"&state={user_id}"
            f"&redirect_uri=http://localhost:8000/api/wearable/fitbit/callback"
        )

    async def exchange_code(self, code: str, user_id: str) -> dict:
        """Step 2: Exchange auth code for access + refresh tokens."""
        import base64
        creds = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.TOKEN_URL,
                headers={"Authorization": f"Basic {creds}",
                         "Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "authorization_code", "code": code,
                      "redirect_uri": "http://localhost:8000/api/wearable/fitbit/callback"},
            )
        tokens = resp.json()
        self.tokens[user_id] = tokens
        all_tokens = _load_tokens()
        all_tokens.setdefault("fitbit", {})[user_id] = tokens
        _save_tokens(all_tokens)
        return tokens

    async def refresh_token(self, user_id: str) -> Optional[str]:
        """Refresh expired access token."""
        import base64
        tok = self.tokens.get(user_id, {})
        if not tok.get("refresh_token"):
            return None
        creds = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.TOKEN_URL,
                headers={"Authorization": f"Basic {creds}",
                         "Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "refresh_token", "refresh_token": tok["refresh_token"]},
            )
        new_tok = resp.json()
        self.tokens[user_id] = new_tok
        all_tokens = _load_tokens()
        all_tokens.setdefault("fitbit", {})[user_id] = new_tok
        _save_tokens(all_tokens)
        return new_tok.get("access_token")

    async def fetch_today(self, user_id: str) -> Optional[dict]:
        """Pull today's summary from Fitbit API and normalize it."""
        token = self.tokens.get(user_id, {}).get("access_token")
        if not token:
            logger.warning(f"No Fitbit token for {user_id}")
            return None

        today = date.today().isoformat()
        headers = {"Authorization": f"Bearer {token}"}

        async with httpx.AsyncClient() as client:
            hr_resp    = await client.get(f"{self.BASE_URL}/user/-/activities/heart/date/today/1d.json", headers=headers)
            act_resp   = await client.get(f"{self.BASE_URL}/user/-/activities/date/{today}.json", headers=headers)
            sleep_resp = await client.get(f"{self.BASE_URL}/user/-/sleep/date/{today}.json", headers=headers)
            hrv_resp   = await client.get(f"{self.BASE_URL}/user/-/hrv/date/{today}.json", headers=headers)

        return self._normalize(user_id, hr_resp.json(), act_resp.json(), sleep_resp.json(), hrv_resp.json())

    async def fetch_and_normalize(self, notification: dict) -> Optional[dict]:
        """Called from webhook — fetch full data for the notification owner."""
        user_id = notification.get("ownerId")
        if not user_id:
            return None
        return await self.fetch_today(user_id)

    def _normalize(self, user_id, hr_data, act_data, sleep_data, hrv_data) -> dict:
        """Convert Fitbit API responses to VitalSense standard format."""
        out = {
            "user_id": user_id,
            "source": "fitbit",
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Heart rate
        try:
            hr_summary = hr_data["activities-heart"][0]["value"]
            out["resting_heart_rate"] = hr_summary.get("restingHeartRate")
            zones = hr_summary.get("heartRateZones", [])
            peak = next((z for z in zones if z["name"] == "Peak"), None)
            out["max_heart_rate"] = peak["max"] if peak else None
            # Latest intraday HR (if available)
            intraday = hr_data.get("activities-heart-intraday", {}).get("dataset", [])
            if intraday:
                out["heart_rate"] = intraday[-1]["value"]
        except Exception:
            pass

        # Activity
        try:
            summary = act_data.get("summary", {})
            out["steps"]    = summary.get("steps", 0)
            out["calories"] = summary.get("caloriesOut")
            out["stress_reading"] = summary.get("activeScore")  # proxy for stress
        except Exception:
            pass

        # Sleep
        try:
            sleep_summary = sleep_data.get("summary", {})
            out["sleep_hours"] = round(sleep_summary.get("totalMinutesAsleep", 0) / 60, 2)
            out["sleep_score"] = sleep_data.get("sleep", [{}])[0].get("efficiency")
        except Exception:
            pass

        # HRV
        try:
            hrv_list = hrv_data.get("hrv", [])
            if hrv_list:
                out["hrv"] = hrv_list[-1]["value"]["dailyRmssd"]
        except Exception:
            pass

        return out


# ── GOOGLE FIT ────────────────────────────────────────────────────────────────
class GoogleFitSync:
    """
    Google Fit REST API integration.

    OAuth2 Setup:
      1. Enable Fitness API in Google Cloud Console
      2. Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET env vars
      3. Required scopes:
         - https://www.googleapis.com/auth/fitness.heart_rate.read
         - https://www.googleapis.com/auth/fitness.activity.read
         - https://www.googleapis.com/auth/fitness.sleep.read
         - https://www.googleapis.com/auth/fitness.blood_pressure.read
         - https://www.googleapis.com/auth/fitness.blood_glucose.read

    Data Pull:
      - Google Fit doesn't push webhooks — poll on schedule (e.g. every 15 min)
      - Or trigger sync via POST /api/wearable/googlefit/sync from mobile app
    """

    BASE_URL = "https://www.googleapis.com/fitness/v1/users/me"
    AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"

    # Data source IDs
    SOURCES = {
        "heart_rate":   "derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm",
        "steps":        "derived:com.google.step_count.delta:com.google.android.gms:merge_step_deltas",
        "bp":           "derived:com.google.blood_pressure:com.google.android.gms:merged",
        "blood_glucose":"derived:com.google.blood_glucose:com.google.android.gms:merged",
        "sleep":        "derived:com.google.sleep.segment:com.google.android.gms:merged",
        "calories":     "derived:com.google.calories.expended:com.google.android.gms:merge_calories_expended",
    }

    def __init__(self):
        self.client_id     = os.getenv("GOOGLE_CLIENT_ID", "YOUR_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "YOUR_CLIENT_SECRET")
        self.tokens        = _load_tokens().get("googlefit", {})

    def get_auth_url(self, user_id: str) -> str:
        scopes = " ".join([
            "https://www.googleapis.com/auth/fitness.heart_rate.read",
            "https://www.googleapis.com/auth/fitness.activity.read",
            "https://www.googleapis.com/auth/fitness.sleep.read",
            "https://www.googleapis.com/auth/fitness.blood_pressure.read",
            "https://www.googleapis.com/auth/fitness.blood_glucose.read",
        ])
        return (
            f"{self.AUTH_URL}"
            f"?client_id={self.client_id}"
            f"&redirect_uri=http://localhost:8000/api/wearable/googlefit/callback"
            f"&response_type=code"
            f"&scope={scopes.replace(' ', '%20')}"
            f"&access_type=offline"
            f"&state={user_id}"
        )

    async def exchange_code(self, code: str, user_id: str) -> dict:
        async with httpx.AsyncClient() as client:
            resp = await client.post(self.TOKEN_URL, data={
                "code": code,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": "http://localhost:8000/api/wearable/googlefit/callback",
                "grant_type": "authorization_code",
            })
        tokens = resp.json()
        self.tokens[user_id] = tokens
        all_tokens = _load_tokens()
        all_tokens.setdefault("googlefit", {})[user_id] = tokens
        _save_tokens(all_tokens)
        return tokens

    def normalize(self, payload) -> dict:
        """Normalize Google Fit data_points into VitalSense format."""
        out = {
            "user_id": payload.user_id,
            "source": "googlefit",
            "timestamp": datetime.utcnow().isoformat(),
        }

        type_map = {
            "com.google.heart_rate.bpm": "heart_rate",
            "com.google.step_count.delta": "steps",
            "com.google.blood_pressure": "systolic_bp",
            "com.google.blood_glucose.summary": "blood_sugar",
            "com.google.calories.expended": "calories",
        }

        for dp in payload.data_points:
            key = type_map.get(dp.dataTypeName)
            if key and dp.value is not None:
                if key == "steps":
                    out[key] = int(dp.value) if out.get(key) is None else out[key] + int(dp.value)
                elif key == "systolic_bp" and isinstance(dp.value, dict):
                    out["systolic_bp"] = dp.value.get("systolic")
                    out["diastolic_bp"] = dp.value.get("diastolic")
                elif key == "blood_sugar":
                    out["blood_sugar"] = float(dp.value)
                else:
                    out[key] = float(dp.value)

        out["data_points"] = len(payload.data_points)
        return out

    async def fetch_today(self, user_id: str) -> Optional[dict]:
        """Pull last 24h of data from Google Fit API."""
        token = self.tokens.get(user_id, {}).get("access_token")
        if not token:
            logger.warning(f"No Google Fit token for {user_id}")
            return None

        now_ms = int(datetime.utcnow().timestamp() * 1000)
        start_ms = now_ms - 24 * 3600 * 1000  # 24h ago
        headers = {"Authorization": f"Bearer {token}"}
        out = {"user_id": user_id, "source": "googlefit", "timestamp": datetime.utcnow().isoformat()}

        async with httpx.AsyncClient() as client:
            for metric, source_id in self.SOURCES.items():
                url = f"{self.BASE_URL}/dataSources/{source_id}/datasets/{start_ms}000000-{now_ms}000000"
                try:
                    resp = await client.get(url, headers=headers)
                    points = resp.json().get("point", [])
                    if points:
                        last = points[-1]["value"][0]
                        if metric == "heart_rate":
                            out["heart_rate"] = last.get("fpVal")
                        elif metric == "steps":
                            out["steps"] = sum(p["value"][0].get("intVal", 0) for p in points)
                        elif metric == "bp":
                            out["systolic_bp"] = last.get("fpVal")
                        elif metric == "blood_glucose":
                            out["blood_sugar"] = last.get("fpVal")
                        elif metric == "calories":
                            out["calories"] = sum(p["value"][0].get("fpVal", 0) for p in points)
                except Exception as e:
                    logger.debug(f"Google Fit {metric} fetch error: {e}")

        return out


# OAuth callback routes (add to main.py)
OAUTH_ROUTES = '''
# Add these to main.py:

@app.get("/api/wearable/fitbit/callback")
async def fitbit_callback(code: str, state: str):
    tokens = await fitbit_sync.exchange_code(code, state)
    db.update_user(state, {"wearable_connected": True, "wearable_source": "fitbit"})
    return RedirectResponse("/?connected=fitbit")

@app.get("/api/wearable/googlefit/callback")
async def googlefit_callback(code: str, state: str):
    tokens = await googlefit_sync.exchange_code(code, state)
    db.update_user(state, {"wearable_connected": True, "wearable_source": "googlefit"})
    return RedirectResponse("/?connected=googlefit")

@app.get("/api/wearable/connect/{user_id}/{source}")
async def get_connect_url(user_id: str, source: str):
    if source == "fitbit":
        return {"url": fitbit_sync.get_auth_url(user_id)}
    elif source == "googlefit":
        return {"url": googlefit_sync.get_auth_url(user_id)}
    raise HTTPException(400, "Unknown source")

@app.post("/api/wearable/sync/{user_id}")
async def manual_sync(user_id: str, background_tasks: BackgroundTasks):
    user = db.get_user(user_id)
    source = user.get("wearable_source") if user else None
    if source == "fitbit":
        data = await fitbit_sync.fetch_today(user_id)
    elif source == "googlefit":
        data = await googlefit_sync.fetch_today(user_id)
    else:
        raise HTTPException(400, "No wearable connected")
    if data:
        db.store_wearable_reading(data)
        background_tasks.add_task(run_prediction_pipeline, user_id)
    return {"status": "synced"}
'''


fitbit_sync    = FitbitSync()
googlefit_sync = GoogleFitSync()
