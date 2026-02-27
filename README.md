# 🫀 VitalSense v2 — Automated Wearable-Powered Disease Prediction

An end-to-end health monitoring platform. The user provides **only age and gender**. All clinical vitals stream automatically from Google Fit / Fitbit. The XGBoost model predicts disease risk in real time and periodically retrains on accumulated data.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                             │
│  frontend/index.html                                            │
│  ├── Onboarding (age + gender only)                             │
│  ├── Real-time vitals dashboard (WebSocket)                     │
│  ├── Risk gauge + trend charts                                  │
│  ├── Voice biomarker recorder                                   │
│  └── Alerts feed                                                │
└────────────────────┬────────────────────────────────────────────┘
                     │ HTTP + WebSocket
┌────────────────────▼────────────────────────────────────────────┐
│                    FastAPI BACKEND  :8000                        │
│  backend/main.py                                                │
│  ├── POST /api/users/onboard                                    │
│  ├── POST /api/wearable/fitbit/webhook   ← Fitbit pushes here  │
│  ├── POST /api/wearable/googlefit/sync  ← OAuth2 pull          │
│  ├── POST /api/voice/submit                                     │
│  ├── GET  /api/predict/{user_id}                                │
│  ├── GET  /api/history/{user_id}                                │
│  ├── POST /api/model/retrain                                    │
│  └── WS   /ws/{user_id}              ← Live vitals push        │
└──────────┬──────────────────────────┬───────────────────────────┘
           │                          │
┌──────────▼──────────┐   ┌───────────▼──────────────────────────┐
│   ML PREDICTOR      │   │   WEARABLE SYNC                       │
│   backend/          │   │   backend/wearable_sync.py            │
│   predictor.py      │   │   ├── FitbitSync (OAuth2 + webhook)  │
│   ├── XGBoost model │   │   └── GoogleFitSync (OAuth2 + poll)  │
│   ├── 21 features   │   └──────────────────────────────────────┘
│   └── Heuristic     │
│       fallback      │
└──────────┬──────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│   RETRAINING PIPELINE   backend/retrainer.py                    │
│   ├── Merge original dataset + new real-world readings          │
│   ├── Retrain XGBoost (200 estimators, stratified split)       │
│   ├── Evaluate new vs current accuracy                          │
│   ├── Promote if accuracy ≥ current - 2%                       │
│   └── Hot-reload predictor without server restart               │
└──────────┬──────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│   FILE STORAGE  data/                                           │
│   ├── raw/users.json            User profiles                   │
│   ├── raw/vitals.jsonl          All wearable readings           │
│   ├── raw/voice.jsonl           Voice biomarker sessions        │
│   ├── processed/predictions.jsonl  Prediction history           │
│   ├── models/*.pkl              Active + backed-up models       │
│   └── logs/retraining_log.jsonl Retrain audit trail             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
vitalsense_v2/
├── run.py                          ← Start server
├── requirements.txt
├── README.md
├── frontend/
│   └── index.html                  ← Full dashboard SPA
├── backend/
│   ├── main.py                     ← FastAPI app + all routes
│   ├── models.py                   ← Pydantic schemas
│   ├── database.py                 ← File-backed data layer
│   ├── predictor.py                ← XGBoost inference engine
│   ├── retrainer.py                ← Periodic retraining pipeline
│   ├── wearable_sync.py            ← Fitbit + Google Fit OAuth
│   └── ws_manager.py               ← WebSocket manager
├── data/
│   ├── vitalsense_unified_dataset.csv  ← Base training data
│   ├── raw/                        ← User + wearable data
│   ├── processed/                  ← Predictions history
│   ├── models/                     ← Trained models
│   └── logs/                       ← Retraining audit
└── scripts/
    ├── retrain_scheduler.py        ← Cron-style retrain runner
    └── seed_demo.py                ← Seed realistic test data
```

---

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Copy model files to project root
```bash
cp /path/to/vitalsense_xgboost_model.pkl .
cp /path/to/vitalsense_feature_names.pkl .
cp /path/to/vitalsense_label_encoder.pkl .
cp /path/to/vitalsense_unified_dataset.csv data/
```

### 3. Run server
```bash
python run.py
# Open http://localhost:8000
```

### 4. (Optional) Seed demo data
```bash
python scripts/seed_demo.py
```

### 5. Open dashboard
Open `http://localhost:8000` or directly open `frontend/index.html` — the dashboard runs in demo mode with simulated vitals even without the backend.

---

## 🔌 Wearable Integration

### Google Fit / Fitbit — OAuth2 Setup

1. **Register your app:**
   - Fitbit: https://dev.fitbit.com → Create app
   - Google: https://console.cloud.google.com → Enable Fitness API

2. **Set environment variables:**
```bash
export FITBIT_CLIENT_ID=your_client_id
export FITBIT_CLIENT_SECRET=your_client_secret
export GOOGLE_CLIENT_ID=your_client_id
export GOOGLE_CLIENT_SECRET=your_client_secret
```

3. **User connects from dashboard → redirect to OAuth → tokens stored → auto-sync begins**

### What data is pulled automatically

| Metric | Fitbit | Google Fit |
|---|---|---|
| Heart Rate (live) | ✅ | ✅ |
| Resting HR | ✅ | ✅ |
| HRV | ✅ | — |
| Blood Pressure | — | ✅ |
| Steps | ✅ | ✅ |
| Sleep hours + score | ✅ | ✅ |
| Calories | ✅ | ✅ |
| SpO2 | ✅ | — |
| Blood glucose | — | ✅ |
| Stress score | ✅ | — |

---

## 🤖 ML Pipeline

### Feature Engineering (auto-computed from wearable data)
The predictor automatically derives all 21 model features:

```python
risk_score_raw  = f(systolic_bp, cholesterol, max_heart_rate, oldpeak)
lifestyle_score = f(steps, sleep_hours, hrv)
lab_risk        = f(cholesterol, blood_sugar, spo2)
symptom_risk    = f(heart_rate, device_stress_reading)
total_risk_score= weighted_composite(all_above)
```

Voice features (`pitch`, `energy`, `stress_score`, `fatigue_score`) are added when a voice check is performed.

### Retraining
```bash
# Run once manually
python scripts/retrain_scheduler.py --once

# Run every 12 hours
python scripts/retrain_scheduler.py --interval 12

# Via API (requires secret)
curl -X POST "http://localhost:8000/api/model/retrain?secret=vitalsense-retrain-2024"
```

**Retraining logic:**
- Merges original 2,121-sample dataset with real-world accumulated readings
- Trains new XGBoost (200 estimators, 15% held-out eval)
- Only promotes new model if accuracy ≥ current − 2%
- Backs up current model before replacing
- Hot-reloads predictor without server restart

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/users/onboard` | Create user (name, age, gender) |
| GET | `/api/users/{id}` | Get user profile |
| POST | `/api/wearable/fitbit/webhook` | Fitbit push notification |
| POST | `/api/wearable/googlefit/sync` | Google Fit data sync |
| GET | `/api/wearable/connect/{id}/{source}` | Get OAuth URL |
| POST | `/api/wearable/sync/{id}` | Manual sync trigger |
| POST | `/api/voice/submit` | Submit voice features |
| GET | `/api/predict/{id}` | Run prediction |
| GET | `/api/history/{id}` | Prediction history |
| GET | `/api/vitals/stream/{id}` | Vitals history |
| POST | `/api/model/retrain` | Trigger retraining |
| GET | `/api/model/status` | Model info |
| WS | `/ws/{id}` | Real-time vitals + alerts |

---

## ⚠️ Disclaimer
VitalSense is a research and educational tool. It is **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare provider.
