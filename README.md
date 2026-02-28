# 🫀 VitalSense

> **Early disease prediction powered by wearable data and personalized machine learning.**

VitalSense is a full-stack health monitoring web app that predicts disease risk in real time by automatically streaming vitals from your wearable device (Google Fit / Fitbit). You provide only your **age and gender** — the rest is handled automatically.

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-backend-green)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)

---

## ✨ Features

- **Minimal onboarding** — only age and gender required from the user
- **Automatic vitals collection** — heart rate, HRV, SpO2, sleep, steps, blood pressure, glucose, and more via Google Fit & Fitbit OAuth2
- **Real-time risk prediction** using a trained XGBoost model with 21 engineered features
- **Live dashboard** with WebSocket-powered vitals streaming, risk gauge, and trend charts
- **Voice biomarker analysis** — pitch, energy, stress, and fatigue scoring
- **Adaptive retraining** — the model automatically retrains on accumulated real-world data and hot-reloads without a server restart
- **Alerts feed** for threshold breaches and anomalies

---

## 🏗️ Architecture

```
User Browser (frontend/index.html)
│  ├── Onboarding (age + gender)
│  ├── Real-time vitals dashboard (WebSocket)
│  ├── Risk gauge + trend charts
│  ├── Voice biomarker recorder
│  └── Alerts feed
│
└──► FastAPI Backend (:8000)
        ├── REST API (onboard, predict, history, retrain)
        ├── WebSocket /ws/{user_id}  ← live vitals push
        ├── Wearable Sync (Fitbit webhook + Google Fit OAuth2 poll)
        ├── ML Predictor (XGBoost, 21 features, heuristic fallback)
        └── Retrainer (periodic, accuracy-gated, hot-reload)
              └── File Storage (data/)
                    ├── raw/users.json, vitals.jsonl, voice.jsonl
                    ├── processed/predictions.jsonl
                    ├── models/*.pkl
                    └── logs/retraining_log.jsonl
```

---

## 📁 Project Structure

```
VitalSense/
├── run.py                          # Start the server
├── requirements.txt
├── .env.example                    # Environment variable template
├── frontend/
│   └── index.html                  # Single-page dashboard app
├── backend/
│   ├── main.py                     # FastAPI app + all routes
│   ├── models.py                   # Pydantic schemas
│   ├── database.py                 # File-backed data layer
│   ├── predictor.py                # XGBoost inference engine
│   ├── retrainer.py                # Periodic retraining pipeline
│   ├── wearable_sync.py            # Fitbit + Google Fit OAuth2
│   └── ws_manager.py               # WebSocket manager
├── data/
│   ├── vitalsense_unified_dataset.csv  # Base training data (2,121 samples)
│   ├── raw/                        # Live user & wearable data
│   ├── processed/                  # Prediction history
│   ├── models/                     # Trained model files
│   └── logs/                       # Retraining audit trail
└── scripts/
    ├── retrain_scheduler.py        # Cron-style retraining runner
    └── seed_demo.py                # Seed realistic test data
```

---


## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/kamlesh-4407/Vital-Sense.git
cd Vital-Sense
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
### 3. Start the server

```bash
python run.py
```

Open your browser at **http://localhost:8000**

---

## 🔌 Wearable Integration

### Google Fit & Fitbit — OAuth2 Setup

**Step 1: Register your app**
- Fitbit: [dev.fitbit.com](https://dev.fitbit.com) → Create App
- Google: [console.cloud.google.com](https://console.cloud.google.com) → Enable Fitness API

**Step 3:** Connect your device from the dashboard → OAuth redirect → tokens stored → auto-sync begins.

---

### Supported Metrics

| Metric | Fitbit | Google Fit |
|---|---|---|
| Heart Rate (live) | ✅ | ✅ |
| Resting Heart Rate | ✅ | ✅ |
| HRV | ✅ | — |
| Blood Pressure | — | ✅ |
| Steps | ✅ | ✅ |
| Sleep Hours + Score | ✅ | ✅ |
| Calories | ✅ | ✅ |
| SpO2 | ✅ | — |
| Blood Glucose | — | ✅ |
| Stress Score | ✅ | — |


Voice features (`pitch`, `energy`, `stress_score`, `fatigue_score`) are added when a voice check is submitted.
---
## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, CSS3, JavaScript (SPA) |
| Backend | FastAPI, Python |
| ML Model | XGBoost |
| Data Schemas | Pydantic |
| Wearable APIs | Fitbit OAuth2, Google Fit REST API |
| Real-time | WebSockets |
| Storage | JSON / JSONL flat files |

---
## 👥 Meet the Team

VitalSense was built by **Scrappy Studio** — a team of passionate developers dedicated to making early health detection accessible to everyone.

| Name | GitHub |
|---|---|
| Kamlesh Y | https://github.com/kamlesh-4407 |
| Kishore B | https://github.com/kishore3106 |
| Ramya G | — |
| Mahashri D | — |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.
