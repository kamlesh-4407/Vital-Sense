"""
VitalSense — Demo Data Seeder
================================
Seeds realistic wearable readings for a test user so the dashboard
shows live data immediately without a real wearable connected.

Usage:
    python scripts/seed_demo.py
"""

import json
import sys
import random
import math
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.database import db

def seed():
    print("🌱 Seeding demo user + 30 days of wearable readings...")

    # Create user
    user = db.create_user(age=42, gender=0, name="Demo Patient")
    uid = user["user_id"]
    print(f"   User: {uid}")

    base = datetime.utcnow() - timedelta(days=30)

    for day in range(30):
        ts = (base + timedelta(days=day)).isoformat()
        t = day / 30  # 0→1 progress

        # Simulate gradual health improvement over 30 days
        hr    = 78 - t*8 + random.gauss(0, 3)
        sbp   = 140 - t*15 + random.gauss(0, 5)
        chol  = 240 - t*20 + random.gauss(0, 8)
        hrv   = 35 + t*15 + random.gauss(0, 4)
        steps = int(4000 + t*4000 + random.gauss(0, 500))
        sleep = 6.5 + t*1.2 + random.gauss(0, 0.3)
        spo2  = 97 + random.gauss(0, 0.3)
        stress= 55 - t*25 + random.gauss(0, 5)

        reading = {
            "user_id": uid,
            "source": "demo",
            "timestamp": ts,
            "heart_rate": round(max(55, hr), 1),
            "resting_heart_rate": round(max(50, hr - 10), 1),
            "systolic_bp": round(max(90, sbp), 1),
            "cholesterol": round(max(150, chol), 1),
            "max_heart_rate": round(min(200, hr + 40), 1),
            "hrv": round(max(20, hrv), 1),
            "steps": max(500, steps),
            "sleep_hours": round(max(4, min(10, sleep)), 2),
            "sleep_score": round(max(40, min(100, 60 + t*30)), 0),
            "spo2": round(max(94, min(100, spo2)), 1),
            "calories": round(1600 + steps * 0.05, 0),
            "stress_reading": round(max(10, min(80, stress)), 1),
            "blood_sugar": 0,
            "oldpeak": round(max(0, 2 - t*1.5 + random.gauss(0,.2)), 2),
            "ca": 1 if sbp > 130 else 0,
        }
        db.store_wearable_reading(reading)

    # Add some voice readings
    for i in range(5):
        ts = (base + timedelta(days=i*6)).isoformat()
        db.store_voice_reading({
            "user_id": uid, "recorded_at": ts,
            "pitch": round(random.uniform(80, 200), 2),
            "energy": round(random.uniform(0.001, 0.05), 6),
            "stress_score": round(random.uniform(20, 70), 2),
            "fatigue_score": round(random.uniform(25, 65), 2),
            "stress_level_enc": random.choice([0, 1, 2]),
            "fatigue_level_enc": random.choice([0, 1]),
        })

    print(f"✅ Seeded 30 days of vitals + 5 voice readings for user {uid}")
    print(f"\n   Open the dashboard and use user_id: {uid}")
    print(f"   Or just open frontend/index.html — it runs in demo mode automatically\n")

if __name__ == "__main__":
    seed()
