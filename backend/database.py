"""
VitalSense — Database Layer
============================
Lightweight file-backed storage using JSON + CSV.
In production, swap with PostgreSQL / TimescaleDB.
"""

import json
import csv
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent / "data"
USERS_FILE   = DATA_DIR / "raw" / "users.json"
VITALS_FILE  = DATA_DIR / "raw" / "vitals.jsonl"
VOICE_FILE   = DATA_DIR / "raw" / "voice.jsonl"
PREDS_FILE   = DATA_DIR / "processed" / "predictions.jsonl"


def _ensure_dirs():
    for p in [DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "models", DATA_DIR / "logs"]:
        p.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _append_jsonl(path: Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


class Database:
    def __init__(self):
        _ensure_dirs()
        self._users: Dict[str, dict] = _load_json(USERS_FILE)

    def _save_users(self):
        USERS_FILE.write_text(json.dumps(self._users, indent=2))

    # ── USERS ────────────────────────────────────────────────────────────────
    def create_user(self, age: int, gender: int, name: str) -> dict:
        uid = str(uuid.uuid4())[:8]
        user = {
            "user_id": uid,
            "name": name,
            "age": age,
            "gender": gender,
            "created_at": datetime.utcnow().isoformat(),
            "wearable_connected": False,
            "wearable_source": None,
        }
        self._users[uid] = user
        self._save_users()
        return user

    def get_user(self, user_id: str) -> Optional[dict]:
        return self._users.get(user_id)

    def update_user(self, user_id: str, fields: dict):
        if user_id in self._users:
            self._users[user_id].update(fields)
            self._save_users()

    # ── WEARABLE VITALS ──────────────────────────────────────────────────────
    def store_wearable_reading(self, data: dict):
        data["stored_at"] = datetime.utcnow().isoformat()
        _append_jsonl(VITALS_FILE, data)

    def get_latest_vitals(self, user_id: str) -> Optional[dict]:
        records = [r for r in _read_jsonl(VITALS_FILE) if r.get("user_id") == user_id]
        if not records:
            return None
        # Merge most recent records to fill all fields
        merged = {}
        for r in reversed(records[-10:]):  # last 10 readings
            for k, v in r.items():
                if k not in merged and v is not None:
                    merged[k] = v
        # Also grab latest voice for the same user
        voice = self.get_latest_voice(user_id)
        if voice:
            merged.update({
                "pitch": voice.get("pitch", 0),
                "energy": voice.get("energy", 0),
                "stress_score": voice.get("stress_score", 0),
                "fatigue_score": voice.get("fatigue_score", 0),
                "stress_level_enc": voice.get("stress_level_enc", 0),
                "fatigue_level_enc": voice.get("fatigue_level_enc", 0),
            })
        return merged

    def get_vitals_history(self, user_id: str, limit: int = 50) -> List[dict]:
        records = [r for r in _read_jsonl(VITALS_FILE) if r.get("user_id") == user_id]
        return records[-limit:]

    # ── VOICE ────────────────────────────────────────────────────────────────
    def store_voice_reading(self, data: dict):
        data["stored_at"] = datetime.utcnow().isoformat()
        _append_jsonl(VOICE_FILE, data)

    def get_latest_voice(self, user_id: str) -> Optional[dict]:
        records = [r for r in _read_jsonl(VOICE_FILE) if r.get("user_id") == user_id]
        return records[-1] if records else None

    # ── PREDICTIONS ──────────────────────────────────────────────────────────
    def store_prediction(self, result: dict):
        _append_jsonl(PREDS_FILE, result)

    def get_prediction_history(self, user_id: str, limit: int = 50) -> List[dict]:
        records = [r for r in _read_jsonl(PREDS_FILE) if r.get("user_id") == user_id]
        return records[-limit:]

    def get_latest_prediction(self, user_id: str) -> Optional[dict]:
        history = self.get_prediction_history(user_id, 1)
        return history[-1] if history else None

    # ── TRAINING DATA ─────────────────────────────────────────────────────────
    def get_all_vitals_for_training(self) -> List[dict]:
        """Return all stored vitals + prediction labels for retraining."""
        vitals = _read_jsonl(VITALS_FILE)
        preds = _read_jsonl(PREDS_FILE)
        # Build user_id → latest label map
        label_map = {}
        for p in preds:
            label_map[p.get("user_id")] = p.get("label")
        # Attach labels
        result = []
        for v in vitals:
            uid = v.get("user_id")
            if uid in label_map:
                v["risk_label"] = label_map[uid]
                result.append(v)
        return result


db = Database()
