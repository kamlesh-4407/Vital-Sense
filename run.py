"""
VitalSense v2 — Server Entry Point
=====================================
Run with:
    python run.py
    # or
    uvicorn run:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn
from backend.main import app

if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  🫀  VitalSense v2 — Starting server...")
    print("  Dashboard  →  http://localhost:8000")
    print("  API Docs   →  http://localhost:8000/docs")
    print("  WebSocket  →  ws://localhost:8000/ws/{user_id}")
    print("═"*55 + "\n")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
