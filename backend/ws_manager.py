"""
VitalSense — WebSocket Connection Manager
Handles per-user WebSocket connections for real-time broadcasting.
"""

from fastapi import WebSocket
from typing import Dict, List
import logging

logger = logging.getLogger("vitalsense.ws")


class ConnectionManager:
    def __init__(self):
        # user_id -> list of active WebSocket connections
        self._connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self._connections.setdefault(user_id, []).append(websocket)
        logger.info(f"WS connected: {user_id} ({len(self._connections[user_id])} sessions)")

    def disconnect(self, websocket: WebSocket, user_id: str):
        if user_id in self._connections:
            self._connections[user_id].discard(websocket) if hasattr(
                self._connections[user_id], 'discard'
            ) else None
            try:
                self._connections[user_id].remove(websocket)
            except ValueError:
                pass
            if not self._connections[user_id]:
                del self._connections[user_id]
        logger.info(f"WS disconnected: {user_id}")

    async def broadcast(self, user_id: str, message: dict):
        """Send a message to all connections for a given user."""
        dead = []
        for ws in self._connections.get(user_id, []):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws, user_id)

    async def broadcast_all(self, message: dict):
        """Broadcast to every connected user (e.g. system alerts)."""
        for user_id in list(self._connections.keys()):
            await self.broadcast(user_id, message)

    @property
    def active_users(self) -> List[str]:
        return list(self._connections.keys())
