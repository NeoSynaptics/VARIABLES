"""
RealtimeWS -- Reusable WebSocket relay server.
Drop-in bidirectional relay: clients subscribe to rooms, messages broadcast to all peers.

Usage:
    uvicorn RealtimeWS.server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


@dataclass
class Client:
    ws: WebSocket
    room: str
    client_id: str
    connected_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class RelayHub:
    def __init__(self):
        self.rooms: dict[str, dict[str, Client]] = {}
        self._hooks: dict[str, list] = {}

    def on(self, event: str, callback):
        self._hooks.setdefault(event, []).append(callback)

    async def _emit(self, event: str, **kwargs):
        for cb in self._hooks.get(event, []):
            result = cb(**kwargs)
            if asyncio.iscoroutine(result):
                await result

    async def join(self, ws: WebSocket, room: str, client_id: str, metadata: dict = None) -> Client:
        await ws.accept()
        client = Client(ws=ws, room=room, client_id=client_id, metadata=metadata or {})
        self.rooms.setdefault(room, {})[client_id] = client
        await self._emit("join", client=client, room=room)
        return client

    async def leave(self, client: Client):
        room_clients = self.rooms.get(client.room, {})
        room_clients.pop(client.client_id, None)
        if not room_clients:
            self.rooms.pop(client.room, None)
        await self._emit("leave", client=client, room=client.room)

    async def broadcast(self, room: str, message: Any, exclude: str = None):
        clients = self.rooms.get(room, {})
        payload = json.dumps(message) if isinstance(message, dict) else message
        dead = []
        for cid, client in clients.items():
            if cid == exclude:
                continue
            try:
                await client.ws.send_text(payload)
            except Exception:
                dead.append(cid)
        for cid in dead:
            client = clients.pop(cid, None)
            if client:
                await self._emit("leave", client=client, room=room)

    async def send_to(self, room: str, client_id: str, message: Any):
        client = self.rooms.get(room, {}).get(client_id)
        if client:
            payload = json.dumps(message) if isinstance(message, dict) else message
            await client.ws.send_text(payload)

    def get_room_clients(self, room: str) -> list[str]:
        return list(self.rooms.get(room, {}).keys())

    def get_all_rooms(self) -> dict[str, int]:
        return {room: len(clients) for room, clients in self.rooms.items()}


hub = RelayHub()


@app.websocket("/ws/{room}/{client_id}")
async def websocket_relay(ws: WebSocket, room: str, client_id: str):
    client = await hub.join(ws, room, client_id)
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                msg = {"type": "text", "content": data}
            await hub._emit("message", client=client, message=msg)
            if msg.get("type") != "control":
                await hub.broadcast(room, {
                    "from": client_id, "room": room, "payload": msg, "ts": time.time(),
                }, exclude=client_id)
    except WebSocketDisconnect:
        await hub.leave(client)


@app.get("/rooms")
async def list_rooms():
    return hub.get_all_rooms()
