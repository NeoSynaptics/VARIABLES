"""
StreamingLLM -- SSE streaming for LLM responses.
Works with Ollama + OpenAI-compatible APIs.

Usage: create_stream_endpoint(app, "/chat/stream")
"""
import json
import time
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import httpx


async def stream_ollama(
    model: str, messages: list[dict],
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", f"{base_url}/api/chat",
            json={"model": model, "messages": messages, "stream": True,
                  "options": {"temperature": temperature}},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
                if chunk.get("done"):
                    return


async def stream_openai_compatible(
    model: str, messages: list[dict],
    base_url: str = "https://api.openai.com/v1",
    api_key: str = "",
    temperature: float = 0.7,
) -> AsyncGenerator[str, None]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST", f"{base_url}/chat/completions",
            headers=headers,
            json={"model": model, "messages": messages, "stream": True,
                  "temperature": temperature},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    return
                chunk = json.loads(data)
                content = chunk["choices"][0].get("delta", {}).get("content", "")
                if content:
                    yield content


async def to_sse(gen: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """Wrap token generator as SSE events."""
    async for token in gen:
        event = json.dumps({"token": token, "ts": time.time()})
        yield f"data: {event}

"
    done_event = json.dumps({"done": True, "ts": time.time()})
    yield f"data: {done_event}

"


def create_stream_endpoint(app: FastAPI, path: str = "/chat/stream"):
    """Add an SSE streaming chat endpoint to a FastAPI app."""
    @app.post(path)
    async def stream_chat(request: Request):
        body = await request.json()
        model = body.get("model", "qwen2.5-coder:14b")
        messages = body.get("messages", [])
        provider = body.get("provider", "ollama")
        if provider == "ollama":
            gen = stream_ollama(model, messages, body.get("base_url", "http://localhost:11434"))
        else:
            gen = stream_openai_compatible(
                model, messages, body.get("base_url", ""),
                api_key=body.get("api_key", ""),
            )
        return StreamingResponse(to_sse(gen), media_type="text/event-stream")
