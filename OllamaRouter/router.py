"""
OllamaRouter -- Smart complexity-based model routing for Ollama.
Trivial queries -> small model ($0). Complex queries -> big model.

Usage:
    router = OllamaRouter()
    result = await router.query("What is 2+2?")       # -> small model
    result = await router.query("Analyze this arch...")  # -> big model
"""
import re, time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import httpx


class Complexity(Enum):
    TRIVIAL = "trivial"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class ModelConfig:
    name: str
    complexity: list[Complexity]
    base_url: str = "http://localhost:11434"
    temperature: float = 0.7
    max_tokens: int = 2048


@dataclass
class RouteResult:
    model: str
    complexity: Complexity
    response: str
    latency_ms: float
    tokens_estimated: int = 0


DEFAULT_MODELS = [
    ModelConfig("qwen2.5-coder:7b", [Complexity.TRIVIAL]),
    ModelConfig("qwen2.5-coder:14b", [Complexity.MODERATE, Complexity.COMPLEX]),
]


class TrivialityDetector:
    TRIVIAL_PATTERNS = [
        r"^(yes|no|ok|sure|thanks|got it)",
        r"^(what is|who is)\s+\w+\?*$",
        r"^\d+\s*[\+\-\*\/]\s*\d+",
    ]
    COMPLEX_SIGNALS = [
        "architect", "design", "refactor", "implement", "analyze",
        "compare", "trade-off", "strategy", "plan", "debug",
        "optimize", "migration", "security",
    ]

    def classify(self, query: str) -> Complexity:
        q = query.strip().lower()
        words = len(q.split())
        if words <= 5:
            return Complexity.TRIVIAL
        for pat in self.TRIVIAL_PATTERNS:
            if re.match(pat, q, re.IGNORECASE):
                return Complexity.TRIVIAL
        hits = sum(1 for s in self.COMPLEX_SIGNALS if s in q)
        if hits >= 2 or words > 50:
            return Complexity.COMPLEX
        if hits == 1 or words > 20:
            return Complexity.MODERATE
        return Complexity.TRIVIAL


class OllamaRouter:
    def __init__(self, models: list[ModelConfig] = None):
        self.models = models or DEFAULT_MODELS
        self.detector = TrivialityDetector()
        self._model_map = {c: m for m in self.models for c in m.complexity}
        self.stats = {"trivial": 0, "moderate": 0, "complex": 0}

    def route(self, query: str) -> tuple[ModelConfig, Complexity]:
        complexity = self.detector.classify(query)
        return self._model_map.get(complexity, self.models[-1]), complexity

    async def query(self, prompt: str, system: str = "", force_model: str = None, timeout: float = 60.0) -> RouteResult:
        if force_model:
            model = next((m for m in self.models if m.name == force_model), self.models[-1])
            complexity = Complexity.COMPLEX
        else:
            model, complexity = self.route(prompt)
        self.stats[complexity.value] = self.stats.get(complexity.value, 0) + 1
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{model.base_url}/api/chat",
                json={"model": model.name, "messages": messages, "stream": False,
                      "options": {"temperature": model.temperature, "num_predict": model.max_tokens}},
            )
            resp.raise_for_status()
            data = resp.json()
        return RouteResult(
            model=model.name, complexity=complexity,
            response=data.get("message", {}).get("content", ""),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
            tokens_estimated=data.get("eval_count", 0),
        )
