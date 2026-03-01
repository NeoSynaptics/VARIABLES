"""Standalone task decomposition engine.

Takes high-level task descriptions or PRDs and breaks them into ordered,
dependency-aware, atomic subtasks using any Ollama-compatible or
OpenAI-compatible LLM API.

Zero project dependencies — pure Python + httpx.

Usage:
    from tool import TaskDecomposer, ollama_backend

    decomposer = TaskDecomposer(llm=ollama_backend("qwen2.5-coder:14b"))
    result = await decomposer.decompose("Build JWT auth with login/register/reset")
    for task in result.tasks:
        print(f"{task.id}: {task.title} (deps: {task.dependencies})")
"""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Backend Protocol
# ---------------------------------------------------------------------------

class LLMBackend(Protocol):
    """Any async callable that takes (system_prompt, user_prompt) → text."""

    async def __call__(self, system: str, user: str) -> str: ...


def ollama_backend(
    model: str = "qwen2.5-coder:14b",
    host: str = "http://localhost:11434",
    timeout: float = 180.0,
) -> LLMBackend:
    """Create an Ollama-backed LLM callable."""

    async def _call(system: str, user: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 4000, "num_ctx": 16384},
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{host}/api/chat", json=payload)
            if resp.status_code != 200:
                return ""
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()

    return _call  # type: ignore[return-value]


def openai_backend(
    model: str = "gpt-4",
    api_key: str = "",
    base_url: str = "https://api.openai.com/v1",
    timeout: float = 120.0,
) -> LLMBackend:
    """Create an OpenAI-compatible backend."""

    async def _call(system: str, user: str) -> str:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": 4000,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
            if resp.status_code != 200:
                return ""
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    return _call  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Data Models (pure dataclasses — no Pydantic)
# ---------------------------------------------------------------------------

@dataclass
class DecomposedTask:
    """A single decomposed subtask."""

    id: str
    title: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    complexity: float = 0.0
    target_file: str = ""
    test_file: str = ""
    task_type: str = "code"  # code | test | research | config | refactor


@dataclass
class Phase:
    """Named group of task IDs (for PRD decomposition)."""

    name: str
    task_ids: list[str] = field(default_factory=list)


@dataclass
class DecompositionOutput:
    """Complete decomposition result."""

    summary: str = ""
    tasks: list[DecomposedTask] = field(default_factory=list)
    effort: str = "medium"
    phases: list[Phase] = field(default_factory=list)
    dependency_order: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph Utilities
# ---------------------------------------------------------------------------

def topological_sort(tasks: list[DecomposedTask]) -> tuple[list[str], list[str]]:
    """Kahn's algorithm. Returns (sorted_ids, cycle_members)."""
    task_ids = {t.id for t in tasks}
    adj: dict[str, list[str]] = {t.id: [] for t in tasks}
    in_degree: dict[str, int] = {t.id: 0 for t in tasks}

    for t in tasks:
        for dep in t.dependencies:
            if dep in task_ids:
                adj[dep].append(t.id)
                in_degree[t.id] += 1

    queue: deque[str] = deque(sorted(tid for tid, deg in in_degree.items() if deg == 0))
    sorted_ids: list[str] = []

    while queue:
        node = queue.popleft()
        sorted_ids.append(node)
        for nb in sorted(adj[node]):
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)

    cycle_members = [tid for tid in in_degree if tid not in set(sorted_ids)]
    return sorted_ids, cycle_members


def detect_cycles(tasks: list[DecomposedTask]) -> list[list[str]]:
    """DFS-based cycle detection. Returns list of cycle paths."""
    task_ids = {t.id for t in tasks}
    adj: dict[str, list[str]] = {t.id: [] for t in tasks}
    for t in tasks:
        for dep in t.dependencies:
            if dep in task_ids:
                adj[dep].append(t.id)

    WHITE, GREY, BLACK = 0, 1, 2
    colour = {tid: WHITE for tid in task_ids}
    parent: dict[str, str | None] = {tid: None for tid in task_ids}
    cycles: list[list[str]] = []

    def _dfs(node: str) -> None:
        colour[node] = GREY
        for nxt in sorted(adj.get(node, [])):
            if colour[nxt] == GREY:
                path = [nxt]
                cur = node
                while cur != nxt:
                    path.append(cur)
                    cur = parent.get(cur, nxt)
                path.append(nxt)
                path.reverse()
                cycles.append(path)
            elif colour[nxt] == WHITE:
                parent[nxt] = node
                _dfs(nxt)
        colour[node] = BLACK

    for tid in sorted(task_ids):
        if colour[tid] == WHITE:
            _dfs(tid)
    return cycles


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = """\
You are a task decomposition engine. Given a high-level task description,
break it into ordered, atomic subtasks that can be implemented one at a time.

RULES:
1. Each subtask MUST be independently testable.
2. Dependencies between subtasks MUST be explicit (by ID).
3. Subtask IDs are sequential: "1", "2", "3", etc.
4. NO circular dependencies. Task N can only depend on tasks with LOWER IDs.
5. Complexity scores: 1-2=trivial, 3-4=straightforward, 5-6=moderate, 7-8=complex, 9-10=research.
6. Start with data models, then core logic, then API, then tests.

Return ONLY valid JSON:
{"summary":"...","subtasks":[{"id":"1","title":"...","description":"...","dependencies":[],"complexity":3,"target_file":"","test_file":"","task_type":"code"}],"estimated_effort":"medium"}
"""

_PRD_PROMPT = """\
You are a PRD-to-task-list converter. Extract every actionable task in dependency order.
Maximum 50 subtasks. Group into phases.

Return ONLY valid JSON:
{"summary":"...","subtasks":[...],"estimated_effort":"large","phases":[{"name":"Foundation","task_ids":["1","2"]}]}
"""


# ---------------------------------------------------------------------------
# TaskDecomposer
# ---------------------------------------------------------------------------

class TaskDecomposer:
    """Standalone task decomposition engine.

    Args:
        llm: An async callable (system, user) → text. Use ollama_backend()
             or openai_backend() to create one.
        max_subtasks: Hard cap on total subtasks per decomposition.
    """

    def __init__(self, llm: LLMBackend, max_subtasks: int = 50) -> None:
        self._llm = llm
        self._max_subtasks = max_subtasks

    async def decompose(
        self,
        description: str,
        context: str = "",
    ) -> DecompositionOutput:
        """Break a high-level task into ordered subtasks."""
        user_msg = description
        if context:
            user_msg = f"Context:\n{context[:4000]}\n\n--- Task ---\n{description}"

        raw = await self._llm(_DECOMPOSE_PROMPT, user_msg)
        return self._parse(raw)

    async def decompose_prd(
        self,
        prd_text: str,
        context: str = "",
    ) -> DecompositionOutput:
        """Parse a full PRD into a task tree with phases."""
        user_msg = prd_text
        if context:
            user_msg = f"Context:\n{context[:4000]}\n\n--- PRD ---\n{prd_text}"

        raw = await self._llm(_PRD_PROMPT, user_msg)
        result = self._parse(raw)

        # Parse phases
        try:
            data = self._extract_json(raw)
            if "phases" in data:
                result.phases = [
                    Phase(name=p.get("name", ""), task_ids=p.get("task_ids", []))
                    for p in data["phases"]
                ]
        except Exception:
            pass

        return result

    def validate(self, tasks: list[DecomposedTask]) -> tuple[bool, list[str]]:
        """Cycle check. Returns (valid, errors)."""
        _, cycle_members = topological_sort(tasks)
        if cycle_members:
            return False, [f"Dependency cycle involving: {', '.join(cycle_members)}"]
        return True, []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_json(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _parse(self, raw: str) -> DecompositionOutput:
        if not raw:
            return DecompositionOutput(warnings=["LLM returned empty response"])

        data = self._extract_json(raw)
        if not data:
            return DecompositionOutput(warnings=["Failed to parse JSON"])

        tasks: list[DecomposedTask] = []
        for i, item in enumerate(data.get("subtasks", [])):
            if not isinstance(item, dict):
                continue
            tasks.append(DecomposedTask(
                id=str(item.get("id", str(i + 1))),
                title=str(item.get("title", f"Task {i + 1}"))[:120],
                description=str(item.get("description", ""))[:500],
                dependencies=[str(d) for d in item.get("dependencies", [])],
                complexity=float(item.get("complexity", 3)),
                target_file=str(item.get("target_file", "")),
                test_file=str(item.get("test_file", "")),
                task_type=str(item.get("task_type", "code")),
            ))

        if len(tasks) > self._max_subtasks:
            tasks = tasks[:self._max_subtasks]

        sorted_ids, cycle_members = topological_sort(tasks)
        warnings: list[str] = []
        if cycle_members:
            warnings.append(f"Dependency cycle detected: {', '.join(cycle_members)}")

        avg_complexity = sum(t.complexity for t in tasks) / len(tasks) if tasks else 0.0

        return DecompositionOutput(
            summary=data.get("summary", ""),
            tasks=tasks,
            effort=data.get("estimated_effort", "medium"),
            dependency_order=sorted_ids,
            warnings=warnings,
        )
