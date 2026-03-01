"""
OllamaGate -- AI Gatekeeper for Claude Code.
Ollama 14B reviews permission requests: accept/deny/other.
"""

import json, re, time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import httpx


class Action(Enum):
    ACCEPT = "accept"
    DENY = "deny"
    OTHER = "other"


class Tier(Enum):
    ALWAYS_ACCEPT = "always_accept"
    ALWAYS_DENY = "always_deny"
    ASK_OLLAMA = "ask_ollama"


@dataclass
class GateResult:
    action: Action
    reason: str
    tier: Tier
    latency_ms: float = 0
    model: str = ""

@dataclass
class SafetyRules:
    always_accept: list[str] = field(default_factory=lambda: [
        "Read", "Glob", "Grep", "WebSearch", "WebFetch", "TodoWrite", "AskUserQuestion"])
    safe_bash_patterns: list[str] = field(default_factory=lambda: [
        r"^ls\b", r"^pwd$", r"^echo\b", r"^cat\b", r"^git\s+(status|log|diff|branch)\b",
        r"^npm\s+(list|ls|outdated|audit)\b", r"^wc\b", r"^whoami$", r"^date$"])
    always_deny_patterns: list[str] = field(default_factory=lambda: [
        r"rm\s+-rf\s+/(?!tmp)", r"rm\s+-rf\s+~", r"format\s+[A-Z]:",
        r"DROP\s+(?:DATABASE|TABLE)", r"git\s+push\s+--force\s+.*main",
        r"mkfs\.", r"dd\s+if=.*of=/dev/"])
    protected_paths: list[str] = field(default_factory=lambda: [
        ".env", "credentials", "secret", ".ssh/", "id_rsa", ".gpg", "token", "password"])
    review_bash_patterns: list[str] = field(default_factory=lambda: [
        r"^rm\b", r"^git\s+push", r"^git\s+reset", r"^git\s+checkout\s+--",
        r"^npm\s+(install|uninstall|publish)", r"^pip\s+install",
        r"^chmod\b", r"^chown\b", r"^kill\b"])

    @classmethod
    def from_file(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


SYSTEM_PROMPT = ("You are a security reviewer for Claude Code. "
    "Review tool actions and decide if safe.\n\n"
    "Respond JSON: {action: accept|deny|other, reason: max 10 words}\n\n"
    "Rules: reads=safe, .env/creds=dangerous, force-push main=dangerous, "
    "when in doubt accept. Keep reason under 10 words.")


class OllamaGate:
    def __init__(self, model="qwen2.5-coder:14b", base_url="http://localhost:11434",
                 rules=None, timeout=5.0):
        self.model, self.base_url = model, base_url
        self.rules = rules or SafetyRules()
        self.timeout = timeout
        self.stats = {"accepted": 0, "denied": 0, "other": 0, "fast_path": 0, "ollama_calls": 0}

    def classify_tier(self, tool_name, args):
        if tool_name in self.rules.always_accept: return Tier.ALWAYS_ACCEPT
        if tool_name == "Bash": return self._classify_bash(args.get("command", ""))
        if tool_name in ("Write", "Edit"):
            for p in self.rules.protected_paths:
                if p in args.get("file_path", "").lower(): return Tier.ALWAYS_DENY
        return Tier.ASK_OLLAMA

    def _classify_bash(self, cmd):
        cmd = cmd.strip()
        for p in self.rules.always_deny_patterns:
            if re.search(p, cmd, re.IGNORECASE): return Tier.ALWAYS_DENY
        for p in self.rules.safe_bash_patterns:
            if re.match(p, cmd, re.IGNORECASE): return Tier.ALWAYS_ACCEPT
        for p in self.rules.review_bash_patterns:
            if re.match(p, cmd, re.IGNORECASE): return Tier.ASK_OLLAMA
        return Tier.ASK_OLLAMA

    async def review(self, tool_name, args, project_context=None):
        tier = self.classify_tier(tool_name, args)
        if tier == Tier.ALWAYS_ACCEPT:
            self.stats["accepted"] += 1; self.stats["fast_path"] += 1
            return GateResult(Action.ACCEPT, "safe operation", tier)
        if tier == Tier.ALWAYS_DENY:
            self.stats["denied"] += 1; self.stats["fast_path"] += 1
            return GateResult(Action.DENY, "blocked by safety rules", tier)
        return await self._ask_ollama(tool_name, args, project_context or {})

    async def _ask_ollama(self, tool_name, args, context):
        prompt = self._build_prompt(tool_name, args, context)
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json={
                    "model": self.model,
                    "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": prompt}],
                    "stream": False, "format": "json",
                    "options": {"temperature": 0.1, "num_predict": 64}})
                resp.raise_for_status(); data = resp.json()
            ms = (time.perf_counter() - start) * 1000
            self.stats["ollama_calls"] += 1
            d = json.loads(data.get("message", {}).get("content", "{}"))
            a = d.get("action", "accept").lower()
            r = d.get("reason", "no reason")[:80]
            action = {"deny": Action.DENY, "other": Action.OTHER}.get(a, Action.ACCEPT)
            self.stats[{Action.DENY: "denied", Action.OTHER: "other"}.get(action, "accepted")] += 1
            return GateResult(action, r, Tier.ASK_OLLAMA, round(ms, 1), self.model)
        except Exception as e:
            ms = (time.perf_counter() - start) * 1000
            self.stats["accepted"] += 1
            return GateResult(Action.ACCEPT, f"ollama unavailable: {str(e)[:40]}",
                              Tier.ASK_OLLAMA, round(ms, 1))

    def _build_prompt(self, tool, args, ctx):
        p = [f"TOOL: {tool}"]
        if tool == "Bash": p.append(f"COMMAND: {args.get('command', '?')}")
        elif tool in ("Write", "Edit"): p.append(f"FILE: {args.get('file_path', '?')}")
        else: p.append(f"ARGS: {json.dumps(args)[:200]}")
        if ctx.get("project"): p.append(f"PROJECT: {ctx['project']}")
        p.append("Is this safe? Respond JSON: {action, reason}")
        return "\n".join(p)

    def get_stats(self): return {**self.stats, "model": self.model}
