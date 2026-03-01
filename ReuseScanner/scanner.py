"""
ReuseScanner — detects reusable code functions across local repos.
Zero external dependencies. Standard library only.

Scans Python and TypeScript source files, scores each extracted function
for reusability, and returns ranked candidates.

Scoring rubric (0–4):
  +1  Pure  — no self/cls param (Python) or minimal this. usage (TS)
  +1  Compact — body < 80 lines
  +1  Annotated — has docstring/JSDoc OR type hints on params/return
  +1  Cross-repo — same function name found in 2+ registered repos
  -1  Project-specific — name or body contains internal keywords

Usage:
    from ReuseScanner.scanner import ReuseScanner

    scanner = ReuseScanner(
        repo_paths=["~/projects/my-api", "~/projects/my-cli"],
        min_score=3,
    )
    candidates = scanner.scan_all()

    for c in candidates:
        print(f"{c.score}/4  {c.language}/{c.category}  {c.name}()")
        print(f"     {c.file_path}")
        print()
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Language detection by extension
_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "typescript",   # treat JS same as TS for scoring
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "typescript",
}

# Directories to skip entirely
_SKIP_DIRS: set[str] = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "env", ".env", "dist", "build", ".tox", ".pytest_cache",
    ".mypy_cache", "vendor", ".next", "coverage", "htmlcov",
}

# File patterns to skip (test files, generated files)
_SKIP_PATTERNS: list[re.Pattern] = [
    re.compile(r"test_.*\.(py|ts|js)$"),
    re.compile(r".*_test\.(py|ts|js)$"),
    re.compile(r".*\.(test|spec)\.(ts|tsx|js|jsx)$"),
    re.compile(r"conftest\.py$"),
    re.compile(r"setup\.(py|cfg)$"),
    re.compile(r"__init__\.py$"),
    re.compile(r".*\.min\.(js|css)$"),
    re.compile(r"package-lock\.json$"),
    re.compile(r"yarn\.lock$"),
]

# Project-specific keywords that signal non-generic code
_PROJECT_KEYWORDS: set[str] = {
    "alchemy", "goldos", "alchemygold", "memorep", "vaultspeed",
    "kemgas", "neosynaptic", "kernelstore", "kernelloop",
    "director", "watchdog",
}

# Category inference patterns → AlchemyTools categories
_CATEGORY_HINTS: dict[str, dict[re.Pattern, str]] = {
    "python": {
        re.compile(r"(jwt|oauth|token|auth|session|credential)", re.I): "auth",
        re.compile(r"(fetch|request|http|client|api|webhook|retry)", re.I): "api",
        re.compile(r"(parse|serialize|validate|transform|format|schema)", re.I): "data",
        re.compile(r"(prompt|completion|embedding|ollama|openai|llm|model)", re.I): "llm",
        re.compile(r"(decorator|pattern|dispatch|event|queue|cache)", re.I): "patterns",
    },
    "typescript": {
        re.compile(r"(fetch|axios|http|request|api|socket|stream)", re.I): "api",
        re.compile(r"(use[A-Z])", re.S): "hooks",
        re.compile(r"(component|render|jsx|tsx|props)", re.I): "components",
        re.compile(r"(reducer|store|dispatch|event|bus|pattern)", re.I): "patterns",
    },
}

_DEFAULT_CATEGORY: dict[str, str] = {"python": "utils", "typescript": "utils"}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A code function flagged as reusable."""

    id: str                         # stable SHA256[:16] of name:file_path
    name: str                       # function name
    file_path: str                  # absolute source path
    origin_repo: str                # repo folder name
    language: str                   # "python" | "typescript"
    score: int                      # 0–4
    body: str                       # function source code
    imports: list[str]              # import lines extracted from file header
    category: str                   # proposed AlchemyTools category
    status: str = "pending"         # pending | saved | dismissed
    created_at: float = field(default_factory=time.time)

    def summary(self) -> str:
        return f"{self.score}/4  {self.language}/{self.category}  {self.name}()  [{self.origin_repo}]"


# ---------------------------------------------------------------------------
# Lightweight code parser (stdlib only)
# ---------------------------------------------------------------------------

# Python: match top-level or class-level function definitions
_PY_FUNC = re.compile(
    r'^(?:    )?(?:async\s+)?def\s+(\w+)\s*\(',
    re.MULTILINE,
)

# TypeScript/JS: match function declarations and arrow functions assigned to const
_TS_FUNC = re.compile(
    r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[\(<]'
    r'|^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(',
    re.MULTILINE,
)

# Import line patterns
_PY_IMPORT = re.compile(r'^(?:import|from)\s+\S', re.MULTILINE)
_TS_IMPORT = re.compile(r'^import\s+.+from\s+', re.MULTILINE)


def _extract_functions(content: str, language: str) -> list[tuple[str, str]]:
    """Extract (name, body) pairs from source code.

    Returns a list of (function_name, function_body) tuples.
    Body is extracted via indentation heuristics (Python) or
    brace counting (TypeScript).
    """
    if language == "python":
        return _extract_python_functions(content)
    else:
        return _extract_ts_functions(content)


def _extract_python_functions(content: str) -> list[tuple[str, str]]:
    """Extract Python function bodies using indentation tracking."""
    results: list[tuple[str, str]] = []
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(    )?(?:async\s+)?def\s+(\w+)\s*\(', line)
        if not m:
            i += 1
            continue

        name = m.group(2)
        base_indent = len(line) - len(line.lstrip())
        body_lines = [line]
        i += 1

        while i < len(lines):
            cur = lines[i]
            # Blank lines are part of the body
            if cur.strip() == "":
                body_lines.append(cur)
                i += 1
                continue
            cur_indent = len(cur) - len(cur.lstrip())
            # Stop when we return to base indentation level
            if cur_indent <= base_indent and cur.strip():
                break
            body_lines.append(cur)
            i += 1

        # Trim trailing blank lines
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()

        results.append((name, "\n".join(body_lines)))

    return results


def _extract_ts_functions(content: str) -> list[tuple[str, str]]:
    """Extract TypeScript/JS function bodies using brace counting."""
    results: list[tuple[str, str]] = []

    for m in _TS_FUNC.finditer(content):
        name = m.group(1) or m.group(2)
        if not name:
            continue

        # Find the opening brace from the match position
        start = m.start()
        brace_start = content.find("{", m.end())
        if brace_start == -1:
            continue

        # Count braces to find the closing brace
        depth = 0
        pos = brace_start
        while pos < len(content):
            if content[pos] == "{":
                depth += 1
            elif content[pos] == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1

        body = content[start:pos + 1]
        if len(body) < 2:
            continue
        results.append((name, body))

    return results


def _extract_imports(content: str, language: str) -> list[str]:
    """Extract import lines from file header (first 30 lines)."""
    header = "\n".join(content.splitlines()[:30])
    pattern = _PY_IMPORT if language == "python" else _TS_IMPORT
    return pattern.findall(header)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(name: str, body: str, language: str, cross_repo_names: set[str]) -> int:
    """Compute reusability score (0–4)."""
    score = 0

    # +1 Pure: not a method (Python: no `def fn(self`, TS: minimal this.)
    if language == "python":
        if not re.match(r"def\s+\w+\s*\(\s*(self|cls)\b", body):
            score += 1
    else:
        if body.count("this.") < 2:
            score += 1

    # +1 Compact: < 80 lines
    if len(body.splitlines()) < 80:
        score += 1

    # +1 Annotated: docstring/JSDoc OR type hints
    has_doc = '"""' in body or "'''" in body or "/**" in body
    has_hints = bool(re.search(
        r"->\s*\w+|:\s*(str|int|float|bool|list|dict|Optional|Any)\b"
        r"|:\s*(string|number|boolean|void|Record|Array|Promise)\b",
        body,
    ))
    if has_doc or has_hints:
        score += 1

    # +1 Cross-repo: same function name appears in 2+ repos
    if name in cross_repo_names:
        score += 1

    # -1 Project-specific keywords
    if any(kw in f"{name} {body}".lower() for kw in _PROJECT_KEYWORDS):
        score -= 1

    return max(0, score)


def _infer_category(name: str, body: str, language: str) -> str:
    """Map function to the best AlchemyTools category."""
    text = f"{name} {body}"
    for pattern, category in _CATEGORY_HINTS.get(language, {}).items():
        if pattern.search(text):
            return category
    return _DEFAULT_CATEGORY.get(language, "utils")


def _candidate_id(name: str, file_path: str) -> str:
    return hashlib.sha256(f"{name}:{file_path}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# ReuseScanner
# ---------------------------------------------------------------------------

class ReuseScanner:
    """Scans local repos for reusable code candidates.

    Args:
        repo_paths:  List of absolute paths to repo roots to scan.
        min_score:   Minimum score to include a candidate (default 3).
        max_lines:   Skip files with more than this many lines (default 300).
    """

    def __init__(
        self,
        repo_paths: list[str],
        min_score: int = 3,
        max_lines: int = 300,
    ):
        self._repos = [Path(p).expanduser() for p in repo_paths]
        self._repos = [r for r in self._repos if r.exists()]
        self._min_score = min_score
        self._max_lines = max_lines

    def scan_all(self) -> list[Candidate]:
        """Scan all repos. Returns candidates ordered by score descending."""
        cross_repo = self._collect_cross_repo_names()

        seen: set[str] = set()
        candidates: list[Candidate] = []

        for repo in self._repos:
            repo_name = repo.name
            for file_path in self._walk(repo):
                lang = _LANG_MAP.get(file_path.suffix.lower())
                if not lang:
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except (OSError, PermissionError):
                    continue

                if len(content.splitlines()) > self._max_lines:
                    continue

                imports = _extract_imports(content, lang)

                for name, body in _extract_functions(content, lang):
                    if not name or len(body.strip()) < 3:
                        continue

                    s = _score(name, body, lang, cross_repo)
                    if s < self._min_score:
                        continue

                    cid = _candidate_id(name, str(file_path))
                    if cid in seen:
                        continue
                    seen.add(cid)

                    candidates.append(Candidate(
                        id=cid,
                        name=name,
                        file_path=str(file_path),
                        origin_repo=repo_name,
                        language=lang,
                        score=s,
                        body=body,
                        imports=imports,
                        category=_infer_category(name, body, lang),
                    ))

        candidates.sort(key=lambda c: -c.score)
        return candidates

    def top(self, n: int = 10) -> list[Candidate]:
        """Return the top N candidates by score."""
        return self.scan_all()[:n]

    # ── Helpers ──────────────────────────────────────────────────────

    def _collect_cross_repo_names(self) -> set[str]:
        """Names that appear in 2+ repos get a cross-repo bonus."""
        counts: dict[str, int] = {}
        for repo in self._repos:
            seen_in_repo: set[str] = set()
            for file_path in self._walk(repo):
                lang = _LANG_MAP.get(file_path.suffix.lower())
                if not lang:
                    continue
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except (OSError, PermissionError):
                    continue
                for name, _ in _extract_functions(content, lang):
                    if name and name not in seen_in_repo:
                        counts[name] = counts.get(name, 0) + 1
                        seen_in_repo.add(name)
        return {name for name, n in counts.items() if n >= 2}

    def _walk(self, root: Path):
        """Walk repo files, skipping noise directories and test files."""
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(skip in path.parts for skip in _SKIP_DIRS):
                continue
            if any(pat.search(path.name) for pat in _SKIP_PATTERNS):
                continue
            yield path
