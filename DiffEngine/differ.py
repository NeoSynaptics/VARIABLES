"""
DiffEngine — Unified diff analysis, classification, and conflict detection.

Usage:
    engine = DiffEngine()
    result = engine.analyze(diff_text)
    # result.classification -> "feature" | "bugfix" | "refactor" | "test" | "docs" | "config"
    # result.files_changed -> [FileChange(...)]
    # result.conflicts -> [Conflict(...)]
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ChangeType(Enum):
    FEATURE = "feature"
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    TEST = "test"
    DOCS = "docs"
    CONFIG = "config"
    UNKNOWN = "unknown"


@dataclass
class FileChange:
    path: str
    additions: int = 0
    deletions: int = 0
    is_new: bool = False
    is_deleted: bool = False
    is_renamed: bool = False
    old_path: Optional[str] = None


@dataclass
class Conflict:
    file: str
    line_start: int
    line_end: int
    ours: str = ""
    theirs: str = ""


@dataclass
class DiffResult:
    classification: ChangeType
    files_changed: list[FileChange] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    total_additions: int = 0
    total_deletions: int = 0
    summary: str = ""


# Classification signals
SIGNALS = {
    ChangeType.TEST: [r"test[s_]?/", r"_test\.py$", r"\.test\.(ts|js)$", r"spec\.(ts|js)$"],
    ChangeType.DOCS: [r"README", r"\.md$", r"docs/", r"CHANGELOG"],
    ChangeType.CONFIG: [r"\.json$", r"\.ya?ml$", r"\.toml$", r"\.env", r"Dockerfile", r"\.gitignore"],
    ChangeType.BUGFIX: [r"fix", r"bug", r"patch", r"hotfix", r"issue"],
    ChangeType.REFACTOR: [r"refactor", r"rename", r"move", r"cleanup", r"reorganize"],
}


class DiffEngine:
    """Parses unified diffs, classifies changes, and detects merge conflicts."""

    def analyze(self, diff_text: str) -> DiffResult:
        files = self._parse_files(diff_text)
        conflicts = self._detect_conflicts(diff_text)
        classification = self._classify(diff_text, files)
        total_add = sum(f.additions for f in files)
        total_del = sum(f.deletions for f in files)
        summary = f"{len(files)} file(s), +{total_add}/-{total_del}"
        if conflicts:
            summary += f", {len(conflicts)} conflict(s)"
        return DiffResult(
            classification=classification,
            files_changed=files,
            conflicts=conflicts,
            total_additions=total_add,
            total_deletions=total_del,
            summary=summary,
        )

    def _parse_files(self, diff_text: str) -> list[FileChange]:
        files = []
        current = None
        for line in diff_text.splitlines():
            if line.startswith("diff --git"):
                if current:
                    files.append(current)
                parts = line.split()
                path = parts[-1].lstrip("b/") if len(parts) >= 4 else "unknown"
                current = FileChange(path=path)
            elif line.startswith("new file"):
                if current:
                    current.is_new = True
            elif line.startswith("deleted file"):
                if current:
                    current.is_deleted = True
            elif line.startswith("rename from") and current:
                current.is_renamed = True
                current.old_path = line.split("rename from ")[-1]
            elif current and line.startswith("+") and not line.startswith("+++"):
                current.additions += 1
            elif current and line.startswith("-") and not line.startswith("---"):
                current.deletions += 1
        if current:
            files.append(current)
        return files

    def _detect_conflicts(self, text: str) -> list[Conflict]:
        conflicts = []
        lines = text.splitlines()
        i = 0
        current_file = "unknown"
        while i < len(lines):
            if lines[i].startswith("diff --git"):
                parts = lines[i].split()
                current_file = parts[-1].lstrip("b/") if len(parts) >= 4 else "unknown"
            if lines[i].startswith("<<<<<<<"):
                start = i
                ours_lines, theirs_lines = [], []
                i += 1
                in_ours = True
                while i < len(lines) and not lines[i].startswith(">>>>>>>"):
                    if lines[i].startswith("======="):
                        in_ours = False
                    elif in_ours:
                        ours_lines.append(lines[i])
                    else:
                        theirs_lines.append(lines[i])
                    i += 1
                conflicts.append(Conflict(
                    file=current_file,
                    line_start=start,
                    line_end=i,
                    ours="
".join(ours_lines),
                    theirs="
".join(theirs_lines),
                ))
            i += 1
        return conflicts

    def _classify(self, diff_text: str, files: list[FileChange]) -> ChangeType:
        text_lower = diff_text.lower()
        scores: dict[ChangeType, int] = {ct: 0 for ct in ChangeType}
        # Score by file paths
        for f in files:
            for ct, patterns in SIGNALS.items():
                for pat in patterns:
                    if re.search(pat, f.path, re.IGNORECASE):
                        scores[ct] += 2
        # Score by diff content keywords
        for ct, patterns in SIGNALS.items():
            for pat in patterns:
                if re.search(pat, text_lower):
                    scores[ct] += 1
        # If mostly new files with real code, it is a feature
        new_files = sum(1 for f in files if f.is_new)
        if new_files > len(files) // 2:
            scores[ChangeType.FEATURE] += 3
        best = max(scores, key=lambda ct: scores[ct])
        return best if scores[best] > 0 else ChangeType.UNKNOWN


def quick_diff_summary(diff_text: str) -> str:
    """One-liner diff summary for logging or display."""
    engine = DiffEngine()
    result = engine.analyze(diff_text)
    return f"[{result.classification.value}] {result.summary}"
