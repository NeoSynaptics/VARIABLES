"""
save_candidate.py — Saves a ReuseScanner candidate as a VARIABLES module.

This is the handoff: ReuseScanner detects good code → save_candidate() lands it
in the vault with proper module.json metadata.

Usage:
    from save_candidate import save_candidate
    module_dir = save_candidate(candidate)          # auto-names from function
    module_dir = save_candidate(candidate, "MyUtil") # explicit name
"""
import json
import re
from pathlib import Path

BASE = Path(__file__).parent


def save_candidate(candidate, module_name: str = None) -> Path:
    """
    Save a ReuseScanner Candidate as a VARIABLES module.

    Creates:
        VARIABLES/{ModuleName}/
            {entry}.py / .ts     ← function code with imports
            module.json          ← metadata

    Rebuilds index.json automatically.
    Returns the created module directory.
    """
    name = module_name or _to_pascal(candidate.name)
    module_dir = BASE / name

    if module_dir.exists():
        raise FileExistsError(f"Module '{name}' already exists at {module_dir}. Use a different name or delete it first.")

    module_dir.mkdir()

    # Entry filename
    ext = ".py" if candidate.language == "python" else ".ts"
    entry_file = f"{_to_snake(name)}{ext}"

    # Write code: imports header + function body
    (module_dir / entry_file).write_text(_build_code(candidate), encoding="utf-8")

    # Write module.json
    meta = {
        "name": name,
        "description": f"Extracted from {candidate.origin_repo}: {candidate.name}()",
        "language": candidate.language,
        "tags": [candidate.category, candidate.language],
        "source": "self",
        "size": "small",
        "entry": entry_file,
        "origin": {
            "repo": candidate.origin_repo,
            "file": candidate.file_path,
            "function": candidate.name,
            "score": candidate.score,
        },
    }
    (module_dir / "module.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Rebuild index so it's always in sync
    _rebuild_index()

    print(f"Saved: {name}/ ({candidate.language}/{candidate.category}, score={candidate.score})")
    return module_dir


def save_raw(
    name: str,
    language: str,
    body: str,
    description: str = "",
    tags: list = None,
    imports: list = None,
    module_name: str = None,
) -> Path:
    """
    Save raw code (not from ReuseScanner) directly as a VARIABLES module.
    Used by AlchemyGoldOS POST /variables/save.
    """
    class _Raw:
        pass

    c = _Raw()
    c.name = name
    c.language = language
    c.body = body
    c.imports = imports or []
    c.origin_repo = "manual"
    c.file_path = ""
    c.score = 0
    c.category = (tags or ["utils"])[0]

    mod_dir = save_candidate(c, module_name)

    # Overwrite description and tags if provided
    if description or tags:
        meta_path = mod_dir / "module.json"
        meta = json.loads(meta_path.read_text())
        if description:
            meta["description"] = description
        if tags:
            meta["tags"] = tags
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return mod_dir


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_code(candidate) -> str:
    lines = []
    if getattr(candidate, "imports", None):
        for imp in candidate.imports:
            lines.append(imp.strip())
        lines.append("")
    lines.append(candidate.body)
    return "\n".join(lines)


def _to_pascal(name: str) -> str:
    """snake_case / camelCase → PascalCase."""
    parts = re.split(r"[_\-\s]+", name)
    return "".join(p.capitalize() for p in parts if p)


def _to_snake(name: str) -> str:
    """PascalCase → snake_case."""
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def _rebuild_index():
    import sys
    sys.path.insert(0, str(BASE))
    try:
        import importlib
        import build_index
        importlib.reload(build_index)
        build_index.build()
    finally:
        sys.path.pop(0)
