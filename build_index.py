"""
build_index.py — Aggregates all module.json files into index.json.

Run this whenever you add a new module to VARIABLES.
Also called automatically by AlchemyGoldOS on first /variables/modules request.

Output: VARIABLES/index.json
"""
import json
from pathlib import Path

BASE = Path(__file__).parent


def build() -> dict:
    modules = []

    # Internal modules — each folder with a module.json
    for module_json in sorted(BASE.rglob("module.json")):
        try:
            data = json.loads(module_json.read_text(encoding="utf-8"))
            data["source"] = data.get("source", "self")
            data["path"] = module_json.parent.name
            modules.append(data)
        except Exception as e:
            print(f"  skip {module_json}: {e}")

    # External modules — from external.json
    external_path = BASE / "external.json"
    if external_path.exists():
        ext = json.loads(external_path.read_text(encoding="utf-8"))
        for name, info in ext.get("modules", {}).items():
            modules.append({
                "name": name,
                "description": info.get("description", ""),
                "language": info.get("language", ""),
                "tags": info.get("tags", []),
                "source": "external",
                "repo": info.get("repo", ""),
                "path": None,
            })

    index = {"count": len(modules), "modules": modules}
    out = BASE / "index.json"
    out.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"index.json: {len(modules)} modules ({sum(1 for m in modules if m['source'] == 'self')} internal, {sum(1 for m in modules if m['source'] == 'external')} external)")
    return index


if __name__ == "__main__":
    build()
