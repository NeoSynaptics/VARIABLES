"""
SparseCheckout — Git sparse checkout for pulling individual VARIABLES modules.

Allows pulling a single module from the VARIABLES repo without cloning everything.

Usage:
    pull = SparsePull("https://github.com/NeoSynaptics/VARIABLES.git")
    pull.fetch_module("OllamaGate", "./my-project/lib/")
    pull.fetch_module("StreamingLLM", "./my-project/lib/")
"""
import os
import subprocess
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PullResult:
    module: str
    destination: str
    files: list[str]
    success: bool
    error: Optional[str] = None


class SparsePull:
    """Pull individual modules from a VARIABLES-style monorepo."""

    def __init__(self, repo_url: str, branch: str = "main"):
        self.repo_url = repo_url
        self.branch = branch

    def fetch_module(self, module_name: str, dest_dir: str, overwrite: bool = False) -> PullResult:
        dest = Path(dest_dir) / module_name
        if dest.exists() and not overwrite:
            return PullResult(module=module_name, destination=str(dest), files=[], success=False,
                              error=f"Destination {dest} already exists. Use overwrite=True.")
        tmp_dir = Path(dest_dir) / f".sparse-tmp-{module_name}"
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            # Init sparse repo
            self._run(["git", "init"], cwd=tmp_dir)
            self._run(["git", "remote", "add", "origin", self.repo_url], cwd=tmp_dir)
            self._run(["git", "config", "core.sparseCheckout", "true"], cwd=tmp_dir)
            # Set sparse patterns
            sparse_file = tmp_dir / ".git" / "info" / "sparse-checkout"
            sparse_file.parent.mkdir(parents=True, exist_ok=True)
            sparse_file.write_text(f"{module_name}/
")
            # Fetch only the needed tree
            self._run(["git", "pull", "--depth=1", "origin", self.branch], cwd=tmp_dir)
            # Move module to destination
            src = tmp_dir / module_name
            if not src.exists():
                return PullResult(module=module_name, destination=str(dest), files=[], success=False,
                                  error=f"Module {module_name} not found in repo")
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            files = [str(f.relative_to(dest)) for f in dest.rglob("*") if f.is_file()]
            return PullResult(module=module_name, destination=str(dest), files=files, success=True)
        except subprocess.CalledProcessError as e:
            return PullResult(module=module_name, destination=str(dest), files=[], success=False,
                              error=f"Git error: {e.stderr or e.stdout or str(e)}")
        except Exception as e:
            return PullResult(module=module_name, destination=str(dest), files=[], success=False,
                              error=str(e))
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def list_modules(self) -> list[dict]:
        """Fetch and parse index.json or scan module directories from the repo."""
        tmp_dir = Path("/tmp/.sparse-list-tmp")
        try:
            tmp_dir.mkdir(parents=True, exist_ok=True)
            self._run(["git", "init"], cwd=tmp_dir)
            self._run(["git", "remote", "add", "origin", self.repo_url], cwd=tmp_dir)
            self._run(["git", "config", "core.sparseCheckout", "true"], cwd=tmp_dir)
            sparse_file = tmp_dir / ".git" / "info" / "sparse-checkout"
            sparse_file.parent.mkdir(parents=True, exist_ok=True)
            sparse_file.write_text("index.json
")
            self._run(["git", "pull", "--depth=1", "origin", self.branch], cwd=tmp_dir)
            index_path = tmp_dir / "index.json"
            if index_path.exists():
                return json.loads(index_path.read_text())
            return []
        except Exception:
            return []
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)

    @staticmethod
    def _run(cmd: list[str], cwd: Path = None) -> str:
        result = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                                capture_output=True, text=True, check=True)
        return result.stdout


def pull_module(repo_url: str, module_name: str, dest: str) -> PullResult:
    """Convenience function for one-shot module pull."""
    return SparsePull(repo_url).fetch_module(module_name, dest)
