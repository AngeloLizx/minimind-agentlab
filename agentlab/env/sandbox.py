from __future__ import annotations

import difflib
import os
import shutil
import tempfile
from pathlib import Path


class PathViolation(ValueError):
    pass


class Sandbox:
    """Task-local repository copy with containment and symlink checks."""

    def __init__(self, source_repo: str | Path, keep: bool = False):
        self.source_repo = Path(source_repo).resolve()
        if not self.source_repo.is_dir():
            raise FileNotFoundError(f"Repository does not exist: {self.source_repo}")
        self.keep = keep
        self._temp_root: Path | None = None
        self.root: Path | None = None
        self._snapshot: dict[str, bytes] = {}

    def __enter__(self) -> "Sandbox":
        self._temp_root = Path(tempfile.mkdtemp(prefix="minimind-agentlab-"))
        self.root = self._temp_root / "repo"
        shutil.copytree(self.source_repo, self.root, symlinks=True)
        self._snapshot = self._read_snapshot()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.keep and self._temp_root:
            shutil.rmtree(self._temp_root, ignore_errors=True)

    def resolve(self, relative_path: str | Path, must_exist: bool = True) -> Path:
        if self.root is None:
            raise RuntimeError("Sandbox is not active")
        raw = Path(relative_path)
        if raw.is_absolute() or os.path.isabs(str(relative_path)):
            raise PathViolation("Absolute paths are forbidden")
        if any(part == ".." for part in raw.parts):
            raise PathViolation("Parent traversal is forbidden")
        candidate = self.root.joinpath(raw)
        resolved = candidate.resolve(strict=must_exist)
        root = self.root.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise PathViolation("Path escapes sandbox") from exc
        if must_exist and candidate.is_symlink():
            target = candidate.resolve(strict=True)
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise PathViolation("Symlink escapes sandbox") from exc
        return resolved

    def install_validator(self, validator_path: str | Path | None) -> None:
        if not validator_path:
            return
        src = Path(validator_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Validator not found: {src}")
        dest_dir = self.resolve("tests", must_exist=False)
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / "test_agentlab_task.py")
        self._snapshot = self._read_snapshot()

    def _read_snapshot(self) -> dict[str, bytes]:
        assert self.root
        result: dict[str, bytes] = {}
        for path in self.root.rglob("*"):
            ignored = {"__pycache__", ".pytest_cache", ".git", ".mypy_cache"}
            if (
                path.is_file()
                and not path.is_symlink()
                and not any(part in ignored for part in path.parts)
                and path.suffix != ".pyc"
            ):
                result[path.relative_to(self.root).as_posix()] = path.read_bytes()
        return result

    def changed_files(self) -> list[str]:
        current = self._read_snapshot()
        return sorted(k for k in set(self._snapshot) | set(current) if self._snapshot.get(k) != current.get(k))

    def diff(self, max_chars: int = 20_000) -> str:
        current = self._read_snapshot()
        chunks: list[str] = []
        for name in sorted(set(self._snapshot) | set(current)):
            before, after = self._snapshot.get(name), current.get(name)
            if before == after:
                continue
            if before is None or after is None:
                chunks.append(f"Binary/add-delete change: {name}\n")
                continue
            try:
                old = before.decode("utf-8").splitlines(keepends=True)
                new = after.decode("utf-8").splitlines(keepends=True)
            except UnicodeDecodeError:
                chunks.append(f"Binary change: {name}\n")
                continue
            chunks.extend(difflib.unified_diff(old, new, f"a/{name}", f"b/{name}"))
        text = "".join(chunks)
        return text[:max_chars] + ("\n[diff truncated]" if len(text) > max_chars else "")
