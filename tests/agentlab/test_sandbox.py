from __future__ import annotations

import os
from pathlib import Path

import pytest

from agentlab.env.sandbox import PathViolation, Sandbox


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "file.py").write_text("value = 1\n", encoding="utf-8")
    return repo


def test_relative_path_and_source_isolation(tmp_path):
    repo = make_repo(tmp_path)
    with Sandbox(repo) as sandbox:
        target = sandbox.resolve("file.py")
        target.write_text("value = 2\n", encoding="utf-8")
        assert target.read_text(encoding="utf-8") == "value = 2\n"
    assert (repo / "file.py").read_text(encoding="utf-8") == "value = 1\n"


@pytest.mark.parametrize("path", ["../secret", "../../etc/passwd"])
def test_rejects_parent_traversal(tmp_path, path):
    with Sandbox(make_repo(tmp_path)) as sandbox:
        with pytest.raises(PathViolation):
            sandbox.resolve(path, must_exist=False)


def test_rejects_absolute_path(tmp_path):
    with Sandbox(make_repo(tmp_path)) as sandbox:
        with pytest.raises(PathViolation):
            sandbox.resolve(str(tmp_path.resolve()))


def test_rejects_symlink_escape(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = repo / "link.txt"
    real_symlink = True
    try:
        os.symlink(outside, link)
    except OSError:
        real_symlink = False
        link.write_text("simulated link", encoding="utf-8")
    with Sandbox(repo) as sandbox:
        if not real_symlink:
            candidate = sandbox.root / "link.txt"
            original_resolve = Path.resolve

            def simulated_resolve(path, strict=False):
                if path == candidate:
                    return outside.resolve()
                return original_resolve(path, strict=strict)

            monkeypatch.setattr(Path, "resolve", simulated_resolve)
        with pytest.raises(PathViolation):
            sandbox.resolve("link.txt")
