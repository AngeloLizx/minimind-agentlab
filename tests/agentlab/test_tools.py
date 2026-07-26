from __future__ import annotations

from pathlib import Path

from agentlab.env.sandbox import Sandbox
from agentlab.tools import ToolRegistry


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "core.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (repo / "tests" / "test_core.py").write_text(
        "from src.core import add\n\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8"
    )
    return repo


def test_list_search_read_patch_and_diff(tmp_path):
    registry = ToolRegistry()
    with Sandbox(make_repo(tmp_path)) as sandbox:
        listed = registry.execute("list_files", sandbox, {"max_depth": 3})
        assert listed.success and "src/core.py" in listed.output
        searched = registry.execute("search_code", sandbox, {"query": "def add"})
        assert searched.success and "src/core.py:1" in searched.output
        read = registry.execute(
            "read_file", sandbox, {"path": "src/core.py", "start_line": 2, "end_line": 2}
        )
        assert read.success and "2 |" in read.output
        patch = registry.execute(
            "apply_patch",
            sandbox,
            {"path": "src/core.py", "old_text": "return a + b", "new_text": "return int(a) + int(b)"},
        )
        assert patch.success
        diff = registry.execute("git_diff", sandbox, {})
        assert diff.success and "return int(a)" in diff.output


def test_patch_requires_unique_existing_text(tmp_path):
    with Sandbox(make_repo(tmp_path)) as sandbox:
        result = ToolRegistry().execute(
            "apply_patch",
            sandbox,
            {"path": "src/core.py", "old_text": "not present", "new_text": "x"},
        )
        assert not result.success and result.error_category == "PATCH_ERROR"


def test_argument_validation_and_path_validation(tmp_path):
    with Sandbox(make_repo(tmp_path)) as sandbox:
        bad_arg = ToolRegistry().execute("read_file", sandbox, {"path": "src/core.py", "extra": 1})
        assert not bad_arg.success and bad_arg.error_category == "INVALID_ARGUMENT"
        bad_path = ToolRegistry().execute("read_file", sandbox, {"path": "../secret"})
        assert not bad_path.success and bad_path.error_category == "PATH_VIOLATION"


def test_run_tests_and_timeout(tmp_path):
    repo = make_repo(tmp_path)
    with Sandbox(repo) as sandbox:
        passed = ToolRegistry().execute(
            "run_tests", sandbox, {"command": "python -m pytest -q", "timeout": 10}
        )
        assert passed.success and passed.metadata["passed"] == 1
    (repo / "tests" / "test_slow.py").write_text(
        "import time\n\ndef test_slow():\n    time.sleep(3)\n", encoding="utf-8"
    )
    with Sandbox(repo) as sandbox:
        timed = ToolRegistry().execute(
            "run_tests", sandbox, {"command": "pytest -q", "timeout": 1}
        )
        assert not timed.success and timed.error_category == "TIMEOUT"
