import pytest

from task_manager import TaskList


def test_blank_titles_are_rejected_and_titles_are_trimmed():
    tasks = TaskList()
    with pytest.raises(ValueError, match="blank"):
        tasks.add("   ")
    tasks.add("  ship it  ")
    assert tasks.pending()[0]["title"] == "ship it"
