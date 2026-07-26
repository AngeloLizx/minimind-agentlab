from task_manager import TaskList


def test_add_complete_and_pending():
    tasks = TaskList()
    tasks.add("write tests")
    tasks.add("run tests")
    tasks.complete(0)
    assert tasks.pending() == [{"title": "run tests", "done": False}]
