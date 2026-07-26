from __future__ import annotations

from agentlab.schemas import AgentAction, AgentState


PATCHES: dict[str, tuple[str, str, str, str]] = {
    "calculator_fix_divide_zero": (
        "src/calculator/core.py",
        "def divide(a, b):\n    return a / b\n",
        "def divide(a, b):\n    if b == 0:\n        raise ValueError(\"divisor must not be zero\")\n    return a / b\n",
        "divide",
    ),
    "calculator_reject_boolean": (
        "src/calculator/core.py",
        "def add(a, b):\n    return a + b\n",
        "def add(a, b):\n    if isinstance(a, bool) or isinstance(b, bool):\n        raise TypeError(\"boolean operands are not supported\")\n    return a + b\n",
        "def add",
    ),
    "text_utils_empty_slug": (
        "src/text_utils/core.py",
        "    return value.strip().lower().replace(\" \", \"-\")\n",
        "    result = value.strip().lower().replace(\" \", \"-\")\n    return result or \"untitled\"\n",
        "slugify",
    ),
    "text_utils_truncate_boundary": (
        "src/text_utils/core.py",
        "    return text[:limit] + \"...\"\n",
        "    if len(text) <= limit:\n        return text\n    return text[:limit] + \"...\"\n",
        "truncate",
    ),
    "task_manager_reject_blank_title": (
        "src/task_manager/core.py",
        "    def add(self, title):\n        self.tasks.append({\"title\": title, \"done\": False})\n",
        "    def add(self, title):\n        if not isinstance(title, str) or not title.strip():\n            raise ValueError(\"title must not be blank\")\n        self.tasks.append({\"title\": title.strip(), \"done\": False})\n",
        "def add",
    ),
}

QUERY_ANSWERS = {
    "calculator_locate_multiply": "multiply is defined in src/calculator/core.py.",
    "calculator_explain_subtract": "subtract returns a - b.",
    "calculator_list_public_api": "The public functions are add, subtract, multiply and divide.",
    "text_utils_locate_slugify": "slugify is defined in src/text_utils/core.py.",
    "text_utils_explain_title_case": "title_case calls split and capitalize for each word.",
    "text_utils_list_exports": "The exports are slugify, title_case and truncate.",
    "task_manager_locate_pending": "pending is defined in src/task_manager/core.py.",
    "task_manager_explain_complete": "complete sets the selected task's done field to True.",
    "task_manager_list_methods": "TaskList exposes add, complete and pending.",
    "calculator_find_tests": "The calculator tests are in tests/test_core.py.",
    "text_utils_find_tests": "The text utility tests are in tests/test_core.py.",
    "calculator_dev_api": "The calculator package exports divide.",
    "text_utils_dev_behavior": "slugify calls strip and then replace for spaces.",
    "task_manager_dev_storage": "Tasks are dictionaries in a list with title and done fields.",
    "task_manager_test_pending_contract": "pending returns a list filtered by the done field.",
}


class ScriptedPolicy:
    """Deterministic policy used for CPU-only tests and reproducible benchmarks."""

    name = "scripted"
    model_name = "deterministic-script"

    def generate(self, state: AgentState, tools: list[dict]) -> AgentAction:
        patch = PATCHES.get(state.task_id)
        index = len(state.tool_history)
        if patch:
            path, old_text, new_text, query = patch
            actions = [
                AgentAction.tool("search_code", {"query": query}),
                AgentAction.tool("read_file", {"path": path, "start_line": 1, "end_line": 200}),
                AgentAction.tool("apply_patch", {"path": path, "old_text": old_text, "new_text": new_text}),
                AgentAction.tool("run_tests", {"command": "pytest -q", "timeout": 30}),
                AgentAction.tool("git_diff", {}),
                AgentAction.final("Implemented the requested minimal change and verified it with pytest."),
            ]
            return actions[min(index, len(actions) - 1)]
        return AgentAction.final(
            QUERY_ANSWERS.get(state.task_id, "Relevant implementation located and inspected.")
        )
