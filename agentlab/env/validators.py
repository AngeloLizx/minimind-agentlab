from __future__ import annotations

import re
from typing import Any


def validate_json_arguments(arguments: Any, schema: dict[str, Any]) -> list[str]:
    if not isinstance(arguments, dict):
        return ["arguments must be an object"]
    errors: list[str] = []
    properties = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in arguments:
            errors.append(f"missing required argument: {key}")
    extra = set(arguments) - set(properties)
    if extra and not schema.get("additionalProperties", False):
        errors.append(f"unknown arguments: {', '.join(sorted(extra))}")
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    for key, value in arguments.items():
        spec = properties.get(key)
        if not spec:
            continue
        expected = type_map.get(spec.get("type"))
        numeric_bool = spec.get("type") in {"integer", "number"} and isinstance(value, bool)
        if expected and (not isinstance(value, expected) or numeric_bool):
            errors.append(f"{key} must be {spec['type']}")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"{key} must be one of {spec['enum']}")
        if isinstance(value, int):
            if "minimum" in spec and value < spec["minimum"]:
                errors.append(f"{key} is below minimum")
            if "maximum" in spec and value > spec["maximum"]:
                errors.append(f"{key} is above maximum")
        if isinstance(value, str) and spec.get("pattern") and not re.search(spec["pattern"], value):
            errors.append(f"{key} has invalid format")
    return errors
