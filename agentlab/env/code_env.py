from __future__ import annotations

from dataclasses import dataclass

from .sandbox import Sandbox


@dataclass(slots=True)
class CodeEnvironment:
    sandbox: Sandbox

    @property
    def repo_path(self) -> str:
        if self.sandbox.root is None:
            raise RuntimeError("Sandbox is not active")
        return str(self.sandbox.root)
