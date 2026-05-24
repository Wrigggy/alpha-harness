"""Shared LLMClient protocol.

`.complete(prompt, max_tokens, seed)` is the only required method — every
call site in this codebase is single-turn, no tool use, no streaming.
"""

from __future__ import annotations

from typing import Optional, Protocol


class LLMClient(Protocol):
    backend: str
    model: str

    def complete(
        self,
        prompt: str,
        max_tokens: int = 4096,
        seed: Optional[int] = None,
    ) -> str:
        ...
