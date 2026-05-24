"""claude_agent_sdk backend — uses Max-plan subscription, no API key."""

from __future__ import annotations

from typing import Optional


class ClaudeCodeClient:
    backend = "claude_code"

    def __init__(self, model: str = "claude-opus-4-7") -> None:
        self.model = model

    def complete(
        self,
        prompt: str,
        max_tokens: int = 4096,  # ignored; the SDK manages it
        seed: Optional[int] = None,
    ) -> str:
        import anyio
        from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

        full = prompt if seed is None else f"{prompt}\n\n[generation_seed_hint={seed}]"

        async def _call() -> str:
            result = ""
            async for message in query(
                prompt=full,
                options=ClaudeAgentOptions(
                    model=self.model, allowed_tools=[], max_turns=1
                ),
            ):
                if isinstance(message, ResultMessage):
                    result = message.result
            if not result:
                raise RuntimeError("Empty response from claude_agent_sdk")
            return result

        return anyio.run(_call)
