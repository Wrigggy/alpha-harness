"""Anthropic API direct backend — uses the anthropic SDK with ANTHROPIC_API_KEY."""

from __future__ import annotations

from typing import Optional


class AnthropicClient:
    backend = "anthropic"

    def __init__(self, model: str = "claude-opus-4-7") -> None:
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. `pip install anthropic`"
            ) from e
        self.model = model
        self._client = anthropic.Anthropic()

    def complete(
        self,
        prompt: str,
        max_tokens: int = 4096,
        seed: Optional[int] = None,  # ignored: anthropic SDK has no seed param
    ) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in msg.content if b.type == "text"]
        if not parts:
            raise RuntimeError("Empty response from anthropic API")
        return "\n".join(parts)
