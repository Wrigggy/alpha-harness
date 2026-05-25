"""OpenRouter backend — uses the openai SDK pointed at openrouter.ai.

Reads `OPENROUTER_API_KEY` from the environment. The OpenRouter base_url is
OpenAI-compatible so the same SDK works.

`provider` pins the upstream inference provider via OpenRouter's provider
routing (https://openrouter.ai/docs/provider-routing). For DeepSeek models,
set provider="DeepSeek" to route to DeepSeek's official endpoint rather than
the cheapest re-host (Fireworks, Together, etc.).
"""

from __future__ import annotations

import os
from typing import Optional


class OpenRouterClient:
    backend = "openrouter"
    base_url = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        model: str = "deepseek/deepseek-v4-pro",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        provider: Optional[str] = None,
        allow_fallbacks: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai package not installed. `pip install openai>=1.0`"
            ) from e
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. `export OPENROUTER_API_KEY=...`"
            )
        self.model = model
        self.provider = provider or os.environ.get("OPENROUTER_PROVIDER")
        self.allow_fallbacks = allow_fallbacks
        self._client = OpenAI(api_key=key, base_url=self.base_url, timeout=timeout)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 4096,
        seed: Optional[int] = None,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if self.provider:
            kwargs["extra_body"] = {
                "provider": {
                    "order": [self.provider],
                    "allow_fallbacks": self.allow_fallbacks,
                }
            }
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = choice.message.content or ""
        if not content:
            raise RuntimeError(
                f"Empty response from OpenRouter ({self.model}, "
                f"finish_reason={choice.finish_reason})"
            )
        return content
