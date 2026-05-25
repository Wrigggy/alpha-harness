"""Unified LLM client dispatcher.

Pick a backend with `get_llm_client(backend=...)` or via the `LLM_BACKEND`
env var (fallback). Three backends:

  claude_code  — claude_agent_sdk via Max-plan subscription (default; no key)
  openrouter   — OpenAI-compatible API at openrouter.ai (needs OPENROUTER_API_KEY).
                 The OpenRouter client accepts a `provider` kwarg to pin routing
                 to a specific upstream (e.g. provider="DeepSeek" for the
                 official DeepSeek inference endpoint instead of Fireworks/etc.).
  anthropic    — anthropic SDK direct (needs ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import os
from typing import Optional

from .base import LLMClient


_DEFAULT_MODELS = {
    "claude_code": "claude-opus-4-7",
    "openrouter": "deepseek/deepseek-v4-pro",
    "anthropic": "claude-opus-4-7",
}


def resolve_backend(backend: Optional[str] = None) -> str:
    """CLI flag > env var > 'claude_code'."""
    if backend:
        return backend
    return os.environ.get("LLM_BACKEND", "claude_code")


def get_llm_client(
    backend: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    backend = resolve_backend(backend)
    model = model or _DEFAULT_MODELS.get(backend)
    if backend == "claude_code":
        from .claude_code import ClaudeCodeClient
        return ClaudeCodeClient(model=model, **kwargs)
    if backend == "openrouter":
        from .openrouter import OpenRouterClient
        return OpenRouterClient(model=model, **kwargs)
    if backend == "anthropic":
        from .anthropic_api import AnthropicClient
        return AnthropicClient(model=model, **kwargs)
    raise ValueError(
        f"Unknown LLM backend {backend!r}. Choices: claude_code, openrouter, anthropic"
    )


__all__ = ["LLMClient", "get_llm_client", "resolve_backend"]
