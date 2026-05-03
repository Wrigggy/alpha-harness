"""LLM judge implementation for OpenAI, DeepSeek, and Anthropic APIs."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

from .base import AlphaJudge, JudgeResult

_DEFAULT_TRANSLATE_PROMPT = (
    "You are a quantitative finance expert. Given a formulaic alpha expression, "
    "describe in plain English what economic signal or market phenomenon this factor captures.\n\n"
    "Expression: {expression}\n\n"
    "Provide a concise (2-3 sentence) description focusing on:\n"
    "1. What market behavior this captures (momentum, mean reversion, liquidity, etc.)\n"
    "2. The economic intuition for why this might predict returns\n"
    "3. The time horizon over which this signal operates\n\n"
    "Description:"
)

_DEFAULT_SCORE_PROMPT = (
    "You are a quantitative finance researcher evaluating whether a discovered alpha factor "
    "has genuine economic meaning or is likely a data-mining artifact.\n\n"
    "## Factor Description\n{nl_description}\n\n"
    "## Original Expression\n{expression}\n\n"
    "## Statistical Performance\n- Information Coefficient (IC): {ic}\n- Rank IC: {rank_ic}\n\n"
    "## Related Literature\n{paper_context}\n\n"
    "## Evaluation Criteria\n"
    "Rate this factor's interpretability from 0.0 to 1.0:\n"
    "- 0.0-0.2: No economic meaning, likely noise or data artifact\n"
    "- 0.2-0.4: Weak connection to known phenomena, questionable mechanism\n"
    "- 0.4-0.6: Plausible economic mechanism, partial literature support\n"
    "- 0.6-0.8: Clear economic logic, supported by related research\n"
    "- 0.8-1.0: Strong, well-documented economic phenomenon\n\n"
    "IMPORTANT: Do NOT penalize novelty. A factor that captures a real but undocumented phenomenon "
    "should score 0.4-0.6, not low. Only score below 0.2 for expressions that are clearly nonsensical "
    "(e.g., repeated nested operations with no meaning).\n\n"
    "Respond in this exact JSON format:\n"
    '{{\n    "score": <float 0-1>,\n    "narrative": "<2-3 sentence economic narrative>",\n'
    '    "reasoning": "<why this score>"\n}}'
)

try:
    from openai import OpenAI

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

try:
    import anthropic

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False


class ApiJudge(AlphaJudge):
    """Alpha judge that uses a configurable API provider."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-5-mini",
        max_tokens: int = 1024,
        translate_prompt_path: str | None = "prompts/translate.txt",
        score_prompt_path: str | None = "prompts/score.txt",
        base_url: str | None = None,
        api_key_env: str | None = None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_key_env = api_key_env
        self._cache: dict[str, JudgeResult] = {}
        self._translate_template = self._load_prompt(translate_prompt_path, _DEFAULT_TRANSLATE_PROMPT)
        self._score_template = self._load_prompt(score_prompt_path, _DEFAULT_SCORE_PROMPT)
        self._client = self._build_client()

        logger.info("ApiJudge initialized with provider={} model={}", self.provider, self.model)

    @staticmethod
    def _load_prompt(path: str | None, default: str) -> str:
        if path is None:
            return default
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return default

    def _resolve_api_key(self) -> str | None:
        self._load_local_dotenv()
        env_name = self.api_key_env
        if env_name:
            return os.getenv(env_name)
        if self.provider == "deepseek":
            return os.getenv("DEEPSEEK_API_KEY")
        if self.provider == "codex":
            return os.getenv("OPENAI_API_KEY") or os.getenv("CODEX_API_KEY")
        if self.provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        return os.getenv("OPENAI_API_KEY")

    @staticmethod
    def _load_local_dotenv() -> None:
        env_path = Path(".env")
        if not env_path.exists():
            return
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    def _build_client(self):
        api_key = self._resolve_api_key()
        if self.provider == "anthropic":
            if not _HAS_ANTHROPIC:
                raise RuntimeError("anthropic SDK is not installed. Install with: pip install anthropic")
            return anthropic.Anthropic(api_key=api_key)

        if self.provider == "deepseek":
            if _HAS_OPENAI:
                return OpenAI(
                    api_key=api_key,
                    base_url=self.base_url or "https://api.deepseek.com",
                )
            if not _HAS_REQUESTS:
                raise RuntimeError("requests is not installed. Cannot use DeepSeek HTTP fallback.")
            return {
                "api_key": api_key,
                "base_url": (self.base_url or "https://api.deepseek.com").rstrip("/"),
                "transport": "deepseek_http",
            }

        if self.provider == "codex":
            if not _HAS_OPENAI:
                raise RuntimeError("openai SDK is not installed. Install with: pip install openai")
            return OpenAI(
                api_key=api_key,
                base_url=self.base_url,
            )

        if not _HAS_OPENAI:
            raise RuntimeError("openai SDK is not installed. Install with: pip install openai")

        if self.provider == "openai":
            return OpenAI(
                api_key=api_key,
                base_url=self.base_url,
            )

        return OpenAI(api_key=api_key, base_url=self.base_url)

    def _call_model(self, prompt: str) -> str:
        if self.provider == "anthropic":
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            parts = [block.text for block in message.content if getattr(block, "type", "") == "text"]
            return "\n".join(parts)

        if isinstance(self._client, dict) and self._client.get("transport") == "deepseek_http":
            response = requests.post(
                f"{self._client['base_url']}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._client['api_key']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                },
                timeout=120,
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"]

        response = self._client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=self.max_tokens,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text

        parts: list[str] = []
        for item in getattr(response, "output", []):
            for content in getattr(item, "content", []):
                if getattr(content, "type", "") == "output_text":
                    parts.append(content.text)
        return "\n".join(parts)

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Could not parse JSON from response: {text[:200]}")

    def translate(self, expression: str) -> str:
        prompt = self._translate_template.format(expression=expression)
        logger.info("Translating expression: {}", expression[:80])
        return self._call_model(prompt).strip()

    def score(self, expression: str, ic: float, matched_papers: list[dict]) -> JudgeResult:
        if expression in self._cache:
            return self._cache[expression]

        nl_description = self.translate(expression)
        if matched_papers:
            paper_context = "\n".join(
                f"- **{p.get('title', 'Unknown')}**: {p.get('abstract', 'N/A')[:200]}"
                for p in matched_papers
            )
        else:
            paper_context = "No directly related papers found."

        rank_ic = ic * 0.85
        prompt = self._score_template.format(
            nl_description=nl_description,
            expression=expression,
            ic=f"{ic:.4f}",
            rank_ic=f"{rank_ic:.4f}",
            paper_context=paper_context,
        )

        logger.info("Scoring expression: {} (IC={:.4f})", expression[:80], ic)
        raw_response = self._call_model(prompt)
        try:
            parsed = self._parse_json_response(raw_response)
            interpretability_score = float(parsed.get("score", 0.0))
            narrative = parsed.get("narrative", "")
            reasoning = parsed.get("reasoning", "")
        except (ValueError, json.JSONDecodeError) as e:
            logger.error("Failed to parse scoring response: {}", e)
            interpretability_score = 0.0
            narrative = "Failed to parse LLM response."
            reasoning = f"Parse error: {e}"

        result = JudgeResult(
            expression=expression,
            nl_description=nl_description,
            interpretability_score=interpretability_score,
            economic_narrative=narrative,
            matched_papers=[p.get("title", "Unknown") for p in matched_papers],
            reasoning=reasoning,
        )
        self._cache[expression] = result
        return result
