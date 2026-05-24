"""Unified LLM judge — backend-agnostic, takes an injected LLMClient."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from src.llm_client import LLMClient, get_llm_client

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


class LLMJudge(AlphaJudge):
    """Alpha judge backed by an injected LLMClient.

    Construct directly with `LLMJudge(client=...)`, or via the convenience
    classmethod `LLMJudge.from_backend(backend='openrouter', model=...)`.
    """

    def __init__(
        self,
        client: LLMClient,
        translate_prompt_path: str | None = "prompts/translate.txt",
        score_prompt_path: str | None = "prompts/score.txt",
    ) -> None:
        self.client = client
        self.model = client.model  # back-compat: callers may read judge.model
        self._cache: dict[str, JudgeResult] = {}
        self._translate_template = self._load_prompt(translate_prompt_path, _DEFAULT_TRANSLATE_PROMPT)
        self._score_template = self._load_prompt(score_prompt_path, _DEFAULT_SCORE_PROMPT)
        logger.info(
            "LLMJudge initialized: backend={} model={}",
            client.backend, client.model,
        )

    @classmethod
    def from_backend(
        cls,
        backend: str | None = None,
        model: str | None = None,
        translate_prompt_path: str | None = "prompts/translate.txt",
        score_prompt_path: str | None = "prompts/score.txt",
        **client_kwargs,
    ) -> "LLMJudge":
        return cls(
            client=get_llm_client(backend=backend, model=model, **client_kwargs),
            translate_prompt_path=translate_prompt_path,
            score_prompt_path=score_prompt_path,
        )

    @staticmethod
    def _load_prompt(path: str | None, default: str) -> str:
        if path is None:
            return default
        p = Path(path)
        if p.exists():
            return p.read_text(encoding="utf-8")
        logger.debug("Prompt file {} not found, using built-in default", path)
        return default

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
        return self.client.complete(prompt, max_tokens=1024).strip()

    def score(self, expression: str, ic: float, matched_papers: list[dict]) -> JudgeResult:
        if expression in self._cache:
            logger.info("Cache hit: {}", expression[:80])
            return self._cache[expression]

        nl_description = self.translate(expression)

        if matched_papers:
            paper_lines = []
            for p in matched_papers:
                title = p.get("title", "Unknown")
                abstract = p.get("abstract", "N/A")
                paper_lines.append(f"- **{title}**: {abstract[:200]}")
            paper_context = "\n".join(paper_lines)
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
        raw_response = self.client.complete(prompt, max_tokens=1024)

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
        logger.info("Scored: {} -> {:.2f}", expression[:80], interpretability_score)
        return result
