"""LLM judge implementation using claude-agent-sdk (Max plan)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from .base import AlphaJudge, JudgeResult

# Default prompt templates (used when prompt files are not found)
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

# Attempt to import claude_agent_sdk
try:
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    _HAS_AGENT_SDK = True
except ImportError:
    _HAS_AGENT_SDK = False
    logger.warning(
        "claude-agent-sdk not installed. ClaudeAgentJudge will raise on use. "
        "Install with: pip install claude-agent-sdk"
    )

try:
    import anyio

    _HAS_ANYIO = True
except ImportError:
    _HAS_ANYIO = False


class ClaudeAgentJudge(AlphaJudge):
    """Alpha judge that uses claude-agent-sdk (Max plan) for scoring."""

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        translate_prompt_path: str | None = "prompts/translate.txt",
        score_prompt_path: str | None = "prompts/score.txt",
    ) -> None:
        self.model = model
        self._cache: dict[str, JudgeResult] = {}
        self._translate_template = self._load_prompt(
            translate_prompt_path, _DEFAULT_TRANSLATE_PROMPT
        )
        self._score_template = self._load_prompt(
            score_prompt_path, _DEFAULT_SCORE_PROMPT
        )

        if not _HAS_AGENT_SDK:
            logger.error(
                "claude-agent-sdk is not installed. ClaudeAgentJudge will not function."
            )
        if not _HAS_ANYIO:
            logger.error(
                "anyio is not installed. ClaudeAgentJudge will not function."
            )

    @staticmethod
    def _load_prompt(path: str | None, default: str) -> str:
        """Load a prompt template from file, falling back to the built-in default."""
        if path is None:
            return default
        p = Path(path)
        if p.exists():
            logger.debug("Loaded prompt template from {}", path)
            return p.read_text(encoding="utf-8")
        logger.debug("Prompt file {} not found, using built-in default", path)
        return default

    async def _call_claude_async(self, prompt: str) -> str:
        """Send a prompt to Claude via the agent SDK and return the result text."""
        if not _HAS_AGENT_SDK:
            raise RuntimeError(
                "claude-agent-sdk is not installed. "
                "Install with: pip install claude-agent-sdk"
            )

        result = ""
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                model=self.model,
                allowed_tools=[],  # no tools needed for scoring
                max_turns=1,
            ),
        ):
            if isinstance(message, ResultMessage):
                result = message.result
        return result

    def _call_claude(self, prompt: str) -> str:
        """Synchronous wrapper around the async agent SDK call."""
        if not _HAS_ANYIO:
            raise RuntimeError(
                "anyio is not installed. Install with: pip install anyio"
            )
        return anyio.run(self._call_claude_async, prompt)

    @staticmethod
    def _parse_json_response(text: str) -> dict[str, Any]:
        """Extract JSON from a response that may contain markdown code blocks."""
        # Try to extract from markdown code block first
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

        # Try parsing directly
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last resort: find the first { ... } block
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Could not parse JSON from response: {text[:200]}")

    def translate(self, expression: str) -> str:
        """Convert expression tree string to natural language."""
        prompt = self._translate_template.format(expression=expression)
        logger.info("Translating expression: {}", expression[:80])
        result = self._call_claude(prompt)
        logger.debug("Translation result: {}", result[:120])
        return result.strip()

    def score(
        self, expression: str, ic: float, matched_papers: list[dict]
    ) -> JudgeResult:
        """Score a candidate alpha for economic interpretability."""
        # Check cache
        if expression in self._cache:
            logger.info("Cache hit for expression: {}", expression[:80])
            return self._cache[expression]

        # Step 1: Translate expression to NL
        nl_description = self.translate(expression)

        # Step 2: Build paper context string
        if matched_papers:
            paper_lines = []
            for p in matched_papers:
                title = p.get("title", "Unknown")
                abstract = p.get("abstract", "N/A")
                paper_lines.append(f"- **{title}**: {abstract[:200]}")
            paper_context = "\n".join(paper_lines)
        else:
            paper_context = "No directly related papers found."

        # Step 3: Format scoring prompt
        rank_ic = ic * 0.85  # approximate rank IC if not provided
        prompt = self._score_template.format(
            nl_description=nl_description,
            expression=expression,
            ic=f"{ic:.4f}",
            rank_ic=f"{rank_ic:.4f}",
            paper_context=paper_context,
        )

        logger.info("Scoring expression: {} (IC={:.4f})", expression[:80], ic)
        raw_response = self._call_claude(prompt)

        # Step 4: Parse the JSON response
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

        # Step 5: Build result
        paper_titles = [p.get("title", "Unknown") for p in matched_papers]
        result = JudgeResult(
            expression=expression,
            nl_description=nl_description,
            interpretability_score=interpretability_score,
            economic_narrative=narrative,
            matched_papers=paper_titles,
            reasoning=reasoning,
        )

        # Cache the result
        self._cache[expression] = result
        logger.info(
            "Scored expression: {} -> {:.2f}",
            expression[:80],
            interpretability_score,
        )
        return result
