from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class JudgeResult:
    expression: str
    nl_description: str
    interpretability_score: float  # 0-1
    economic_narrative: str
    matched_papers: list[str] = field(default_factory=list)
    reasoning: str = ""


class AlphaJudge(ABC):
    @abstractmethod
    def translate(self, expression: str) -> str:
        """Convert expression tree string to natural language."""

    @abstractmethod
    def score(self, expression: str, ic: float, matched_papers: list[dict]) -> JudgeResult:
        """Score a candidate alpha for economic interpretability."""

    def batch_score(self, candidates: list[dict]) -> list[JudgeResult]:
        """Score multiple candidates. Default: sequential calls to score()."""
        results = []
        for c in candidates:
            results.append(
                self.score(c["expression"], c["ic"], c.get("matched_papers", []))
            )
        return results
