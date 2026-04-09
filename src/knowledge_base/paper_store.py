"""Paper corpus manager — load, save, and query summarized research papers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class PaperSummary:
    id: str
    title: str
    factor_type: list[str]  # e.g., ["momentum", "mean_reversion"]
    mechanism: str  # one-sentence economic intuition
    asset_class: list[str]  # e.g., ["equity", "crypto"]
    frequency: list[str]  # e.g., ["intraday", "daily"]
    decay_horizon: str  # e.g., "3-5 days"
    key_finding: str  # one-sentence empirical result
    source: str  # "arxiv", "ssrn", "journal"
    url: str = ""


class PaperStore:
    """Manages a JSON-backed corpus of paper summaries."""

    def __init__(self, corpus_path: str = "papers/corpus.json") -> None:
        self.corpus_path = Path(corpus_path)
        self._papers: list[PaperSummary] = []
        if self.corpus_path.exists():
            self._papers = self.load()
            logger.info("Loaded {} papers from {}", len(self._papers), self.corpus_path)
        else:
            logger.warning("Corpus file not found: {}", self.corpus_path)

    def load(self) -> list[PaperSummary]:
        """Load papers from the JSON corpus file."""
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        papers = [PaperSummary(**entry) for entry in raw]
        self._papers = papers
        return papers

    def save(self, papers: list[PaperSummary]) -> None:
        """Overwrite the corpus file with the given papers."""
        self.corpus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.corpus_path, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in papers], f, indent=2, ensure_ascii=False)
        self._papers = papers
        logger.info("Saved {} papers to {}", len(papers), self.corpus_path)

    def add_paper(self, paper: PaperSummary) -> None:
        """Append a paper to the corpus (skips if id already exists)."""
        if self.get_by_id(paper.id) is not None:
            logger.warning("Paper '{}' already exists, skipping", paper.id)
            return
        self._papers.append(paper)
        self.save(self._papers)

    def get_by_id(self, paper_id: str) -> PaperSummary | None:
        """Return a paper by its id, or None if not found."""
        for p in self._papers:
            if p.id == paper_id:
                return p
        return None

    @property
    def papers(self) -> list[PaperSummary]:
        return list(self._papers)

    def list_factor_types(self) -> list[str]:
        """Return all unique factor_type tags across the corpus."""
        types: set[str] = set()
        for p in self._papers:
            types.update(p.factor_type)
        return sorted(types)
