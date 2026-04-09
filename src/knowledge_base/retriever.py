"""Tag-based paper retrieval with overlap scoring."""

from __future__ import annotations

from loguru import logger

from src.knowledge_base.paper_store import PaperStore, PaperSummary


class PaperRetriever:
    """Retrieve papers matching given tags, ranked by overlap score."""

    def __init__(self, store: PaperStore) -> None:
        self.store = store

    def retrieve(
        self,
        factor_type: list[str] | None = None,
        asset_class: str | None = None,
        frequency: str | None = None,
        top_k: int = 5,
    ) -> list[PaperSummary]:
        """Retrieve papers matching the given tags, ranked by overlap score.

        Overlap score = (number of matching tags) / (number of query tags).
        Tags are matched across factor_type, asset_class, and frequency fields.
        """
        query_tags = self._build_query_tags(factor_type, asset_class, frequency)
        if not query_tags:
            logger.warning("Empty query — returning first {} papers", top_k)
            return self.store.papers[:top_k]

        scored: list[tuple[float, PaperSummary]] = []
        for paper in self.store.papers:
            paper_tags = self._extract_tags(paper)
            overlap = len(query_tags & paper_tags)
            if overlap > 0:
                score = overlap / len(query_tags)
                scored.append((score, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [paper for _, paper in scored[:top_k]]
        logger.debug(
            "Retrieved {} papers for query tags {} (top_k={})",
            len(results),
            query_tags,
            top_k,
        )
        return results

    @staticmethod
    def _build_query_tags(
        factor_type: list[str] | None,
        asset_class: str | None,
        frequency: str | None,
    ) -> set[str]:
        tags: set[str] = set()
        if factor_type:
            tags.update(f"factor:{t}" for t in factor_type)
        if asset_class:
            tags.add(f"asset:{asset_class}")
        if frequency:
            tags.add(f"freq:{frequency}")
        return tags

    @staticmethod
    def _extract_tags(paper: PaperSummary) -> set[str]:
        tags: set[str] = set()
        tags.update(f"factor:{t}" for t in paper.factor_type)
        tags.update(f"asset:{a}" for a in paper.asset_class)
        tags.update(f"freq:{f}" for f in paper.frequency)
        return tags
