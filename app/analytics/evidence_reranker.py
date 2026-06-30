"""Heuristic evidence reranking for constrained analytical questions."""

from __future__ import annotations

from ..domain.aliases import MATERIAL_ALIASES
from ..domain.normalization import normalize_text
from ..domain.query_constraints import QueryConstraints
from .query_models import EvidenceItem


class EvidenceReranker:
    """Boost evidence that matches query constraints and penalize conflicts."""

    def rerank(self, items: list[EvidenceItem], constraints: QueryConstraints) -> list[EvidenceItem]:
        reranked: list[EvidenceItem] = []
        for item in items:
            score = item.score + self._constraint_score(item.quote, constraints)
            if len(item.quote.strip()) < 40:
                score -= 0.20
            reranked.append(item.model_copy(update={"score": max(0.0, min(1.0, score))}))
        reranked.sort(key=lambda value: value.score, reverse=True)
        return reranked

    def _constraint_score(self, quote: str, constraints: QueryConstraints) -> float:
        text = normalize_text(quote)
        score = 0.0
        if constraints.materials and any(normalize_text(item) in text for item in constraints.materials):
            score += 0.35
        if constraints.regimes and any(normalize_text(item) in text for item in constraints.regimes):
            score += 0.25
        if constraints.properties and any(normalize_text(item) in text for item in constraints.properties):
            score += 0.25
        if constraints.equipment and any(normalize_text(item) in text for item in constraints.equipment):
            score += 0.10
        if constraints.laboratories and any(normalize_text(item) in text for item in constraints.laboratories):
            score += 0.10
        if constraints.materials and _has_conflicting_material(text, constraints.materials):
            score -= 0.30
        return score


def _has_conflicting_material(text: str, expected: list[str]) -> bool:
    expected_norm = {normalize_text(item) for item in expected}
    canonical_values = {normalize_text(value) for value in MATERIAL_ALIASES.values()}
    for value in canonical_values:
        if value in text and value not in expected_norm:
            return True
    return False
