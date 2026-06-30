"""Deterministic query planner for ontology-driven graph QA."""

from __future__ import annotations

import re

from ..domain.aliases import MATERIAL_ALIASES, PROPERTY_ALIASES, REGIME_ALIASES
from ..domain.normalization import canonical_material, canonical_property, canonical_regime, normalize_text
from ..domain.query_constraints import QueryConstraints, QueryIntent


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


class QueryPlanner:
    """Parse a raw user question into canonical graph constraints."""

    MATERIAL_PATTERNS = [
        re.compile(r"\b12[ХX]18[НH]10[ТT]\b", re.IGNORECASE),
        re.compile(r"\b09Г2С\b", re.IGNORECASE),
        re.compile(r"\b7075(?:-T6)?\b", re.IGNORECASE),
        re.compile(r"\bVT6\b", re.IGNORECASE),
        re.compile(r"\bВТ6\b", re.IGNORECASE),
        re.compile(r"\bTi-?6Al-?4V\b", re.IGNORECASE),
        re.compile(r"\b(?:сплав(?:у|а|ом|е)?|сталь|стали|alloy|steel)\s+(?P<name>[A-Za-zА-Яа-я0-9\-]+)\b", re.IGNORECASE),
    ]

    OBJECT_MARKERS = ["насос", "pump", "клапан", "valve", "dn50", "npk-200"]
    EQUIPMENT_MARKERS = ["оборудован", "установк", "печь", "прибор", "твердомер", "equipment"]
    TEAM_MARKERS = ["лаборатор", "команд", "группа", "laboratory", "team"]
    GAP_MARKERS = ["пробел", "нет данных", "не хватает", "не исслед", "gap"]
    HISTORY_MARKERS = ["история решений", "цепочка решений", "что пробовали", "историю решений"]

    def parse(self, question: str) -> QueryConstraints:
        """Return canonical constraints for the provided question."""
        raw_question = question or ""
        q = normalize_text(raw_question)
        materials = self._materials(raw_question)
        regimes = self._aliases_in_text(q, REGIME_ALIASES, canonical_regime)
        properties = self._aliases_in_text(q, PROPERTY_ALIASES, canonical_property)

        intent = QueryIntent.UNKNOWN
        if any(marker in q for marker in self.HISTORY_MARKERS):
            intent = QueryIntent.DECISION_HISTORY
        elif any(marker in q for marker in self.GAP_MARKERS):
            intent = QueryIntent.GAP_ANALYSIS
        elif materials and regimes and properties:
            intent = QueryIntent.MATERIAL_REGIME_PROPERTY_EFFECT
        elif any(marker in q for marker in self.EQUIPMENT_MARKERS):
            intent = QueryIntent.EQUIPMENT_USAGE
        elif any(marker in q for marker in self.TEAM_MARKERS):
            intent = QueryIntent.TEAM_ACTIVITY
        elif any(marker in q for marker in self.OBJECT_MARKERS):
            intent = QueryIntent.ENTITY_OVERVIEW

        return QueryConstraints(
            intent=intent,
            raw_question=raw_question,
            materials=materials,
            regimes=regimes,
            properties=properties,
            require_exact_match=bool(materials and regimes and properties),
        )

    def _materials(self, question: str) -> list[str]:
        values: list[str] = []
        for pattern in self.MATERIAL_PATTERNS:
            for match in pattern.finditer(question or ""):
                name = match.groupdict().get("name") or match.group(0)
                values.append(canonical_material(name))
        q = normalize_text(question)
        for alias in MATERIAL_ALIASES:
            if normalize_text(alias) in q:
                values.append(canonical_material(alias))
        return _unique(values)

    @staticmethod
    def _aliases_in_text(text: str, aliases: dict[str, str], canonicalizer) -> list[str]:
        values: list[str] = []
        compact = re.sub(r"[\s_\-]+", "", text)
        for alias in aliases:
            alias_norm = normalize_text(alias)
            alias_compact = re.sub(r"[\s_\-]+", "", alias_norm)
            if alias_norm in text or alias_compact in compact:
                values.append(canonicalizer(alias))
        return _unique(values)

