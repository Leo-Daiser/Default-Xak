"""Canonical resolver for extracted raw names and units."""

from __future__ import annotations

import re

from ..domain.normalization import canonical_material, canonical_property, canonical_regime


UNIT_ALIASES = {
    "mpa": "MPa",
    "мпа": "MPa",
    "мПа".lower(): "MPa",
    "gpa": "GPa",
    "гпа": "GPa",
    "hv": "HV",
    "hrc": "HRC",
    "%": "%",
    "°c": "C",
    "°с": "C",
    "c": "C",
    "с": "C",
    "celsius": "C",
    "сelsius": "C",
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "ч": "h",
    "ч.": "h",
    "min": "min",
    "mins": "min",
    "minute": "min",
    "minutes": "min",
    "мин": "min",
    "мин.": "min",
}

EXPERIMENT_ID_RE = re.compile(r"\b(?:E\d+|EXP-[A-ZА-Я0-9_.-]+)\b", re.IGNORECASE)


def clean_raw(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip(" .;|\"'«»"))


def resolve_material(raw: str | None) -> str:
    value = clean_raw(raw)
    if EXPERIMENT_ID_RE.fullmatch(value):
        return ""
    return canonical_material(value)


def resolve_regime(raw: str | None) -> str:
    return canonical_regime(clean_raw(raw))


def resolve_property(raw: str | None) -> str:
    return canonical_property(clean_raw(raw))


def resolve_unit(raw: str | None) -> str | None:
    value = clean_raw(raw).replace("°", "°").lower()
    value = value.replace(" ", "")
    return UNIT_ALIASES.get(value, clean_raw(raw) or None)
