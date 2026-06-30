"""Small unit conversion helpers for user-facing comparison answers."""

from __future__ import annotations

from typing import Any


KSI_TO_MPA = 6.894757


def normalize_unit_label(unit: str | None) -> str:
    """Return a canonical display label for common strength units."""

    raw = str(unit or "").strip()
    key = raw.lower().replace("м", "m").replace("а", "a")
    if key in {"mpa", "mпa", "мpa"} or raw in {"МПа", "мПа"}:
        return "MPa"
    if key == "ksi":
        return "ksi"
    return raw


def normalize_strength_to_mpa(value: Any, unit: str | None) -> tuple[float | None, str | None]:
    """Convert a strength value to MPa when supported.

    Returns `(converted_value, note)`. `note` is populated only when the
    original value was converted from another unit.
    """

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None, None
    normalized_unit = normalize_unit_label(unit)
    if normalized_unit == "MPa":
        return numeric, None
    if normalized_unit == "ksi":
        converted = numeric * KSI_TO_MPA
        return converted, f"{numeric:g} ksi ≈ {converted:.0f} MPa"
    return None, None
