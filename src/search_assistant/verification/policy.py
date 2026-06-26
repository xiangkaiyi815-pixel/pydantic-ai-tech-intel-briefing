from __future__ import annotations

import re

from search_assistant.contracts import Classification


_FRESHNESS_WORDS = re.compile(
    r"\b(latest|current|today|yesterday|tomorrow|recent|newest|now|202[0-9])\b",
    re.IGNORECASE,
)
_VERSION_OR_NUMBER = re.compile(r"\b\d+(?:\.\d+){1,}\b|\b\d{4}\b|[$¥€]\s?\d+|\b\d+(?:\.\d+)?%")
_HIGH_STAKES = re.compile(
    r"\b(medical|legal|financial|finance|security|safety|law|tax|investment|doctor|medicine)\b",
    re.IGNORECASE,
)
_API_OR_POLICY = re.compile(r"\b(api|sdk|policy|price|schedule|ranking|benchmark|version)\b", re.IGNORECASE)


def requires_verification(text: str, classification: Classification) -> bool:
    if classification in {"research", "hard", "high_stakes"}:
        return True
    return any(
        pattern.search(text)
        for pattern in (_FRESHNESS_WORDS, _VERSION_OR_NUMBER, _HIGH_STAKES, _API_OR_POLICY)
    )


def requires_calibration(classification: Classification, has_unverified_claims: bool) -> bool:
    return classification in {"research", "hard", "high_stakes"} or has_unverified_claims


def extract_key_claims(text: str) -> list[str]:
    claims: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        cleaned = sentence.strip().rstrip(".!?")
        if cleaned and (
            _FRESHNESS_WORDS.search(cleaned)
            or _VERSION_OR_NUMBER.search(cleaned)
            or _HIGH_STAKES.search(cleaned)
            or _API_OR_POLICY.search(cleaned)
        ):
            claims.append(cleaned)
    return claims
