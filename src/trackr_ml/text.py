from __future__ import annotations

import re
import unicodedata

MULTISPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""

    normalized = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char)
    )
    collapsed = MULTISPACE_PATTERN.sub(" ", without_accents.lower())
    return collapsed.strip()


def compact_whitespace(text: str) -> str:
    return MULTISPACE_PATTERN.sub(" ", text).strip()
