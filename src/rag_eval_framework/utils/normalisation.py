"""Shared score and text normalisation helpers."""

from __future__ import annotations

import re
import unicodedata


def normalise_for_match(text: str) -> str:
    """Normalise text for lenient matching of labels against retrieved text.

    Applies NFKC (full-width → half-width, etc.), lower-cases, and removes all
    whitespace, since PDF / OCR extraction often breaks lines mid-sentence.

    Examples
    --------
    >>> normalise_for_match("１２４ 円")
    '124円'
    """
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).lower()


def normalise_1_5(value: float) -> float:
    """Map a 1-5 score (as returned by Azure AI Evaluation SDK) to 0.0-1.0.

    Values outside the [1, 5] range are clamped.

    Examples
    --------
    >>> normalise_1_5(1)
    0.0
    >>> normalise_1_5(5)
    1.0
    >>> normalise_1_5(3)
    0.5
    """
    return round(max(0.0, min(1.0, (value - 1.0) / 4.0)), 4)
