from __future__ import annotations

import re
import unicodedata

_DASH_TRANSLATION = str.maketrans(
    {
        "\u2212": "-",  # minus sign
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
    }
)


def normalize_inline_text(value: str, *, normalize_decimal_comma: bool = False) -> str:
    """
    Normalize OCR/PDF text for robust regex and LLM extraction.
    """
    if not value:
        return ""
    text = _base_normalize(value)
    text = re.sub(r"\s+", " ", text).strip()
    if normalize_decimal_comma:
        text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return text


def normalize_block_text(value: str, *, normalize_decimal_comma: bool = False) -> str:
    """
    Normalize while preserving line boundaries (useful during PDF ingestion).
    """
    if not value:
        return ""
    text = _base_normalize(value)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if normalize_decimal_comma:
        text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return text


def normalize_for_match(value: str) -> str:
    text = normalize_inline_text(value, normalize_decimal_comma=True).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _base_normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", value.replace("\x00", " "))
    return text.translate(_DASH_TRANSLATION)
