"""DOI resolution via Crossref.

When the ChatGPT extraction step does not find a DOI, or when we want to
double-check, we query Crossref with (title, first author, year) to get
a validated DOI.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import requests

CROSSREF_BASE = "https://api.crossref.org"
CROSSREF_TIMEOUT = 15


def resolve_doi(
    title: str,
    first_author_last_name: Optional[str] = None,
    year: Optional[int] = None,
    timeout: int = CROSSREF_TIMEOUT,
) -> Optional[str]:
    """Try to resolve a DOI from Crossref using bibliographic metadata.

    Returns a DOI string (e.g. ``10.1234/abc``) or *None*.
    """
    if not title or len(title.strip()) < 10:
        return None
    if title.strip().lower() in ("unknown", "untitled", "n/a", "sans titre"):
        return None

    query = title.strip()
    params: Dict[str, str] = {
        "query.bibliographic": query,
        "rows": "5",
    }
    if first_author_last_name:
        params["query.author"] = first_author_last_name
    headers = {
        "User-Agent": "Impact-Pipeline/1.0 (mailto:contact@impact-tool.org)",
    }
    try:
        resp = requests.get(
            f"{CROSSREF_BASE}/works",
            params=params,
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    items = data.get("message", {}).get("items", [])
    if not isinstance(items, list) or not items:
        return None

    norm_title = _normalize(title)
    best_doi: Optional[str] = None
    best_score = -1.0
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        titles = item.get("title", [])
        if isinstance(titles, list) and titles:
            cand_title = str(titles[0])
        else:
            continue
        score = _overlap(norm_title, _normalize(cand_title))
        # Year bonus
        pub_year = _extract_year(item)
        if isinstance(year, int) and pub_year == year:
            score += 0.15
        if score > best_score:
            best_score = score
            doi_val = item.get("DOI")
            if isinstance(doi_val, str) and doi_val.strip():
                best_doi = doi_val.strip()
                best_score = score

    return best_doi if best_score >= 0.40 else None


def validate_doi(doi: str, timeout: int = CROSSREF_TIMEOUT) -> bool:
    """Check if a DOI resolves successfully via Crossref."""
    if not doi:
        return False
    normalized = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    try:
        resp = requests.get(
            f"{CROSSREF_BASE}/works/{normalized}",
            headers={"User-Agent": "Impact-Pipeline/1.0"},
            timeout=timeout,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_year(item: Dict[str, Any]) -> Optional[int]:
    """Extract publication year from a Crossref work item."""
    published = item.get("published-print") or item.get("published-online") or item.get("issued")
    if isinstance(published, dict):
        parts = published.get("date-parts", [[]])
        if isinstance(parts, list) and parts:
            first = parts[0]
            if isinstance(first, list) and first:
                try:
                    return int(first[0])
                except (ValueError, TypeError):
                    pass
    return None


def _normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    ta = {t for t in a.split() if len(t) > 2}
    tb = {t for t in b.split() if len(t) > 2}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), 1)
