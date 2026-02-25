"""OpenAlex API client for article, source and author metadata.

This module provides a clean interface to OpenAlex endpoints with
proper error handling, timeout management and polite-pool support.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests

OPENALEX_BASE = "https://api.openalex.org"
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "").strip()
OPENALEX_TIMEOUT = 20


# ------------------------------------------------------------------
# Low-level helpers
# ------------------------------------------------------------------

def oa_get(path: str, params: Optional[Dict[str, str]] = None, timeout: int = OPENALEX_TIMEOUT) -> Optional[Dict[str, Any]]:
    """Generic OpenAlex GET with polite-pool and error handling."""
    url = f"{OPENALEX_BASE}{path}" if path.startswith("/") else path
    p: Dict[str, str] = dict(params or {})
    email = OPENALEX_EMAIL or os.getenv("OPENALEX_EMAIL", "").strip()
    if email:
        p["mailto"] = email
    try:
        resp = requests.get(url, params=p, timeout=timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


# ------------------------------------------------------------------
# Works
# ------------------------------------------------------------------

def get_work_by_doi(doi: str) -> Optional[Dict[str, Any]]:
    """Fetch an OpenAlex work by DOI.

    Normalizes the DOI (strips https://doi.org/ prefix) before querying.
    Returns the full work JSON or None if not found.
    """
    if not doi:
        return None
    normalized = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not normalized:
        return None
    return oa_get(f"/works/https://doi.org/{quote_plus(normalized)}")


def search_work_by_title(
    title: str,
    first_author_last_name: Optional[str] = None,
    year: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Search OpenAlex works by title.  Optionally filter by first-author
    last name and publication year.  Returns the best match or None."""
    if not title or len(title) < 10:
        return None
    if title.strip().lower() in ("unknown", "untitled", "n/a", "sans titre"):
        return None
    params: Dict[str, str] = {"search": title, "per_page": "5"}
    filters: List[str] = []
    if isinstance(year, int):
        filters.append(f"publication_year:{year}")
    if filters:
        params["filter"] = ",".join(filters)

    data = oa_get("/works", params)
    if not data:
        return None
    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        return None

    # Score candidates
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0
    norm_title = _normalize(title)
    for work in results:
        if not isinstance(work, dict):
            continue
        cand = _normalize(str(work.get("display_name", "")))
        score = _overlap(norm_title, cand)
        if isinstance(year, int) and work.get("publication_year") == year:
            score += 0.15
        if first_author_last_name:
            authorships = work.get("authorships", [])
            if isinstance(authorships, list) and authorships:
                first_auth = authorships[0]
                if isinstance(first_auth, dict):
                    author_obj = first_auth.get("author", {})
                    if isinstance(author_obj, dict):
                        name = _normalize(str(author_obj.get("display_name", "")))
                        if _normalize(first_author_last_name) in name:
                            score += 0.20
        if score > best_score:
            best_score = score
            best = work
    return best if best_score >= 0.35 else None


# ------------------------------------------------------------------
# Sources (journals)
# ------------------------------------------------------------------

def get_source_by_id(source_id: str) -> Optional[Dict[str, Any]]:
    """Fetch an OpenAlex source (journal/venue) by its OpenAlex ID.

    Accepts full URL (https://openalex.org/S123) or short id (S123).
    """
    if not source_id:
        return None
    if source_id.startswith("https://openalex.org/"):
        short = source_id.replace("https://openalex.org/", "")
    else:
        short = source_id
    return oa_get(f"/sources/{short}")


# ------------------------------------------------------------------
# Authors
# ------------------------------------------------------------------

def get_author_by_id(author_id: str) -> Optional[Dict[str, Any]]:
    """Fetch an OpenAlex author by ID."""
    if not author_id:
        return None
    if author_id.startswith("https://openalex.org/"):
        short = author_id.replace("https://openalex.org/", "")
    else:
        short = author_id
    return oa_get(f"/authors/{short}")


# ------------------------------------------------------------------
# Utility: extract structured info from work payload
# ------------------------------------------------------------------

def extract_source_id_from_work(work: Dict[str, Any]) -> Optional[str]:
    """Extract the source (journal) OpenAlex ID from a work payload."""
    loc = work.get("primary_location", {})
    if not isinstance(loc, dict):
        return None
    src = loc.get("source", {})
    if not isinstance(src, dict):
        return None
    return src.get("id")


def extract_author_ids_from_work(work: Dict[str, Any], positions: str = "first_last") -> List[str]:
    """Return author OpenAlex ids.

    positions: 'first_last' (default) returns first + last author,
               'all' returns everyone.
    """
    authorships = work.get("authorships", [])
    if not isinstance(authorships, list) or not authorships:
        return []
    if positions == "all":
        ids = []
        for a in authorships:
            if isinstance(a, dict):
                auth = a.get("author", {})
                if isinstance(auth, dict) and auth.get("id"):
                    ids.append(auth["id"])
        return ids
    # first + last
    ids = []
    first = authorships[0]
    if isinstance(first, dict):
        auth = first.get("author", {})
        if isinstance(auth, dict) and auth.get("id"):
            ids.append(auth["id"])
    if len(authorships) > 1:
        last = authorships[-1]
        if isinstance(last, dict):
            auth = last.get("author", {})
            if isinstance(auth, dict) and auth.get("id"):
                aid = auth["id"]
                if aid not in ids:
                    ids.append(aid)
    return ids


def extract_concepts_from_work(work: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return concepts/topics from a work, sorted by score desc."""
    # OpenAlex v2 uses "topics" but older data may have "concepts"
    concepts = work.get("concepts", [])
    if not isinstance(concepts, list):
        concepts = []
    # Also try topics
    topics = work.get("topics", [])
    if isinstance(topics, list) and topics:
        merged = list(concepts)
        for t in topics:
            if isinstance(t, dict):
                merged.append(t)
        concepts = merged
    # Sort by score
    scored = []
    for c in concepts:
        if not isinstance(c, dict):
            continue
        s = c.get("score", 0.0)
        if not isinstance(s, (int, float)):
            s = 0.0
        scored.append((s, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored]


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

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
