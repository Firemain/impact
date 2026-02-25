from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote_plus

try:
    from ..models import ExternalCredibilityResult, IngestArtifacts
    from ..text_normalize import normalize_inline_text
except ImportError:  # pragma: no cover
    from src.models import ExternalCredibilityResult, IngestArtifacts  # type: ignore
    from src.text_normalize import normalize_inline_text  # type: ignore


def run(
    ingest_artifacts: IngestArtifacts,
    output_dir: str | Path,
    openai_model: str = "gpt-4.1-mini",
    openai_api_base: str = "https://api.openai.com/v1",
    openai_timeout_seconds: int = 30,
    use_openai_extraction: bool = False,
) -> ExternalCredibilityResult:
    metadata = ingest_artifacts.metadata
    title = normalize_inline_text(metadata.title)
    authors = list(metadata.authors)
    year = metadata.year
    doi = normalize_inline_text(metadata.doi or "")
    document_type = getattr(metadata, "document_type", "unknown") or "unknown"
    organization = getattr(metadata, "organization", None) or None
    if not doi:
        doi = _extract_doi_from_artifacts(ingest_artifacts)
    openalex_email = normalize_inline_text(os.getenv("OPENALEX_EMAIL", "").strip())
    notes: list[str] = []

    title_match_found = False
    venue = "unknown"
    publisher = "unknown"
    citation_count: Optional[int] = None
    authors_found = 0
    score = 0.0

    openalex_payload, openalex_note = _fetch_openalex_work(
        doi=doi,
        title=title,
        authors=authors,
        year=year,
        openalex_email=openalex_email,
        timeout_seconds=openai_timeout_seconds,
    )
    notes.append(openalex_note)
    if openalex_email:
        notes.append(f"OpenAlex polite pool via mailto={openalex_email}")
    if isinstance(openalex_payload, dict):
        title_match_found = True
        venue = _pick_nested_str(openalex_payload, ["primary_location", "source", "display_name"]) or "unknown"
        publisher = _pick_nested_str(openalex_payload, ["primary_location", "source", "host_organization_name"]) or "unknown"
        citation_count = _to_optional_int(openalex_payload.get("cited_by_count"))
        authorships = openalex_payload.get("authorships", [])
        if isinstance(authorships, list):
            authors_found = len(authorships)

    llm_score_note = "LLM credibility scoring disabled."
    is_report = document_type in ("report", "working_paper")
    if is_report:
        # Reports: use organization-based scoring instead of journal metrics
        score = _report_credibility_score(
            title_match_found=title_match_found,
            citation_count=citation_count,
            organization=organization,
        )
        level = _level_from_score(score)
        notes.append(f"document_type={document_type}, scored via organization-based heuristic.")
        if organization:
            notes.append(f"organization={organization}")
    elif use_openai_extraction:
        llm_score, llm_level, llm_score_note = _llm_credibility_score(
            title=title,
            venue=venue,
            publisher=publisher,
            citation_count=citation_count,
            authors_found=authors_found,
            model=openai_model,
            api_base=openai_api_base,
            timeout_seconds=openai_timeout_seconds,
        )
        if llm_score is not None:
            score = llm_score
            level = llm_level
        else:
            score = _deterministic_external_score(
                title_match_found=title_match_found,
                citation_count=citation_count,
                venue=venue,
            )
            level = _level_from_score(score)
    else:
        score = _deterministic_external_score(
            title_match_found=title_match_found,
            citation_count=citation_count,
            venue=venue,
        )
        level = _level_from_score(score)
    notes.append(llm_score_note)

    result = ExternalCredibilityResult(
        title_match_found=title_match_found,
        venue=venue,
        publisher=publisher,
        citation_count=citation_count,
        authors_found=authors_found,
        external_score=round(score, 3),
        credibility_level=level,
        document_type=document_type,
        organization=organization,
        notes=notes,
    )
    _write_json(Path(output_dir) / "06_external_credibility.json", result.model_dump(mode="json"))
    return result


def _fetch_openalex_work(
    doi: str,
    title: str,
    authors: list[str],
    year: Optional[int],
    openalex_email: str,
    timeout_seconds: int,
) -> tuple[Optional[Dict[str, Any]], str]:
    if (not doi or doi == "unknown") and (not title or title == "unknown"):
        return None, "OpenAlex skipped: missing doi/title."
    try:
        import requests  # type: ignore
    except Exception as exc:
        return None, f"OpenAlex skipped: requests unavailable ({_compact_error(exc)})."

    if doi:
        normalized_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
        endpoint = f"https://api.openalex.org/works/https://doi.org/{quote_plus(normalized_doi)}"
        try:
            params = {"mailto": openalex_email} if openalex_email else None
            response = requests.get(endpoint, params=params, timeout=timeout_seconds)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("id"):
                return payload, "OpenAlex DOI match found."
        except Exception:
            pass

    query_parts: list[str] = [title] if title else []
    if authors:
        first_author = normalize_inline_text(authors[0]).split(" ")[-1]
        if first_author:
            query_parts.append(first_author)
    query = " ".join(part for part in query_parts if part).strip()
    if not query:
        return None, "OpenAlex skipped: empty search query."

    endpoint = "https://api.openalex.org/works"
    params: Dict[str, str] = {
        "search": query,
        "per-page": "5",
    }
    if isinstance(year, int):
        params["filter"] = f"publication_year:{year}"
    if openalex_email:
        params["mailto"] = openalex_email
    try:
        response = requests.get(endpoint, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", []) if isinstance(payload, dict) else []
        if not isinstance(results, list) or not results:
            return None, "OpenAlex: no match."
        best_work = _select_best_openalex_result(results, title=title, year=year)
        if isinstance(best_work, dict):
            return best_work, "OpenAlex title/authors/year match found."
        return None, "OpenAlex: no reliable match."
    except Exception as exc:
        return None, f"OpenAlex failed ({_compact_error(exc)})."


def _select_best_openalex_result(
    results: list[Any],
    title: str,
    year: Optional[int],
) -> Optional[Dict[str, Any]]:
    normalized_title = _normalize_for_match(title)
    best_score = -1.0
    best_work: Optional[Dict[str, Any]] = None
    for work in results:
        if not isinstance(work, dict):
            continue
        candidate_title = normalize_inline_text(str(work.get("display_name", "")))
        if not candidate_title:
            continue
        score = _title_overlap_score(normalized_title, _normalize_for_match(candidate_title))
        pub_year = work.get("publication_year")
        if isinstance(year, int) and pub_year == year:
            score += 0.15
        if score > best_score:
            best_score = score
            best_work = work
    if best_score < 0.35:
        return None
    return best_work


def _title_overlap_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_tokens = {token for token in a.split() if len(token) > 2}
    b_tokens = {token for token in b.split() if len(token) > 2}
    if not a_tokens or not b_tokens:
        return 0.0
    overlap = len(a_tokens & b_tokens)
    return overlap / max(len(a_tokens), 1)


def _extract_doi_from_artifacts(ingest_artifacts: IngestArtifacts) -> str:
    doi_re = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b")
    for passage in ingest_artifacts.text_index[:80]:
        text = normalize_inline_text(passage.text)
        match = doi_re.search(text)
        if match:
            return match.group(0).rstrip(".,;)]")
    return ""


def _normalize_for_match(text: str) -> str:
    normalized = normalize_inline_text(text, normalize_decimal_comma=False).lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _llm_credibility_score(
    title: str,
    venue: str,
    publisher: str,
    citation_count: Optional[int],
    authors_found: int,
    model: str,
    api_base: str,
    timeout_seconds: int,
) -> tuple[Optional[float], str, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "Unknown", "LLM credibility skipped: OPENAI_API_KEY missing."
    try:
        import requests  # type: ignore
    except Exception as exc:
        return None, "Unknown", f"LLM credibility skipped: requests unavailable ({_compact_error(exc)})."

    system_prompt = (
        "You score external credibility of a scientific article from metadata only. "
        "Return strict JSON only."
    )
    user_prompt = (
        "Score credibility in [0,1] with conservative assumptions.\n"
        "Schema:\n"
        "{\n"
        '  "external_score": 0.0,\n'
        '  "credibility_level": "High|Moderate|Low|Unknown",\n'
        '  "reason": "short text"\n'
        "}\n\n"
        f"title: {title}\n"
        f"venue: {venue}\n"
        f"publisher: {publisher}\n"
        f"citation_count: {citation_count if citation_count is not None else 'unknown'}\n"
        f"authors_found: {authors_found}\n"
    )
    parsed = _openai_json_call(
        requests_module=requests,
        api_key=api_key,
        model=model,
        api_base=api_base,
        timeout_seconds=timeout_seconds,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    if not isinstance(parsed, dict):
        return None, "Unknown", "LLM credibility scoring failed."

    score = _to_optional_float(parsed.get("external_score"))
    if score is None:
        return None, "Unknown", "LLM credibility invalid score."
    score = max(0.0, min(1.0, score))
    level = str(parsed.get("credibility_level", "Unknown")).strip()
    if level not in {"High", "Moderate", "Low", "Unknown"}:
        level = _level_from_score(score)
    reason = normalize_inline_text(str(parsed.get("reason", "")))
    note = "LLM credibility scoring applied."
    if reason:
        note = f"{note} reason={reason[:180]}"
    return score, level, note


def _openai_json_call(
    requests_module,
    api_key: str,
    model: str,
    api_base: str,
    timeout_seconds: int,
    system_prompt: str,
    user_prompt: str,
) -> Optional[Dict[str, Any]]:
    endpoint = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests_module.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _deterministic_external_score(
    title_match_found: bool,
    citation_count: Optional[int],
    venue: str,
) -> float:
    score = 0.0
    if title_match_found:
        score += 0.4
    if citation_count is not None:
        if citation_count >= 200:
            score += 0.4
        elif citation_count >= 50:
            score += 0.3
        elif citation_count >= 10:
            score += 0.2
        elif citation_count >= 1:
            score += 0.1
    if venue and venue != "unknown":
        score += 0.2
    return max(0.0, min(1.0, score))


# Well-known research organizations that produce credible reports
_KNOWN_RESEARCH_ORGS = {
    "rti international", "rand corporation", "rand", "brookings",
    "urban institute", "mathematica", "abt associates", "mdrc",
    "world bank", "imf", "oecd", "undp", "unicef", "who",
    "national bureau of economic research", "nber", "iza",
    "j-pal", "innovations for poverty action", "ipa",
    "mckinsey", "deloitte", "bcg", "pwc", "kpmg", "ey",
    "institute for fiscal studies", "ifs",
    "center for global development", "cgd",
    "american institutes for research", "air",
    "westat", "rmc research", "src", "norc",
    "national academies", "national academy of sciences",
    "government accountability office", "gao",
    "congressional budget office", "cbo",
    "european commission", "ec",
}


def _report_credibility_score(
    title_match_found: bool,
    citation_count: Optional[int],
    organization: Optional[str],
) -> float:
    """Credibility scoring adapted for reports and working papers.

    Reports don't have journals, so we score based on:
    - Title match in OpenAlex (indexation = visibility) → 0.25
    - Organization reputation → 0.35
    - Citation count → 0.40
    """
    score = 0.0
    # Indexation bonus
    if title_match_found:
        score += 0.25
    # Organization reputation
    if organization:
        org_lower = organization.strip().lower()
        if any(known in org_lower for known in _KNOWN_RESEARCH_ORGS):
            score += 0.35
        else:
            score += 0.15  # identified but not well-known org
    # Citations
    if citation_count is not None:
        if citation_count >= 100:
            score += 0.40
        elif citation_count >= 30:
            score += 0.30
        elif citation_count >= 10:
            score += 0.20
        elif citation_count >= 1:
            score += 0.10
    return max(0.0, min(1.0, score))


def _level_from_score(score: float) -> str:
    if score >= 0.75:
        return "High"
    if score >= 0.45:
        return "Moderate"
    if score > 0.0:
        return "Low"
    return "Unknown"


def _pick_nested_str(payload: Dict[str, Any], path: list[str]) -> str:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    if isinstance(current, str):
        return normalize_inline_text(current)
    return ""


def _to_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = normalize_inline_text(str(value))
    try:
        return int(float(text))
    except Exception:
        return None


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = normalize_inline_text(str(value))
    try:
        return float(text)
    except Exception:
        return None


def _compact_error(exc: Exception) -> str:
    text = normalize_inline_text(str(exc))
    return text[:220] + ("..." if len(text) > 220 else "")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
