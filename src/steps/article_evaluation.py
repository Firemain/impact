"""Article evaluation step – orchestrates metadata extraction via LLM,
DOI resolution, OpenAlex enrichment, SCImago lookup and multi-dimension
scoring into a single pipeline step.

Output: ``08_article_evaluation.json``
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from ..models import IngestArtifacts
except ImportError:  # pragma: no cover
    from src.models import IngestArtifacts  # type: ignore

from .doi_resolution import resolve_doi
from .metadata_extraction import extract_first_pages_text, extract_metadata_via_llm
from .openalex_client import (
    extract_author_ids_from_work,
    extract_source_id_from_work,
    get_author_by_id,
    get_source_by_id,
    get_work_by_doi,
    search_work_by_title,
)
from .scimago_client import find_journal_in_scimago
from .scoring import (
    CorpusStats,
    FieldStats,
    ScoringThresholds,
    run_article_scoring,
)


def run(
    ingest_artifacts: IngestArtifacts,
    output_dir: str | Path,
    openai_model: str = "gpt-4.1-mini",
    openai_api_base: str = "https://api.openai.com/v1",
    openai_timeout_seconds: int = 45,
    scimago_csv_path: Optional[str] = None,
    corpus_stats: Optional[CorpusStats] = None,
    field_stats: Optional[FieldStats] = None,
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[ScoringThresholds] = None,
) -> Dict[str, Any]:
    """Full article-evaluation step.

    1. Extract first-pages text from PDF.
    2. Call ChatGPT → structured metadata JSON.
    3. Resolve/validate DOI via Crossref.
    4. Fetch work from OpenAlex (DOI or title search).
    5. Fetch source (journal) + authors from OpenAlex.
    6. SCImago lookup (if ISSN or journal name available).
    7. Compute all sub-scores + global score.
    8. Write ``08_article_evaluation.json``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    notes: List[str] = []

    # ---- 1. Extract text from first pages ----
    pdf_path = ingest_artifacts.metadata.source_path
    first_pages = extract_first_pages_text(pdf_path, max_pages=3)
    if not first_pages:
        # Fallback: use existing text_index passages from pages 1-3
        first_pages = "\n\n".join(
            p.text for p in ingest_artifacts.text_index if p.page <= 3
        )
    notes.append(f"first_pages_chars={len(first_pages)}")

    # ---- 2. ChatGPT metadata extraction ----
    llm_metadata = extract_metadata_via_llm(
        first_pages,
        model=openai_model,
        api_base=openai_api_base,
        timeout_seconds=openai_timeout_seconds,
    )
    if llm_metadata:
        notes.append("llm_metadata_extraction=ok")
    else:
        notes.append("llm_metadata_extraction=failed, using ingest metadata")
        llm_metadata = _fallback_metadata(ingest_artifacts)

    # ---- 3. Resolve DOI ----
    doi = llm_metadata.get("doi") or ingest_artifacts.metadata.doi
    if not doi:
        # Try Crossref resolution
        title = llm_metadata.get("title") or ingest_artifacts.metadata.title
        first_author_ln = _first_author_last_name(llm_metadata)
        year = llm_metadata.get("year") or ingest_artifacts.metadata.year
        doi = resolve_doi(title, first_author_ln, year)
        if doi:
            notes.append(f"doi_resolved_via_crossref={doi}")
        else:
            notes.append("doi_resolution_failed")
    else:
        notes.append(f"doi_from_extraction={doi}")

    # ---- 4. OpenAlex work ----
    work: Optional[Dict[str, Any]] = None
    if doi:
        work = get_work_by_doi(doi)
        if work:
            notes.append("openalex_work=found_by_doi")
    if not work:
        title = llm_metadata.get("title") or ingest_artifacts.metadata.title
        first_ln = _first_author_last_name(llm_metadata)
        year = llm_metadata.get("year") or ingest_artifacts.metadata.year
        work = search_work_by_title(title, first_ln, year)
        if work:
            notes.append("openalex_work=found_by_title_search")
        else:
            notes.append("openalex_work=not_found")

    # ---- 5. Source + authors ----
    source: Optional[Dict[str, Any]] = None
    author_payloads: List[Dict[str, Any]] = []
    if work:
        # Source
        source_id = extract_source_id_from_work(work)
        if source_id:
            source = get_source_by_id(source_id)
            if source:
                notes.append(f"openalex_source={source.get('display_name', 'unknown')}")

        # Authors (first + last)
        author_ids = extract_author_ids_from_work(work, positions="first_last")
        for aid in author_ids:
            author_data = get_author_by_id(aid)
            if author_data:
                author_payloads.append(author_data)
        notes.append(f"openalex_authors_fetched={len(author_payloads)}")

    # ---- 6. SCImago ----
    scimago_info: Optional[Dict[str, object]] = None
    journal_meta = llm_metadata.get("journal", {})
    journal_name = ""
    issn = None
    if isinstance(journal_meta, dict):
        journal_name = str(journal_meta.get("name") or "")
        issn = journal_meta.get("issn_print") or journal_meta.get("issn_online")
    # Also try from OpenAlex source
    if not journal_name and source:
        journal_name = str(source.get("display_name", ""))
    if not issn and source:
        issn_list = source.get("issn", [])
        if isinstance(issn_list, list) and issn_list:
            issn = str(issn_list[0])

    article_year = llm_metadata.get("year") or ingest_artifacts.metadata.year
    try:
        scimago_info = find_journal_in_scimago(
            journal_name=journal_name,
            issn=str(issn) if issn else None,
            year=article_year if isinstance(article_year, int) else None,
            data_dir=scimago_csv_path,
        )
        if scimago_info:
            notes.append(
                f"scimago=found ({scimago_info.get('quartile', '?')}, "
                f"year={scimago_info.get('scimago_year', '?')})"
            )
        else:
            notes.append("scimago=not_found")
    except FileNotFoundError:
        notes.append("scimago=csv_not_available")
        scimago_info = None

    # ---- 7. Scoring ----
    document_type = getattr(ingest_artifacts.metadata, "document_type", "unknown") or "unknown"
    organization = getattr(ingest_artifacts.metadata, "organization", None)
    # Also pick from LLM metadata if available
    if llm_metadata:
        if not organization:
            organization = llm_metadata.get("organization")
        llm_doc_type = llm_metadata.get("document_type")
        if isinstance(llm_doc_type, str) and llm_doc_type != "unknown":
            document_type = llm_doc_type
    is_report = document_type in ("report", "working_paper")

    if work:
        scoring_result = run_article_scoring(
            work=work,
            source=source,
            author_payloads=author_payloads,
            scimago_info=scimago_info,
            corpus_stats=corpus_stats,
            field_stats=field_stats,
            weights=weights,
            thresholds=thresholds,
        )
    else:
        # No OpenAlex data – adapt note based on document type
        if is_report:
            note_reason = f"document is a {document_type}"
            if organization:
                note_reason += f" by {organization}"
            note_reason += " — journal-based metrics not applicable"
        else:
            note_reason = "no OpenAlex data"
        scoring_result = {
            "scores": {
                "article": {"score": 0.0, "raw_citations": 0, "note": note_reason},
                "journal": {"score": 0.0, "note": note_reason},
                "author": {"score": 0.0, "note": note_reason},
                "field_norm": {"score": 0.0, "note": note_reason},
                "network": {"score": 0.0, "note": note_reason},
                "global": {"value": 0.0, "note": note_reason},
            },
        }
        notes.append(f"scoring=minimal ({note_reason})")

    # ---- 8. Build final payload & write ----
    result = {
        "doi": doi,
        "title": llm_metadata.get("title") or ingest_artifacts.metadata.title,
        "authors_extracted": llm_metadata.get("authors", []),
        "journal_extracted": llm_metadata.get("journal", {}),
        "year": llm_metadata.get("year") or ingest_artifacts.metadata.year,
        "keywords": llm_metadata.get("keywords", []),
        "document_type": document_type,
        "organization": organization,
        "openalex_work_id": work.get("id") if work else None,
        "openalex_source_id": extract_source_id_from_work(work) if work else None,
        "openalex_cited_by_count": work.get("cited_by_count") if work else None,
        **scoring_result,
        "notes": notes,
    }

    out_file = output_path / "08_article_evaluation.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _fallback_metadata(artifacts: IngestArtifacts) -> Dict[str, Any]:
    """Build a metadata dict from the ingest artifacts when LLM fails."""
    return {
        "title": artifacts.metadata.title,
        "authors": [{"full_name": a, "last_name": a.split()[-1] if a else ""} for a in artifacts.metadata.authors],
        "journal": {"name": None, "issn_print": None, "issn_online": None, "publisher": None},
        "year": artifacts.metadata.year,
        "doi": artifacts.metadata.doi,
        "keywords": [],
    }


def _first_author_last_name(metadata: Dict[str, Any]) -> Optional[str]:
    """Extract the last name of the first author from LLM metadata."""
    authors = metadata.get("authors", [])
    if not isinstance(authors, list) or not authors:
        return None
    first = authors[0]
    if isinstance(first, dict):
        ln = first.get("last_name")
        if ln:
            return str(ln).strip()
        fn = first.get("full_name", "")
        if fn:
            parts = str(fn).strip().split()
            return parts[-1] if parts else None
    if isinstance(first, str):
        parts = first.strip().split()
        return parts[-1] if parts else None
    return None
