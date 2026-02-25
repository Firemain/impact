"""Article scoring module.

Computes five sub-scores and a weighted global score:

* **ArticleScore** – raw citation impact (OpenAlex cited_by_count).
* **JournalScore** – venue prestige (OpenAlex 2yr citedness + SCImago quartile/SJR).
* **AuthorScore** – author prominence (h-index, cited_by_count).
* **FieldNormScore** – field-normalised citation percentile.
* **NetworkScore** – collaboration breadth (works, institutions diversity).

All sub-scores are normalised to [0, 1].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------
# Configurable thresholds
# ------------------------------------------------------------------

@dataclass
class ScoringThresholds:
    """Configurable thresholds for scoring normalisation.

    All values define the cap at which the corresponding sub-score reaches 1.0.
    """
    citations_max: int = 500       # ArticleScore (+ FieldNorm fallback)
    h_index_max: int = 40          # AuthorScore
    institutions_max: int = 8      # NetworkScore – institutions
    author_works_max: int = 200    # NetworkScore – avg publications


DEFAULT_THRESHOLDS = ScoringThresholds()


# ------------------------------------------------------------------
# Corpus stats placeholder (used for percentile normalisation)
# ------------------------------------------------------------------

class CorpusStats:
    """Lightweight container for corpus-level citation stats.

    Pass an existing ``CorpusStats`` to scoring functions for percentile
    normalisation.  When *None* is supplied the functions fall back to
    sensible heuristics.
    """

    def __init__(self, citation_values: Optional[List[int]] = None):
        self.citation_values: List[int] = sorted(citation_values or [])

    def percentile(self, value: int) -> float:
        if not self.citation_values:
            return 0.5
        rank = sum(1 for v in self.citation_values if v <= value)
        return rank / len(self.citation_values)


class FieldStats:
    """Per-field citation stats for field normalisation."""

    def __init__(self, stats: Optional[Dict[str, Dict[str, Any]]] = None):
        self._stats: Dict[str, Dict[str, Any]] = stats or {}

    def get(self, concept_id: str) -> Optional[Dict[str, Any]]:
        return self._stats.get(concept_id)

    def percentile_in_field(self, concept_id: str, citations: int) -> Optional[float]:
        field = self.get(concept_id)
        if not field:
            return None
        values = field.get("citations", [])
        if not values:
            return None
        rank = sum(1 for v in values if v <= citations)
        return rank / len(values)


# ==================================================================
# 1. ArticleScore
# ==================================================================

def compute_article_score(
    work: Dict[str, Any],
    corpus_stats: Optional[CorpusStats] = None,
    thresholds: Optional[ScoringThresholds] = None,
) -> Dict[str, Any]:
    """Score based on raw citation count of the article.

    Returns a dict with component details + a ``score`` in [0, 1].
    """
    t = thresholds or DEFAULT_THRESHOLDS
    raw_citations = _int(work.get("cited_by_count", 0))
    log_citations = math.log1p(raw_citations)
    log_cap = math.log1p(t.citations_max)

    if corpus_stats and corpus_stats.citation_values:
        percentile = corpus_stats.percentile(raw_citations)
    else:
        percentile = min(log_citations / log_cap, 1.0)

    score = round(0.5 * percentile + 0.5 * min(log_citations / log_cap, 1.0), 4)

    return {
        "raw_citations": raw_citations,
        "log_citations": round(log_citations, 4),
        "percentile_citations": round(percentile, 4),
        "citations_max_threshold": t.citations_max,
        "score": score,
    }


# ==================================================================
# 2. JournalScore
# ==================================================================

def compute_journal_score(
    source: Optional[Dict[str, Any]],
    scimago_info: Optional[Dict[str, object]] = None,
) -> Dict[str, Any]:
    """Score based on journal prestige (OpenAlex + SCImago).

    Returns a dict with component details + a ``score`` in [0, 1].
    """
    oa_citedness: Optional[float] = None
    oa_log: Optional[float] = None
    if source and isinstance(source, dict):
        stats = source.get("summary_stats", {})
        if isinstance(stats, dict):
            raw = stats.get("2yr_mean_citedness")
            if raw is not None:
                oa_citedness = float(raw)
                oa_log = math.log1p(oa_citedness)

    scimago_quartile: Optional[str] = None
    scimago_sjr: Optional[float] = None
    scimago_score_raw: float = 0.0
    if scimago_info and isinstance(scimago_info, dict):
        scimago_quartile = str(scimago_info.get("quartile", ""))
        sjr = scimago_info.get("sjr")
        if sjr is not None:
            scimago_sjr = float(sjr)
        from .scimago_client import compute_journal_score_from_scimago
        scimago_score_raw = compute_journal_score_from_scimago(
            scimago_quartile or "", scimago_sjr
        )

    # Combine: max possible scimago_score_raw ≈ 5, oa_log caps around 2.5
    parts: List[float] = []
    if oa_log is not None:
        parts.append(min(oa_log / 2.5, 1.0))
    if scimago_score_raw > 0:
        parts.append(min(scimago_score_raw / 5.0, 1.0))

    if parts:
        score = round(sum(parts) / len(parts), 4)
    else:
        score = 0.0

    return {
        "openalex_mean_citedness": oa_citedness,
        "openalex_log_citedness": round(oa_log, 4) if oa_log is not None else None,
        "scimago_quartile": scimago_quartile,
        "scimago_sjr": scimago_sjr,
        "scimago_score_raw": round(scimago_score_raw, 4),
        "score": score,
    }


# ==================================================================
# 3. AuthorScore
# ==================================================================

def compute_author_score(
    authors: List[Dict[str, Any]],
    thresholds: Optional[ScoringThresholds] = None,
) -> Dict[str, Any]:
    """Score based on h-index and citations of selected authors
    (typically first + last).

    Returns a dict with component details + a ``score`` in [0, 1].
    """
    t = thresholds or DEFAULT_THRESHOLDS
    h_indices: List[int] = []
    cited_by_counts: List[int] = []
    author_details: List[Dict[str, Any]] = []

    for author in authors:
        if not isinstance(author, dict):
            continue
        stats = author.get("summary_stats", {})
        h = 0
        if isinstance(stats, dict):
            h = _int(stats.get("h_index", 0))
        cited = _int(author.get("cited_by_count", 0))
        h_indices.append(h)
        cited_by_counts.append(cited)
        author_details.append({
            "display_name": str(author.get("display_name", "unknown")),
            "h_index": h,
            "cited_by_count": cited,
            "works_count": _int(author.get("works_count", 0)),
        })

    if not h_indices:
        return {
            "authors": [],
            "aggregated_h_index": 0,
            "score": 0.0,
        }

    aggregated_h = max(h_indices) if h_indices else 0
    score = round(min(aggregated_h, t.h_index_max) / t.h_index_max, 4)

    return {
        "authors": author_details,
        "authors_h_index": h_indices,
        "authors_cited_by": cited_by_counts,
        "aggregated_h_index": aggregated_h,
        "score": score,
    }


# ==================================================================
# 4. FieldNormScore
# ==================================================================

def compute_field_norm_score(
    work: Dict[str, Any],
    field_stats: Optional[FieldStats] = None,
    thresholds: Optional[ScoringThresholds] = None,
) -> Dict[str, Any]:
    """Field-normalised citation score.

    Uses the primary concept/topic of the work to determine the field,
    then computes a percentile within that field (if stats available).
    Falls back to a heuristic based on log-citations.
    """
    from .openalex_client import extract_concepts_from_work

    concepts = extract_concepts_from_work(work)
    primary = concepts[0] if concepts else {}
    concept_id = str(primary.get("id", ""))
    concept_name = str(primary.get("display_name", "unknown"))
    citations = _int(work.get("cited_by_count", 0))

    field_percentile: Optional[float] = None
    field_zscore: Optional[float] = None
    if field_stats and concept_id:
        field_percentile = field_stats.percentile_in_field(concept_id, citations)
        field_data = field_stats.get(concept_id)
        if field_data:
            mean = field_data.get("mean", 0)
            std = field_data.get("std", 1)
            if isinstance(mean, (int, float)) and isinstance(std, (int, float)) and std > 0:
                field_zscore = round((citations - mean) / std, 4)

    t = thresholds or DEFAULT_THRESHOLDS
    if field_percentile is not None:
        score = round(field_percentile, 4)
    else:
        # Fallback: same log heuristic as ArticleScore
        score = round(min(math.log1p(citations) / math.log1p(t.citations_max), 1.0), 4)

    return {
        "field_concept_id": concept_id,
        "field_concept_name": concept_name,
        "field_percentile": round(field_percentile, 4) if field_percentile is not None else None,
        "field_zscore": field_zscore,
        "score": score,
    }


# ==================================================================
# 5. NetworkScore
# ==================================================================

def compute_network_score(
    work: Dict[str, Any],
    authors: List[Dict[str, Any]],
    thresholds: Optional[ScoringThresholds] = None,
) -> Dict[str, Any]:
    """Simple network / collaboration score.

    Based on:
    - average author works count (productivity proxy)
    - number of distinct institutions across authorships
    """
    works_counts: List[int] = []
    for author in authors:
        if isinstance(author, dict):
            works_counts.append(_int(author.get("works_count", 0)))

    avg_works = sum(works_counts) / len(works_counts) if works_counts else 0.0

    # Count distinct institutions from the work's authorships
    institutions: set[str] = set()
    authorships = work.get("authorships", [])
    if isinstance(authorships, list):
        for auth_entry in authorships:
            if not isinstance(auth_entry, dict):
                continue
            insts = auth_entry.get("institutions", [])
            if isinstance(insts, list):
                for inst in insts:
                    if isinstance(inst, dict) and inst.get("display_name"):
                        institutions.add(str(inst["display_name"]))

    inst_count = len(institutions)

    t = thresholds or DEFAULT_THRESHOLDS
    works_term = min(math.log1p(avg_works) / math.log1p(t.author_works_max), 1.0)
    inst_term = min(inst_count / t.institutions_max, 1.0)
    score = round(0.5 * works_term + 0.5 * inst_term, 4)

    return {
        "avg_author_works_count": round(avg_works, 2),
        "institutions_count": inst_count,
        "institutions": sorted(institutions)[:20],
        "score": score,
    }


# ==================================================================
# Combination
# ==================================================================

DEFAULT_WEIGHTS = {
    "article": 0.35,
    "journal": 0.20,
    "author": 0.15,
    "field_norm": 0.20,
    "network": 0.10,
}


def combine_scores(
    article_score: float,
    journal_score: float,
    author_score: float,
    field_norm_score: float,
    network_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute the weighted global article-evaluation score.

    Returns a dict with the global value + weights used.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    # Normalise weights
    total_w = sum(w.values())
    if total_w <= 0:
        total_w = 1.0

    value = (
        w["article"] * article_score
        + w["journal"] * journal_score
        + w["author"] * author_score
        + w["field_norm"] * field_norm_score
        + w["network"] * network_score
    ) / total_w

    return {
        "value": round(value, 4),
        "weights": w,
        "components": {
            "article": round(article_score, 4),
            "journal": round(journal_score, 4),
            "author": round(author_score, 4),
            "field_norm": round(field_norm_score, 4),
            "network": round(network_score, 4),
        },
    }


# ------------------------------------------------------------------
# Full pipeline helper
# ------------------------------------------------------------------

def run_article_scoring(
    work: Dict[str, Any],
    source: Optional[Dict[str, Any]],
    author_payloads: List[Dict[str, Any]],
    scimago_info: Optional[Dict[str, object]] = None,
    corpus_stats: Optional[CorpusStats] = None,
    field_stats: Optional[FieldStats] = None,
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[ScoringThresholds] = None,
) -> Dict[str, Any]:
    """Run the full scoring pipeline and return a comprehensive JSON-friendly dict."""
    art = compute_article_score(work, corpus_stats, thresholds=thresholds)
    jour = compute_journal_score(source, scimago_info)
    auth = compute_author_score(author_payloads, thresholds=thresholds)
    field = compute_field_norm_score(work, field_stats, thresholds=thresholds)
    net = compute_network_score(work, author_payloads, thresholds=thresholds)

    global_score = combine_scores(
        article_score=art["score"],
        journal_score=jour["score"],
        author_score=auth["score"],
        field_norm_score=field["score"],
        network_score=net["score"],
        weights=weights,
    )

    return {
        "scores": {
            "article": art,
            "journal": jour,
            "author": auth,
            "field_norm": field,
            "network": net,
            "global": global_score,
        },
    }


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _int(val: Any) -> int:
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    try:
        return int(float(str(val)))
    except Exception:
        return 0
