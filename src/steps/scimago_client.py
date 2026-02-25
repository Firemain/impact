"""SCImago Journal Rank lookup via yearly CSV files.

Expects files named ``scimagojr YYYY.csv`` in ``data/``.
When looking up a journal the module picks the CSV that matches the
article's publication year.  If that year is unavailable it falls back
to the closest available year.

Downloads: https://www.scimagojr.com/journalrank.php
"""
from __future__ import annotations

import math
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Cache: year → normalised DataFrame
_CACHE: Dict[int, pd.DataFrame] = {}


# ------------------------------------------------------------------
# Available years discovery
# ------------------------------------------------------------------

def available_years(data_dir: Path | None = None) -> List[int]:
    """Return sorted list of years for which a ``scimagojr YYYY.csv`` exists."""
    d = Path(data_dir) if data_dir else _DATA_DIR
    years: List[int] = []
    for f in d.glob("scimagojr *.csv"):
        m = re.search(r"scimagojr\s+(\d{4})\.csv$", f.name, re.IGNORECASE)
        if m:
            years.append(int(m.group(1)))
    return sorted(years)


def _pick_year(target: Optional[int], data_dir: Path | None = None) -> Optional[int]:
    """Pick the best available year file.

    Priority: exact match → closest earlier → closest later → latest.
    """
    years = available_years(data_dir)
    if not years:
        return None
    if target is not None and target in years:
        return target
    if target is not None:
        # Closest year (prefer earlier)
        earlier = [y for y in years if y <= target]
        if earlier:
            return earlier[-1]
        return years[0]  # earliest available
    return years[-1]  # latest


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_scimago_year(year: int, data_dir: Path | None = None) -> pd.DataFrame:
    """Load and normalise one yearly SCImago CSV.  Result is cached."""
    if year in _CACHE:
        return _CACHE[year]

    d = Path(data_dir) if data_dir else _DATA_DIR
    csv_path = d / f"scimagojr {year}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"SCImago CSV introuvable : {csv_path}")

    df = pd.read_csv(csv_path, sep=";", dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    # ── ISSN ──
    for col in ("Issn", "ISSN"):
        if col in df.columns:
            df["issn_raw"] = df[col].fillna("").astype(str)
            break
    else:
        df["issn_raw"] = ""
    df["issn_norm"] = df["issn_raw"].apply(_normalize_issn_field)

    # ── Title ──
    for col in ("Title", "title"):
        if col in df.columns:
            df["title_norm"] = df[col].fillna("").str.strip().str.lower()
            df["title_display"] = df[col].fillna("").str.strip()
            break
    else:
        df["title_norm"] = ""
        df["title_display"] = ""

    # ── SJR (European decimal comma → float) ──
    for col in ("SJR", "sjr"):
        if col in df.columns:
            df["sjr_float"] = pd.to_numeric(
                df[col].str.replace(",", ".", regex=False), errors="coerce"
            )
            break
    else:
        df["sjr_float"] = float("nan")

    # ── Quartile ──
    for col in ("SJR Best Quartile", "Quartile", "quartile"):
        if col in df.columns:
            df["quartile"] = df[col].fillna("").str.strip().str.upper()
            break
    else:
        df["quartile"] = ""

    # ── H-index ──
    for col in ("H index", "H Index", "h_index"):
        if col in df.columns:
            df["h_index"] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            break
    else:
        df["h_index"] = 0

    # ── Categories & Areas ──
    for col in ("Categories", "categories"):
        if col in df.columns:
            df["categories_raw"] = df[col].fillna("")
            break
    else:
        df["categories_raw"] = ""
    for col in ("Areas", "areas"):
        if col in df.columns:
            df["areas_raw"] = df[col].fillna("")
            break
    else:
        df["areas_raw"] = ""

    # ── Country ──
    for col in ("Country", "country"):
        if col in df.columns:
            df["country"] = df[col].fillna("").str.strip()
            break
    else:
        df["country"] = ""

    # ── Publisher ──
    for col in ("Publisher",):
        if col in df.columns:
            df["publisher"] = df[col].fillna("").str.strip()
            break
    else:
        df["publisher"] = ""

    # ── Open Access ──
    for col in ("Open Access",):
        if col in df.columns:
            df["open_access"] = df[col].fillna("").str.strip().str.lower()
            break
    else:
        df["open_access"] = ""

    df["_year"] = year
    _CACHE[year] = df
    return df


# ------------------------------------------------------------------
# Lookup
# ------------------------------------------------------------------

def find_journal_in_scimago(
    journal_name: str,
    issn: Optional[str] = None,
    year: Optional[int] = None,
    data_dir: str | Path | None = None,
    path: str | Path | None = None,        # kept for backward compat
) -> Optional[Dict[str, object]]:
    """Find a journal in the SCImago yearly data.

    Picks the CSV for the closest available *year*.
    Priority: ISSN match → fuzzy title match.

    Returns a rich dict or *None*::

        {
            "quartile": "Q1",
            "sjr": 1.234,
            "title": "Journal of …",
            "h_index": 85,
            "categories": "Oncology (Q1); ...",
            "areas": "Medicine",
            "country": "United States",
            "publisher": "Elsevier",
            "open_access": "yes",
            "scimago_year": 2021,
        }
    """
    d = Path(data_dir or path or _DATA_DIR)
    chosen_year = _pick_year(year, d)
    if chosen_year is None:
        raise FileNotFoundError(
            f"Aucun fichier scimagojr YYYY.csv trouvé dans {d}"
        )
    try:
        df = load_scimago_year(chosen_year, d)
    except FileNotFoundError:
        return None

    # 1) ISSN match
    if issn:
        norm_issn = _normalize_issn(issn)
        if norm_issn:
            matches = df[df["issn_norm"].str.contains(norm_issn, na=False)]
            if not matches.empty:
                return _row_to_dict(matches.iloc[0])

    # 2) Fuzzy title match
    if journal_name:
        norm_name = journal_name.strip().lower()
        candidates = df["title_norm"].dropna().tolist()
        close = get_close_matches(norm_name, candidates, n=1, cutoff=0.75)
        if close:
            idx = df.index[df["title_norm"] == close[0]]
            if len(idx) > 0:
                return _row_to_dict(df.loc[idx[0]])

    return None


def find_journal_trajectory(
    journal_name: str,
    issn: Optional[str] = None,
    data_dir: str | Path | None = None,
) -> List[Dict[str, object]]:
    """Look up a journal across ALL available years.

    Returns a list of dicts (one per year found), sorted chronologically.
    Useful for showing SJR/quartile evolution over time.
    """
    d = Path(data_dir) if data_dir else _DATA_DIR
    results: List[Dict[str, object]] = []
    for y in available_years(d):
        hit = find_journal_in_scimago(journal_name, issn=issn, year=y, data_dir=d)
        if hit:
            results.append(hit)
    return results


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def compute_journal_score_from_scimago(
    quartile: str,
    sjr: Optional[float] = None,
) -> float:
    """Compute a 0–5 scale score from SCImago quartile + SJR.

    Base: Q1=4, Q2=3, Q3=2, Q4=1, unknown=0.
    Bonus: log1p(SJR) capped at ~1.
    """
    base_map = {"Q1": 4, "Q2": 3, "Q3": 2, "Q4": 1}
    base = base_map.get(quartile.strip().upper(), 0)
    sjr_term = 0.0
    if sjr is not None and not math.isnan(sjr):
        sjr_term = min(math.log1p(sjr), 1.0)
    return base + sjr_term


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _row_to_dict(row) -> Dict[str, object]:
    sjr_val = row.get("sjr_float")
    if isinstance(sjr_val, float) and math.isnan(sjr_val):
        sjr_val = None
    return {
        "quartile": str(row.get("quartile", "")),
        "sjr": sjr_val,
        "title": str(row.get("title_display", "")),
        "h_index": int(row.get("h_index", 0)),
        "categories": str(row.get("categories_raw", "")),
        "areas": str(row.get("areas_raw", "")),
        "country": str(row.get("country", "")),
        "publisher": str(row.get("publisher", "")),
        "open_access": str(row.get("open_access", "")),
        "scimago_year": int(row.get("_year", 0)),
    }


def _normalize_issn_field(raw: str) -> str:
    """Normalize a possibly multi-valued ISSN field (e.g. '1234-5678, 8765-4321')."""
    parts = re.findall(r"\d{4}-?\d{3}[\dXx]", raw)
    return "|".join(_normalize_issn(p) for p in parts if _normalize_issn(p))


def _normalize_issn(issn: str) -> str:
    digits = re.sub(r"[^0-9Xx]", "", issn.strip())
    if len(digits) == 8:
        return f"{digits[:4]}-{digits[4:]}"
    return digits
