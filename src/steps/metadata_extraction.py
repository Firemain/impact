"""ChatGPT-based metadata extraction from the first pages of a PDF.

Sends the raw text of pages 1-2 to a ChatGPT model and asks for a
strict JSON with title, authors, journal, DOI, etc.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def extract_metadata_via_llm(
    first_pages_text: str,
    model: str = "gpt-4.1-mini",
    api_base: str = "https://api.openai.com/v1",
    timeout_seconds: int = 45,
) -> Optional[Dict[str, Any]]:
    """Send first pages text to ChatGPT and get structured metadata JSON.

    Returns a dict matching the schema below, or *None* on failure.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    if not first_pages_text or len(first_pages_text.strip()) < 50:
        return None

    system_prompt = _SYSTEM_PROMPT
    user_prompt = _USER_PROMPT_TEMPLATE.format(text=first_pages_text[:8000])

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
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_seconds)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return _post_process(parsed)
    except Exception:
        pass
    return None


def extract_first_pages_text(pdf_path: str | Path, max_pages: int = 3) -> str:
    """Extract raw text from the first *max_pages* of a PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return ""
    text_parts: List[str] = []
    try:
        doc = fitz.open(str(pdf_path))
        for i in range(min(max_pages, len(doc))):
            page = doc[i]
            text_parts.append(page.get_text("text"))
        doc.close()
    except Exception:
        return ""
    return "\n\n".join(text_parts)


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------

_SYSTEM_PROMPT = """\
Tu es un assistant specialise dans l'extraction de metadonnees de documents scientifiques ou de rapports.
Ta seule tache est de lire le texte fourni (titres, auteurs, entetes de revue, resume, etc.)
et de renvoyer un JSON STRICT, sans texte additionnel, qui contient les metadonnees du document.

Regles importantes :
- Ne JAMAIS ajouter de commentaires en dehors du JSON.
- Si une information n'est pas presente, mets null.
- Ne jamais inventer un DOI ou un ISSN. Si tu n'es pas sur, mets null.
- Le champ "authors" doit respecter exactement la structure demandee.
- Le champ "journal" doit respecter exactement la structure demandee.
- Le champ "document_type" doit etre un des types suivants exactement :
  "journal_article", "report", "working_paper", "thesis", "book_chapter", "preprint", "unknown".
  Indices pour classifier :
    - "report" : Prepare for / Prepared by, organisation commanditaire, pas de journal
    - "working_paper" : Working Paper, Discussion Paper, NBER, IZA, etc.
    - "thesis" : These, Dissertation, Memoire
    - "preprint" : arXiv, SSRN, preprint
    - "journal_article" : publie dans une revue avec volume/issue/pages
- Le champ "organization" est l'organisme qui a publie ou commande le document (ex: RTI International, RAND, World Bank).
- Le champ "funder" est l'organisme qui a finance l'etude (ex: Social Innovation Fund, NIH, NSF).

FORMAT OBLIGATOIRE :
{
  "title": "Full exact title of the document",
  "authors": [
    {
      "full_name": "First Last",
      "last_name": "Last",
      "first_name": "First",
      "affiliation": "Main affiliation if present",
      "is_corresponding": true
    }
  ],
  "journal": {
    "name": "Journal name or null if not a journal article",
    "issn_print": "1234-5678",
    "issn_online": "1234-567X",
    "publisher": "Publisher name"
  },
  "year": 2021,
  "volume": "34",
  "issue": "2",
  "pages": "123-134",
  "doi": "10.1234/example.doi.2021.001",
  "doi_detected": true,
  "keywords": ["keyword 1", "keyword 2"],
  "document_type": "journal_article",
  "organization": "Organization name or null",
  "funder": "Funding organization or null"
}
"""

_USER_PROMPT_TEMPLATE = """\
Voici les premieres pages d'un document scientifique (article, rapport, working paper, etc.).
Extrait :
<<<
{text}
>>>

Renvoie UNIQUEMENT le JSON suivant, rien d'autre.
"""


# ------------------------------------------------------------------
# Post-processing
# ------------------------------------------------------------------

def _post_process(data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitise LLM output."""
    # Ensure authors is a list
    if not isinstance(data.get("authors"), list):
        data["authors"] = []
    # Ensure journal is a dict
    if not isinstance(data.get("journal"), dict):
        data["journal"] = {"name": None, "issn_print": None, "issn_online": None, "publisher": None}
    # Strip whitespace from string fields
    for key in ("title", "doi", "volume", "issue", "pages"):
        if isinstance(data.get(key), str):
            data[key] = data[key].strip() or None
    # Validate DOI format
    doi = data.get("doi")
    if isinstance(doi, str) and not re.match(r"^10\.\d{4,9}/", doi):
        data["doi"] = None
        data["doi_detected"] = False
    return data
