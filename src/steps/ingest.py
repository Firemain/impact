from __future__ import annotations

import hashlib
import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree

try:
    from ..models import (
        DecisionLog,
        ExtractedTable,
        IngestArtifacts,
        IngestConfig,
        PaperMetadata,
        TableRow,
        TextPassage,
    )
    from ..text_normalize import normalize_block_text, normalize_for_match as normalize_text_for_match
except ImportError:  # pragma: no cover
    from src.models import (  # type: ignore
        DecisionLog,
        ExtractedTable,
        IngestArtifacts,
        IngestConfig,
        PaperMetadata,
        TableRow,
        TextPassage,
    )
    from src.text_normalize import normalize_block_text, normalize_for_match as normalize_text_for_match  # type: ignore


SECTION_ALIASES = {
    "title": ["title"],
    "abstract": ["abstract", "summary"],
    "introduction": ["introduction", "background"],
    "methods": ["methods", "materials and methods", "methodology", "participants"],
    "intervention": ["intervention", "procedure", "protocol", "treatment"],
    "results": ["results", "findings"],
    "discussion": ["discussion"],
    "conclusion": ["conclusion", "conclusions"],
    "limitations": ["limitations", "strengths and limitations"],
    "funding": ["funding", "acknowledgments", "acknowledgements"],
    "references": ["references", "bibliography"],
    "appendix": ["appendix", "supplementary material", "supplementary materials"],
}

METADATA_LABEL_PATTERN = re.compile(
    r"^(title|institution|authors?|doi|no\.?\s*of\s*pages|last\s*updated|citation|issn|co-registration|contributions|support/funding|potential conflicts|corresponding author|editors?)\b",
    re.IGNORECASE,
)

METADATA_NOISE_PATTERNS = [
    r"\bview metadata\b",
    r"\bbrought to you by\b",
    r"\bprovided by\b",
    r"\bcore\.ac\.uk\b",
    r"\bcopyright\b",
    r"\bissn\b",
    r"\bdoi\b",
    r"\bsearch executed\b",
    r"\bfirst published\b",
    r"\blast updated\b",
]

NON_AUTHOR_TOKENS = {
    "review",
    "reviews",
    "collaboration",
    "institution",
    "department",
    "university",
    "institute",
    "centre",
    "center",
    "hospital",
    "school",
    "documentation",
    "core",
    "health",
    "services",
}


def run(
    pdf_path: str | Path,
    output_root: str | Path = "outputs",
    paper_id: Optional[str] = None,
    config: Optional[IngestConfig] = None,
) -> IngestArtifacts:
    """
    Ingest a scientific PDF into normalized artifacts.

    Outputs:
      - 00_metadata.json
      - 01_text_index.json
      - 02_tables.json
      - 00_ingest_logs.json
    """
    _load_dotenv_if_available()
    ingest_config = config or IngestConfig()
    logs: List[DecisionLog] = []

    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.exists():
        raise FileNotFoundError(f"PDF not found: {pdf}")
    if pdf.suffix.lower() != ".pdf":
        _log(logs, "ingest", "Input does not end with .pdf", "warning", path=str(pdf))

    pdf_hash = _sha256_file(pdf)
    resolved_paper_id = paper_id or _build_paper_id(pdf.stem, pdf_hash)
    output_dir = Path(output_root).resolve() / resolved_paper_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if ingest_config.use_grobid and ingest_config.grobid_url == IngestConfig().grobid_url:
        ingest_config = ingest_config.model_copy(
            update={"grobid_url": os.getenv("GROBID_URL", ingest_config.grobid_url)}
        )
    if ingest_config.use_openai_metadata:
        ingest_config = ingest_config.model_copy(
            update={
                "openai_api_base": os.getenv(
                    "OPENAI_API_BASE",
                    ingest_config.openai_api_base,
                )
            }
        )

    pages_text, parser_name = _extract_pages_text(pdf, logs)
    toc_entries = _extract_toc_entries(pages_text, logs)
    grobid_data = _fetch_grobid_data(pdf, ingest_config, logs)
    if grobid_data.headings:
        toc_entries = _merge_unique(toc_entries, grobid_data.headings)

    text_index = _build_text_index(
        pages_text=pages_text,
        paper_id=resolved_paper_id,
        toc_entries=toc_entries,
        config=ingest_config,
        logs=logs,
    )

    tables = _extract_tables(
        pdf_path=pdf,
        pages_text=pages_text,
        output_dir=output_dir,
        logs=logs,
    )
    metadata = _build_metadata(
        paper_id=resolved_paper_id,
        pdf_path=pdf,
        pdf_hash=pdf_hash,
        parser_name=parser_name,
        pages_text=pages_text,
        grobid_data=grobid_data,
        config=ingest_config,
        logs=logs,
    )

    artifacts = IngestArtifacts(
        metadata=metadata,
        text_index=text_index,
        tables=tables,
        logs=logs,
    )

    _log(
        logs,
        "ingest",
        "Ingestion complete",
        "info",
        paper_id=resolved_paper_id,
        output_dir=str(output_dir),
        num_passages=len(text_index),
        num_tables=len(tables),
    )

    _write_json(output_dir / "00_metadata.json", artifacts.metadata.model_dump(mode="json"))
    _write_json(
        output_dir / "01_text_index.json",
        [entry.model_dump(mode="json") for entry in artifacts.text_index],
    )
    _write_json(
        output_dir / "02_tables.json",
        [table.model_dump(mode="json") for table in artifacts.tables],
    )
    _write_json(
        output_dir / "00_ingest_logs.json",
        [entry.model_dump(mode="json") for entry in artifacts.logs],
    )
    return artifacts


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _log(
    logs: List[DecisionLog],
    step: str,
    message: str,
    level: str = "info",
    **details: object,
) -> None:
    logs.append(
        DecisionLog(step=step, level=level, message=message, details=details or {})
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_paper_id(stem: str, pdf_hash: str) -> str:
    slug = _slugify(stem) or "paper"
    return f"{slug[:40]}-{pdf_hash[:8]}"


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return value


def _extract_pages_text(pdf_path: Path, logs: List[DecisionLog]) -> Tuple[List[str], str]:
    try:
        import fitz  # type: ignore

        with fitz.open(pdf_path) as document:
            pages = [_normalize_text(page.get_text("text")) for page in document]
        _log(logs, "ingest.extract_text", "Text extracted with PyMuPDF", parser="pymupdf")
        return pages, "pymupdf"
    except Exception as exc:
        _log(
            logs,
            "ingest.extract_text",
            "PyMuPDF unavailable or failed, fallback to pdfplumber",
            "warning",
            error=str(exc),
        )

    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(pdf_path) as pdf:
            pages = [_normalize_text(page.extract_text() or "") for page in pdf.pages]
        _log(logs, "ingest.extract_text", "Text extracted with pdfplumber", parser="pdfplumber")
        return pages, "pdfplumber"
    except Exception as exc:
        _log(
            logs,
            "ingest.extract_text",
            "pdfplumber extraction failed",
            "error",
            error=str(exc),
        )
        raise RuntimeError(
            "Unable to extract text. Install pymupdf or pdfplumber."
        ) from exc


def _extract_toc_entries(pages_text: Sequence[str], logs: List[DecisionLog]) -> List[str]:
    toc_entries: List[str] = []
    max_pages = min(12, len(pages_text))
    toc_page_index: Optional[int] = None

    for page_idx in range(max_pages):
        lower_text = pages_text[page_idx].lower()
        if "table of contents" in lower_text or re.search(r"\bsommaire\b|\bcontents\b", lower_text):
            toc_page_index = page_idx
            break

    if toc_page_index is None:
        return []

    entry_pattern = re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\s+)?(.+?)\s*(?:\.{2,}|\s{2,})(\d{1,3})\s*$"
    )

    for page_idx in range(toc_page_index, min(toc_page_index + 3, len(pages_text))):
        for raw_line in pages_text[page_idx].splitlines():
            line = _normalize_text(raw_line)
            if not line:
                continue
            match = entry_pattern.match(line)
            if not match:
                continue
            candidate = _normalize_heading(match.group(1))
            if candidate and candidate not in toc_entries:
                toc_entries.append(candidate)

    if toc_entries:
        _log(
            logs,
            "ingest.toc",
            "TOC entries detected",
            "info",
            toc_entries=len(toc_entries),
        )
    return toc_entries


def _build_text_index(
    pages_text: Sequence[str],
    paper_id: str,
    toc_entries: Sequence[str],
    config: IngestConfig,
    logs: List[DecisionLog],
) -> List[TextPassage]:
    passages: List[TextPassage] = []
    current_section = "unknown"

    for page_number, page_text in enumerate(pages_text, start=1):
        blocks, current_section = _split_page_into_blocks(
            page_text=page_text,
            current_section=current_section,
            toc_entries=toc_entries,
        )

        for block_index, (section_name, block_text) in enumerate(blocks, start=1):
            for chunk_index, chunk in enumerate(
                _chunk_text(
                    text=block_text,
                    chunk_chars=config.chunk_chars,
                    chunk_overlap=config.chunk_overlap,
                    min_chunk_chars=config.min_chunk_chars,
                ),
                start=1,
            ):
                evidence_id = _build_evidence_id(
                    paper_id=paper_id,
                    page=page_number,
                    section=section_name,
                    block_index=block_index,
                    chunk_index=chunk_index,
                    text=chunk,
                )
                passages.append(
                    TextPassage(
                        evidence_id=evidence_id,
                        page=page_number,
                        section_guess=section_name,
                        text=chunk,
                    )
                )

    _log(
        logs,
        "ingest.segment",
        "Text segmentation complete",
        "info",
        passages=len(passages),
        pages=len(pages_text),
    )
    return passages


def _split_page_into_blocks(
    page_text: str,
    current_section: str,
    toc_entries: Sequence[str],
) -> Tuple[List[Tuple[str, str]], str]:
    lines = [line.rstrip() for line in page_text.splitlines()]
    heading_candidates: List[Tuple[int, str]] = []

    for idx, line in enumerate(lines):
        normalized = _normalize_heading(line)
        if not normalized:
            continue
        if _looks_like_heading(normalized, toc_entries):
            heading_candidates.append((idx, _canonical_section(normalized)))

    blocks: List[Tuple[int, str]] = []
    buffer: List[str] = []
    buffer_start_idx = 0

    def flush_buffer() -> None:
        nonlocal buffer, buffer_start_idx
        if not buffer:
            return
        text = _normalize_text(" ".join(buffer))
        if text:
            blocks.append((buffer_start_idx, text))
        buffer = []

    for line_idx, line in enumerate(lines):
        normalized_line = _normalize_text(line)
        if _looks_like_heading(_normalize_heading(line), toc_entries):
            flush_buffer()
            continue
        if normalized_line:
            if not buffer:
                buffer_start_idx = line_idx
            buffer.append(normalized_line)
        else:
            flush_buffer()
    flush_buffer()

    resolved_blocks: List[Tuple[str, str]] = []
    sorted_headings = sorted(heading_candidates, key=lambda item: item[0])

    for block_start, block_text in blocks:
        section = current_section
        for heading_line_idx, heading_name in sorted_headings:
            if heading_line_idx <= block_start:
                section = heading_name
            else:
                break
        resolved_blocks.append((section, block_text))

    if sorted_headings:
        current_section = sorted_headings[-1][1]
    return resolved_blocks, current_section


def _looks_like_heading(line: str, toc_entries: Sequence[str]) -> bool:
    if not line:
        return False
    if len(line) > 120 or len(line) < 3:
        return False
    if line.lower().startswith(("figure ", "table ", "fig.", "copyright")):
        return False
    if re.fullmatch(r"[0-9\W]+", line):
        return False
    lower = line.lower()

    if any(lower == alias for aliases in SECTION_ALIASES.values() for alias in aliases):
        return True
    if re.match(
        r"^(?:\d+(?:\.\d+){0,3}\s+)?(abstract|introduction|background|methods?|results?|discussion|conclusions?|references|appendix)\b",
        lower,
    ):
        return True
    if re.match(r"^\d+(?:\.\d+){0,4}\s+[a-z]", lower):
        return True

    alpha_chars = [char for char in line if char.isalpha()]
    if alpha_chars:
        upper_ratio = sum(char.isupper() for char in alpha_chars) / len(alpha_chars)
        if upper_ratio > 0.85 and len(line.split()) <= 12:
            return True

    normalized = _normalize_heading(line)
    for toc_entry in toc_entries:
        if SequenceMatcher(None, normalized.lower(), toc_entry.lower()).ratio() >= 0.88:
            return True
    return False


def _normalize_heading(line: str) -> str:
    cleaned = _normalize_text(line)
    cleaned = re.sub(r"^\d+(?:\.\d+)*\s*", "", cleaned)
    cleaned = cleaned.strip(":-. ")
    return cleaned


def _canonical_section(heading: str) -> str:
    normalized = _normalize_heading(heading).lower()
    for section_name, aliases in SECTION_ALIASES.items():
        if normalized in aliases:
            return section_name
    return normalized if normalized else "unknown"


def _chunk_text(
    text: str,
    chunk_chars: int,
    chunk_overlap: int,
    min_chunk_chars: int,
) -> List[str]:
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    safe_overlap = min(max(chunk_overlap, 0), max(chunk_chars - 1, 0))
    step = max(1, chunk_chars - safe_overlap)

    while start < len(text):
        end = min(start + chunk_chars, len(text))
        if end < len(text):
            while end > start + min_chunk_chars and text[end - 1].isalnum() and text[end].isalnum():
                end -= 1
        chunk = text[start:end].strip()
        if chunk and (len(chunk) >= min_chunk_chars or not chunks):
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step
    return chunks


def _build_evidence_id(
    paper_id: str,
    page: int,
    section: str,
    block_index: int,
    chunk_index: int,
    text: str,
) -> str:
    raw = f"{paper_id}|{page}|{section}|{block_index}|{chunk_index}|{text[:240]}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"evd_{digest}"


def _normalize_text(text: str) -> str:
    return normalize_block_text(text, normalize_decimal_comma=True)


def _extract_tables(
    pdf_path: Path,
    pages_text: Sequence[str],
    output_dir: Path,
    logs: List[DecisionLog],
) -> List[ExtractedTable]:
    tables = _extract_tables_pdfplumber(pdf_path, logs)
    if not tables:
        tables = _extract_tables_camelot(pdf_path, logs)

    marker_pages = _detect_table_marker_pages(pages_text)
    structured_pages = {table.page for table in tables if table.status == "ok" and table.structured}
    pages_for_images = sorted(page for page in marker_pages if page not in structured_pages)
    image_map = _render_table_pages_as_images(
        pdf_path=pdf_path,
        pages=pages_for_images,
        output_dir=output_dir,
        logs=logs,
    )

    if image_map:
        updated_tables: List[ExtractedTable] = []
        for table in tables:
            image_path = image_map.get(table.page)
            if image_path and (not table.structured or table.status != "ok"):
                status = table.status
                if status in {"empty", "failed"}:
                    status = "image_only"
                table = table.model_copy(
                    update={
                        "image_path": image_path,
                        "status": status,
                    }
                )
            updated_tables.append(table)
        tables = updated_tables

        existing_pages = {table.page for table in tables}
        for page, image_path in image_map.items():
            if page in existing_pages:
                continue
            tables.append(
                ExtractedTable(
                    table_id=_build_table_id(page, 0, []),
                    page=page,
                    title_guess=f"table_page_{page}",
                    rows=[],
                    structured=False,
                    image_path=image_path,
                    extraction_method="none",
                    status="image_only",
                )
            )

    if not tables:
        _log(logs, "ingest.tables", "No tables extracted", "warning")
    return tables


def _extract_tables_pdfplumber(pdf_path: Path, logs: List[DecisionLog]) -> List[ExtractedTable]:
    try:
        import pdfplumber  # type: ignore
    except Exception as exc:
        _log(
            logs,
            "ingest.tables.pdfplumber",
            "pdfplumber unavailable",
            "warning",
            error=str(exc),
        )
        return []

    extracted: List[ExtractedTable] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                raw_tables = page.extract_tables() or []
                for table_idx, raw_table in enumerate(raw_tables, start=1):
                    rows: List[TableRow] = []
                    for row_index, row in enumerate(raw_table):
                        if row is None:
                            continue
                        cells = [_normalize_text(cell or "") for cell in row]
                        if any(cells):
                            rows.append(TableRow(row_index=row_index, cells=cells))

                    table_id = _build_table_id(page_number, table_idx, rows)
                    status = "ok" if rows else "empty"
                    structured = _is_structured_table(rows)
                    extracted.append(
                        ExtractedTable(
                            table_id=table_id,
                            page=page_number,
                            title_guess="unknown",
                            rows=rows,
                            structured=structured,
                            extraction_method="pdfplumber",
                            status=status,
                        )
                    )
    except Exception as exc:
        _log(
            logs,
            "ingest.tables.pdfplumber",
            "pdfplumber table extraction failed",
            "warning",
            error=str(exc),
        )
        return []

    _log(
        logs,
        "ingest.tables.pdfplumber",
        "Tables extracted with pdfplumber",
        "info",
        tables=len(extracted),
    )
    return extracted


def _extract_tables_camelot(pdf_path: Path, logs: List[DecisionLog]) -> List[ExtractedTable]:
    try:
        import camelot  # type: ignore
    except Exception as exc:
        _log(
            logs,
            "ingest.tables.camelot",
            "camelot unavailable",
            "warning",
            error=str(exc),
        )
        return []

    extracted: List[ExtractedTable] = []
    try:
        candidate_tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")
        for idx, table in enumerate(candidate_tables, start=1):
            page = int(table.parsing_report.get("page", 0)) if table.parsing_report else 0
            rows: List[TableRow] = []
            dataframe = table.df.fillna("")
            for row_index, row_values in enumerate(dataframe.values.tolist()):
                cells = [_normalize_text(cell) for cell in row_values]
                if any(cells):
                    rows.append(TableRow(row_index=row_index, cells=cells))

            table_id = _build_table_id(page, idx, rows)
            structured = _is_structured_table(rows)
            extracted.append(
                ExtractedTable(
                    table_id=table_id,
                    page=page,
                    title_guess="unknown",
                    rows=rows,
                    structured=structured,
                    extraction_method="camelot",
                    status="ok" if rows else "empty",
                )
            )
    except Exception as exc:
        _log(
            logs,
            "ingest.tables.camelot",
            "camelot table extraction failed",
            "warning",
            error=str(exc),
        )
        return []

    _log(logs, "ingest.tables.camelot", "Tables extracted with camelot", "info", tables=len(extracted))
    return extracted


def _build_table_id(page: int, index: int, rows: Sequence[TableRow]) -> str:
    preview = ""
    if rows:
        preview = "|".join(rows[0].cells[:5])
    digest = hashlib.sha1(f"{page}|{index}|{preview}".encode("utf-8")).hexdigest()[:16]
    return f"tbl_{digest}"


def _is_structured_table(rows: Sequence[TableRow]) -> bool:
    if len(rows) < 2:
        return False
    non_empty_counts = [sum(1 for cell in row.cells if cell.strip()) for row in rows]
    if not non_empty_counts:
        return False
    return max(non_empty_counts) >= 3 and sum(count >= 2 for count in non_empty_counts) >= 2


def _detect_table_marker_pages(pages_text: Sequence[str]) -> List[int]:
    marker_re = re.compile(r"\btable\s*(?:\d+|[ivxlcdm]+)\b", re.IGNORECASE)
    marker_pages: List[int] = []
    for page_number, page_text in enumerate(pages_text, start=1):
        if marker_re.search(page_text):
            marker_pages.append(page_number)
    return marker_pages


def _render_table_pages_as_images(
    pdf_path: Path,
    pages: Sequence[int],
    output_dir: Path,
    logs: List[DecisionLog],
) -> Dict[int, str]:
    if not pages:
        return {}

    try:
        import fitz  # type: ignore
    except Exception as exc:
        _log(
            logs,
            "ingest.tables.image_render",
            "PyMuPDF unavailable for table page rendering",
            "warning",
            error=str(exc),
        )
        return {}

    image_dir = output_dir / "table_pages"
    image_dir.mkdir(parents=True, exist_ok=True)
    rendered: Dict[int, str] = {}
    try:
        with fitz.open(pdf_path) as document:
            for page_number in pages:
                if page_number < 1 or page_number > len(document):
                    continue
                page = document[page_number - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                image_path = image_dir / f"table_page_{page_number}.png"
                pix.save(image_path)
                rendered[page_number] = str(image_path)
    except Exception as exc:
        _log(
            logs,
            "ingest.tables.image_render",
            "Failed to render table candidate pages",
            "warning",
            error=str(exc),
        )
        return {}

    _log(
        logs,
        "ingest.tables.image_render",
        "Rendered table candidate pages to images",
        "info",
        pages=len(rendered),
    )
    return rendered


class _GrobidData:
    def __init__(self) -> None:
        self.title: Optional[str] = None
        self.authors: List[str] = []
        self.year: Optional[int] = None
        self.headings: List[str] = []


def _fetch_grobid_data(
    pdf_path: Path,
    config: IngestConfig,
    logs: List[DecisionLog],
) -> _GrobidData:
    data = _GrobidData()
    if not config.use_grobid:
        return data

    tei_xml = _fetch_grobid_tei(pdf_path, config, logs)
    if not tei_xml:
        return data
    return _parse_grobid_tei(tei_xml, logs)


def _fetch_grobid_tei(
    pdf_path: Path,
    config: IngestConfig,
    logs: List[DecisionLog],
) -> Optional[str]:
    try:
        import requests
    except Exception as exc:
        _log(
            logs,
            "ingest.grobid",
            "requests unavailable, cannot call GROBID",
            "warning",
            error=str(exc),
        )
        return None

    endpoint = f"{config.grobid_url.rstrip('/')}/api/processFulltextDocument"
    try:
        with pdf_path.open("rb") as file_handle:
            response = requests.post(
                endpoint,
                files={"input": (pdf_path.name, file_handle, "application/pdf")},
                data={
                    "consolidateHeader": "0",
                    "includeRawCitations": "0",
                    "segmentSentences": "0",
                },
                timeout=config.request_timeout_seconds,
            )
        response.raise_for_status()
    except Exception as exc:
        _log(
            logs,
            "ingest.grobid",
            "GROBID request failed, using local parser only",
            "warning",
            endpoint=endpoint,
            error=str(exc),
        )
        return None

    _log(logs, "ingest.grobid", "GROBID response received", "info", endpoint=endpoint)
    return response.text


def _parse_grobid_tei(tei_xml: str, logs: List[DecisionLog]) -> _GrobidData:
    data = _GrobidData()
    try:
        root = ElementTree.fromstring(tei_xml)
    except Exception as exc:
        _log(logs, "ingest.grobid", "Invalid TEI XML from GROBID", "warning", error=str(exc))
        return data

    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    title_node = root.find(".//tei:titleStmt/tei:title", ns)
    if title_node is not None and title_node.text:
        data.title = _normalize_text(title_node.text)

    for author in root.findall(".//tei:sourceDesc//tei:author", ns):
        forename = author.find(".//tei:forename", ns)
        surname = author.find(".//tei:surname", ns)
        full_name = _normalize_text(
            " ".join(part for part in [forename.text if forename is not None else "", surname.text if surname is not None else ""])
        )
        if full_name and full_name not in data.authors:
            data.authors.append(full_name)

    date_node = root.find(".//tei:publicationStmt//tei:date", ns)
    if date_node is not None:
        text_value = date_node.attrib.get("when", "") or (date_node.text or "")
        year = _extract_year(text_value)
        if year:
            data.year = year

    for head_node in root.findall(".//tei:body//tei:head", ns):
        if head_node.text:
            heading = _normalize_heading(head_node.text)
            if heading:
                canonical = _canonical_section(heading)
                if canonical not in data.headings:
                    data.headings.append(canonical)

    _log(
        logs,
        "ingest.grobid",
        "GROBID metadata parsed",
        "info",
        title_found=bool(data.title),
        authors=len(data.authors),
        headings=len(data.headings),
    )
    return data


def _build_metadata(
    paper_id: str,
    pdf_path: Path,
    pdf_hash: str,
    parser_name: str,
    pages_text: Sequence[str],
    grobid_data: _GrobidData,
    config: IngestConfig,
    logs: List[DecisionLog],
) -> PaperMetadata:
    metadata_lines = _collect_metadata_lines(pages_text, config.metadata_pages_scan)
    title = _guess_title(metadata_lines, pages_text) or "unknown"
    authors = _guess_authors(metadata_lines, pages_text)
    year = _guess_year(metadata_lines, pages_text)
    doi = _guess_doi(metadata_lines, pages_text)

    metadata_source = "heuristics"
    if grobid_data.title:
        title = grobid_data.title
        metadata_source = "grobid"
    if grobid_data.authors:
        authors = grobid_data.authors
        metadata_source = "grobid"
    if grobid_data.year:
        year = grobid_data.year
        metadata_source = "grobid"

    llm_metadata = _extract_metadata_with_openai(
        pages_text=pages_text,
        config=config,
        logs=logs,
    )
    document_type: str = "unknown"
    organization: Optional[str] = None
    journal_name: Optional[str] = None
    funder: Optional[str] = None

    if llm_metadata is not None:
        llm_title = llm_metadata.get("title")
        llm_authors = llm_metadata.get("authors")
        llm_year = llm_metadata.get("year")
        if isinstance(llm_title, str) and llm_title.strip():
            title = llm_title.strip()
            metadata_source = "openai_metadata"
        if isinstance(llm_authors, list) and llm_authors:
            authors = [str(author).strip() for author in llm_authors if str(author).strip()]
            metadata_source = "openai_metadata"
        if isinstance(llm_year, int):
            year = llm_year
            metadata_source = "openai_metadata"
        # New fields: document_type, organization, journal_name, funder
        llm_doc_type = llm_metadata.get("document_type")
        if isinstance(llm_doc_type, str) and llm_doc_type in (
            "journal_article", "report", "working_paper", "thesis",
            "book_chapter", "preprint",
        ):
            document_type = llm_doc_type
        llm_org = llm_metadata.get("organization")
        if isinstance(llm_org, str) and llm_org.strip():
            organization = llm_org.strip()
        llm_funder = llm_metadata.get("funder")
        if isinstance(llm_funder, str) and llm_funder.strip():
            funder = llm_funder.strip()
        llm_journal = llm_metadata.get("journal")
        if isinstance(llm_journal, dict):
            jn = llm_journal.get("name")
            if isinstance(jn, str) and jn.strip():
                journal_name = jn.strip()

    title = _clean_title(title)
    authors = _dedupe_preserve_order([_clean_author_name(author) for author in authors if author])
    if not authors:
        authors = ["unknown"]

    _log(
        logs,
        "ingest.metadata",
        "Metadata extraction completed",
        "info",
        source=metadata_source,
        title=title,
        authors_count=len(authors),
        year=year,
        doi=doi,
        document_type=document_type,
        organization=organization,
    )

    return PaperMetadata(
        paper_id=paper_id,
        source_path=str(pdf_path),
        pdf_hash=pdf_hash,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        num_pages=len(pages_text),
        parser=parser_name,
        document_type=document_type,
        organization=organization,
        journal_name=journal_name,
        funder=funder,
    )


def _collect_metadata_lines(pages_text: Sequence[str], pages_scan: int) -> List[str]:
    if not pages_text:
        return []
    lines: List[str] = []
    limit = max(1, min(pages_scan, len(pages_text)))
    for page_text in pages_text[:limit]:
        for raw_line in page_text.splitlines():
            line = _normalize_text(raw_line)
            if not line:
                continue
            lines.append(line)
    return lines


def _guess_title(metadata_lines: Sequence[str], pages_text: Sequence[str]) -> Optional[str]:
    labeled_title = _guess_title_from_labeled_block(metadata_lines)
    if labeled_title:
        return labeled_title

    if pages_text:
        front_title = _guess_title_from_front_page(pages_text[0])
        if front_title:
            return front_title

    for line in metadata_lines:
        if _is_title_line(line):
            return line
    return None


def _guess_title_from_labeled_block(metadata_lines: Sequence[str]) -> Optional[str]:
    for index, line in enumerate(metadata_lines):
        match = re.match(r"^title\s+(.+)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        collected = [match.group(1).strip()]
        for next_line in metadata_lines[index + 1 : index + 8]:
            if METADATA_LABEL_PATTERN.match(next_line):
                break
            if _looks_like_author_list_line(next_line):
                break
            if _is_noise_line(next_line):
                break
            if len(next_line.split()) <= 1:
                break
            collected.append(next_line)
        return _clean_title(" ".join(collected))
    return None


def _guess_title_from_front_page(first_page_text: str) -> Optional[str]:
    first_page_lines = [_normalize_text(line) for line in first_page_text.splitlines() if _normalize_text(line)]
    if not first_page_lines:
        return None

    best: Optional[str] = None
    best_score = -1.0

    for start_index, line in enumerate(first_page_lines[:70]):
        if not _is_title_line(line):
            continue

        candidate_lines = [line]
        for next_line in first_page_lines[start_index + 1 : start_index + 5]:
            if not _is_title_line(next_line):
                break
            if _looks_like_author_list_line(next_line):
                break
            candidate_lines.append(next_line)

        candidate = _clean_title(" ".join(candidate_lines))
        score = _score_title_candidate(candidate)
        if score > best_score:
            best = candidate
            best_score = score
    return best


def _score_title_candidate(candidate: str) -> float:
    lower = candidate.lower()
    words = candidate.split()
    if not words:
        return -1.0

    score = 0.0
    score += min(len(words), 18) * 0.4
    score += sum(1 for word in words if word[0].isupper()) * 0.12
    if 5 <= len(words) <= 18:
        score += 2.0
    if "-" in candidate:
        score += 0.5
    if re.search(r"\b(trial|review|intervention|health|effect|programme|program)\b", lower):
        score += 0.6
    if re.search(r"\b(first published|last updated|search executed|campbell systematic reviews)\b", lower):
        score -= 5.0
    if _looks_like_author_list_line(candidate):
        score -= 6.0
    if _is_noise_line(candidate):
        score -= 8.0
    if re.search(r"\b\d{4}\b", candidate):
        score -= 1.2
    return score


def _is_title_line(line: str) -> bool:
    if not line:
        return False
    if _is_noise_line(line):
        return False
    if _looks_like_author_list_line(line):
        return False
    lower = line.lower()
    if METADATA_LABEL_PATTERN.match(line):
        return False
    if re.search(r"\b(first published|last updated|search executed|doi|issn)\b", lower):
        return False
    words = line.split()
    if len(words) < 3 or len(words) > 20:
        return False
    digit_count = sum(char.isdigit() for char in line)
    if digit_count > max(2, len(line) * 0.08):
        return False
    alpha_words = [word for word in words if any(char.isalpha() for char in word)]
    if not alpha_words:
        return False
    lower_ratio = sum(word.islower() for word in alpha_words) / len(alpha_words)
    titlecase_ratio = sum(word[:1].isupper() for word in alpha_words) / len(alpha_words)
    return lower_ratio > 0.12 and titlecase_ratio > 0.4


def _guess_authors(metadata_lines: Sequence[str], pages_text: Sequence[str]) -> List[str]:
    labeled = _guess_authors_from_labeled_block(metadata_lines)
    if labeled:
        return labeled

    for line in metadata_lines:
        if _looks_like_author_list_line(line):
            parsed = _parse_authors_from_line(line)
            if parsed:
                return parsed

    if pages_text:
        first_page_lines = [
            _normalize_text(line)
            for line in pages_text[0].splitlines()
            if _normalize_text(line)
        ]
        for line in first_page_lines:
            if _looks_like_author_list_line(line):
                parsed = _parse_authors_from_line(line)
                if parsed:
                    return parsed
    return []


def _guess_authors_from_labeled_block(metadata_lines: Sequence[str]) -> List[str]:
    collected: List[str] = []
    for index, line in enumerate(metadata_lines):
        match = re.match(r"^authors?\s+(.+)$", line, flags=re.IGNORECASE)
        if not match:
            continue
        collected.extend(_parse_authors_from_line(match.group(1)))
        for next_line in metadata_lines[index + 1 : index + 20]:
            if METADATA_LABEL_PATTERN.match(next_line):
                break
            if _is_noise_line(next_line):
                break
            parsed = _parse_authors_from_line(next_line)
            if parsed:
                collected.extend(parsed)
            elif collected:
                break
        break

    return _dedupe_preserve_order([author for author in collected if _is_likely_person_name(author)])


def _looks_like_author_list_line(line: str) -> bool:
    if not line:
        return False
    if re.search(r"\b(authors?|editor|corresponding author)\b", line.lower()):
        return True
    parsed = _parse_authors_from_line(line)
    return len(parsed) >= 2


def _parse_authors_from_line(line: str) -> List[str]:
    line = _normalize_text(line)
    if not line:
        return []
    line = re.sub(r"^authors?\s*", "", line, flags=re.IGNORECASE).strip()
    line = re.sub(r"\s+and\s+", ", ", line, flags=re.IGNORECASE)

    candidates: List[str] = []

    # "Surname, Name" format.
    surname_match = re.fullmatch(
        r"([A-Z][A-Za-z'`-]+)\s*,\s*([A-Z][A-Za-z'`.-]+(?:\s+[A-Z][A-Za-z'`.-]+)*)",
        line,
    )
    if surname_match:
        candidates.append(f"{surname_match.group(2)} {surname_match.group(1)}")
        return [name for name in candidates if _is_likely_person_name(name)]

    # Comma-separated full names.
    if "," in line and not re.search(r"\bdoi\b|\bissn\b", line.lower()):
        segments = [segment.strip() for segment in line.split(",") if segment.strip()]
        for segment in segments:
            normalized = _clean_author_name(segment)
            if _is_likely_person_name(normalized):
                candidates.append(normalized)
        if candidates:
            return _dedupe_preserve_order(candidates)

    # Single full name.
    normalized_single = _clean_author_name(line)
    if _is_likely_person_name(normalized_single):
        candidates.append(normalized_single)
    return _dedupe_preserve_order(candidates)


def _is_likely_person_name(candidate: str) -> bool:
    if not candidate:
        return False
    candidate = _clean_author_name(candidate)
    words = [word for word in candidate.split() if word]
    if len(words) < 2 or len(words) > 5:
        return False
    if any(char.isdigit() for char in candidate):
        return False
    if re.search(r"[@/]", candidate):
        return False
    lower_words = [re.sub(r"[^a-z]", "", word.lower()) for word in words]
    if any(word in NON_AUTHOR_TOKENS for word in lower_words if word):
        return False
    capitalized_tokens = sum(word[:1].isupper() for word in words)
    if capitalized_tokens < max(2, len(words) - 1):
        return False
    return True


def _guess_year(metadata_lines: Sequence[str], pages_text: Sequence[str]) -> Optional[int]:
    preferred_text = " ".join(metadata_lines[:120])
    labeled_years: List[int] = []
    for line in metadata_lines:
        lower = line.lower()
        if any(keyword in lower for keyword in ["last updated", "published", "citation", "year"]):
            year = _extract_year(line)
            if year is not None:
                labeled_years.append(year)
    if labeled_years:
        return max(labeled_years)

    year = _extract_year(preferred_text)
    if year is not None:
        return year
    return _extract_year(" ".join(pages_text[:3]))


def _guess_doi(metadata_lines: Sequence[str], pages_text: Sequence[str]) -> Optional[str]:
    preferred_text = " ".join(metadata_lines[:160])
    doi = _extract_doi(preferred_text)
    if doi:
        return doi
    return _extract_doi(" ".join(pages_text[:3]))


def _extract_metadata_with_openai(
    pages_text: Sequence[str],
    config: IngestConfig,
    logs: List[DecisionLog],
) -> Optional[Dict[str, Any]]:
    if not config.use_openai_metadata:
        return None

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        _log(
            logs,
            "ingest.metadata.llm",
            "OPENAI_API_KEY missing; skipping LLM metadata extraction",
            "warning",
        )
        return None

    text_for_llm = "\n\n".join(pages_text[: max(1, config.metadata_pages_scan)])
    text_for_llm = text_for_llm[:16000]
    if not text_for_llm.strip():
        return None

    try:
        import requests
    except Exception as exc:
        _log(
            logs,
            "ingest.metadata.llm",
            "requests unavailable; skipping LLM metadata extraction",
            "warning",
            error=str(exc),
        )
        return None

    endpoint = f"{config.openai_api_base.rstrip('/')}/chat/completions"
    system_prompt = (
        "You extract bibliographic metadata from raw PDF text. "
        "Return strict JSON only with keys: title (string), authors (array of strings), year (integer or null). "
        "Do not invent values. If unsure return null or an empty array."
    )
    user_prompt = (
        "Extract metadata from the text below.\n"
        "Rules:\n"
        "- Prefer true article/review title, not journal name.\n"
        "- Authors must be person names only.\n"
        "- Year must be a publication/update year present in text.\n\n"
        f"TEXT:\n{text_for_llm}"
    )
    payload = {
        "model": config.openai_model,
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
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=config.openai_timeout_seconds,
        )
        response.raise_for_status()
        response_payload = response.json()
        content = response_payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except Exception as exc:
        _log(
            logs,
            "ingest.metadata.llm",
            "LLM metadata extraction failed",
            "warning",
            endpoint=endpoint,
            error=str(exc),
        )
        return None

    validated = _validate_openai_metadata(parsed, text_for_llm, logs)
    if validated is None:
        return None

    _log(
        logs,
        "ingest.metadata.llm",
        "LLM metadata extraction accepted",
        "info",
        model=config.openai_model,
        has_title=bool(validated.get("title")),
        authors_count=len(validated.get("authors", [])),
        year=validated.get("year"),
    )
    return validated


def _validate_openai_metadata(
    candidate: Dict[str, Any],
    source_text: str,
    logs: List[DecisionLog],
) -> Optional[Dict[str, Any]]:
    if not isinstance(candidate, dict):
        return None

    cleaned: Dict[str, Any] = {"title": None, "authors": [], "year": None}
    normalized_source = _normalize_for_match(source_text)

    title = candidate.get("title")
    if isinstance(title, str):
        title = _clean_title(title)
        if title and _supports_text_evidence(title, normalized_source):
            cleaned["title"] = title

    raw_authors = candidate.get("authors")
    if isinstance(raw_authors, list):
        validated_authors: List[str] = []
        for raw_author in raw_authors:
            author = _clean_author_name(str(raw_author))
            if not _is_likely_person_name(author):
                continue
            if _supports_text_evidence(author, normalized_source, threshold=0.66):
                validated_authors.append(author)
        cleaned["authors"] = _dedupe_preserve_order(validated_authors)

    year = candidate.get("year")
    if isinstance(year, int) and 1900 <= year <= 2100 and str(year) in source_text:
        cleaned["year"] = year
    elif isinstance(year, str) and year.isdigit():
        year_value = int(year)
        if 1900 <= year_value <= 2100 and year in source_text:
            cleaned["year"] = year_value

    if cleaned["title"] is None and not cleaned["authors"] and cleaned["year"] is None:
        _log(
            logs,
            "ingest.metadata.llm",
            "LLM metadata rejected by evidence validation",
            "warning",
        )
        return None
    return cleaned


def _supports_text_evidence(text: str, normalized_source: str, threshold: float = 0.75) -> bool:
    target = _normalize_for_match(text)
    if not target:
        return False
    if target in normalized_source:
        return True

    tokens = [token for token in target.split() if len(token) > 2]
    if not tokens:
        return False
    present = sum(1 for token in tokens if token in normalized_source)
    return (present / len(tokens)) >= threshold


def _normalize_for_match(text: str) -> str:
    return normalize_text_for_match(text)


def _clean_title(title: str) -> str:
    cleaned = _normalize_text(title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,:")
    return cleaned


def _clean_author_name(author: str) -> str:
    cleaned = _normalize_text(author)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ;,:.")
    return cleaned


def _is_noise_line(line: str) -> bool:
    lower = line.lower()
    return any(re.search(pattern, lower) for pattern in METADATA_NOISE_PATTERNS)


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _extract_year(text: str) -> Optional[int]:
    years = re.findall(r"\b(19\d{2}|20\d{2}|2100)\b", text)
    if not years:
        return None
    integers = sorted({int(year) for year in years if 1900 <= int(year) <= 2100})
    if not integers:
        return None
    return integers[-1]


def _extract_doi(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text)
    if not match:
        return None
    doi = match.group(0).rstrip(".,;)]")
    return doi if doi else None


def _merge_unique(left: Iterable[str], right: Iterable[str]) -> List[str]:
    merged: List[str] = []
    for entry in list(left) + list(right):
        if entry and entry not in merged:
            merged.append(entry)
    return merged
