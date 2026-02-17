from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from ..models import BlockRoutingResult, IngestArtifacts, QuickQualityResult
    from ..text_normalize import normalize_inline_text
except ImportError:  # pragma: no cover
    from src.models import BlockRoutingResult, IngestArtifacts, QuickQualityResult  # type: ignore
    from src.text_normalize import normalize_inline_text  # type: ignore


RANDOM_RE = re.compile(r"\brandomi[sz]ed|randomly assigned\b", re.IGNORECASE)
CONTROL_RE = re.compile(r"\bcontrol group|comparator|usual care|wait[- ]?list\b", re.IGNORECASE)
N_RE = re.compile(r"\bn\s*=\s*\d+\b", re.IGNORECASE)
ATTRITION_RE = re.compile(r"\battrition|drop[- ]?out|withdraw(n|al)|lost to follow[- ]?up\b", re.IGNORECASE)
BLIND_RE = re.compile(r"\bblind(ed|ing)?\b", re.IGNORECASE)


def run(
    ingest_artifacts: IngestArtifacts,
    block_routing: BlockRoutingResult,
    output_dir: str | Path,
    use_openai_extraction: bool = False,
    openai_model: str = "gpt-4.1-mini",
    openai_api_base: str = "https://api.openai.com/v1",
    openai_timeout_seconds: int = 45,
    openai_extraction_max_snippets: int = 40,
    openai_snippet_chars: int = 1200,
) -> QuickQualityResult:
    text_by_id = {item.evidence_id: item.text for item in ingest_artifacts.text_index}
    selected_ids = [evidence_id for evidence_id in block_routing.relevant_quality_blocks if evidence_id in text_by_id]

    llm_note = "LLM quick quality disabled."
    result: Optional[QuickQualityResult] = None
    if use_openai_extraction and selected_ids:
        result, llm_note = _quality_with_openai(
            selected_ids=selected_ids,
            text_by_id=text_by_id,
            model=openai_model,
            api_base=openai_api_base,
            timeout_seconds=openai_timeout_seconds,
            batch_size=openai_extraction_max_snippets,
            snippet_chars=openai_snippet_chars,
        )

    if result is None:
        result = _quality_with_heuristics(selected_ids=selected_ids, text_by_id=text_by_id)
        result.notes.append("heuristic_fallback")
    result.notes.append(llm_note)

    _write_json(Path(output_dir) / "05_quality_quick.json", result.model_dump(mode="json"))
    return result


def _quality_with_openai(
    selected_ids: Sequence[str],
    text_by_id: Dict[str, str],
    model: str,
    api_base: str,
    timeout_seconds: int,
    batch_size: int,
    snippet_chars: int,
) -> tuple[Optional[QuickQualityResult], str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None, "LLM quick quality requested but OPENAI_API_KEY is missing."
    try:
        import requests  # type: ignore
    except Exception as exc:
        return None, f"LLM quick quality skipped: requests unavailable ({_compact_error(exc)})."

    per_call = max(1, int(batch_size))
    total_batches = (len(selected_ids) + per_call - 1) // per_call
    failed_batches = 0
    success_batches = 0

    randomization_values: List[str] = []
    control_values: List[str] = []
    sample_values: List[str] = []
    attrition_values: List[str] = []
    blinding_values: List[str] = []
    justifications: List[str] = []
    evidence_ids: List[str] = []

    for start in range(0, len(selected_ids), per_call):
        batch_ids = selected_ids[start : start + per_call]
        payload = [
            {
                "evidence_id": evidence_id,
                "text": normalize_inline_text(text_by_id[evidence_id], normalize_decimal_comma=True)[:snippet_chars],
            }
            for evidence_id in batch_ids
        ]
        parsed = _openai_json_call(
            requests_module=requests,
            api_key=api_key,
            model=model,
            api_base=api_base,
            timeout_seconds=timeout_seconds,
            system_prompt="You produce a quick methodological quality check from snippets. Return strict JSON only.",
            user_prompt=(
                "Score only from explicit evidence.\n"
                "Schema:\n"
                "{\n"
                '  "randomization": "yes|no|unclear",\n'
                '  "control_group": "yes|no|unclear",\n'
                '  "sample_size_reported": "yes|no|unclear",\n'
                '  "attrition_reported": "yes|no|unclear",\n'
                '  "blinding_reported": "yes|no|unclear",\n'
                '  "internal_quality_score": 0.0,\n'
                '  "justification": "short text",\n'
                '  "evidence_ids": ["string"]\n'
                "}\n\n"
                f"Snippets JSON:\n{json.dumps(payload, ensure_ascii=False)}"
            ),
        )
        if not isinstance(parsed, dict):
            failed_batches += 1
            continue
        success_batches += 1
        randomization_values.append(_yes_no_unclear(parsed.get("randomization")))
        control_values.append(_yes_no_unclear(parsed.get("control_group")))
        sample_values.append(_yes_no_unclear(parsed.get("sample_size_reported")))
        attrition_values.append(_yes_no_unclear(parsed.get("attrition_reported")))
        blinding_values.append(_yes_no_unclear(parsed.get("blinding_reported")))

        justification = normalize_inline_text(str(parsed.get("justification", "")))
        if justification:
            justifications.append(justification)

        known_ids = {item["evidence_id"] for item in payload}
        raw_ids = parsed.get("evidence_ids", [])
        if isinstance(raw_ids, list):
            for value in raw_ids:
                evidence_id = normalize_inline_text(str(value))
                if evidence_id in known_ids and evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)

    if success_batches == 0:
        return None, "LLM quick quality failed."

    randomization = _aggregate_flag(randomization_values)
    control_group = _aggregate_flag(control_values)
    sample_size_reported = _aggregate_flag(sample_values)
    attrition_reported = _aggregate_flag(attrition_values)
    blinding_reported = _aggregate_flag(blinding_values)

    score = 0.0
    for value in [
        randomization,
        control_group,
        sample_size_reported,
        attrition_reported,
        blinding_reported,
    ]:
        if value == "yes":
            score += 0.2
        elif value == "unclear":
            score += 0.1
    justification = " | ".join(justifications[:3]) if justifications else "LLM aggregated quick quality."
    result = QuickQualityResult(
        randomization=randomization,
        control_group=control_group,
        sample_size_reported=sample_size_reported,
        attrition_reported=attrition_reported,
        blinding_reported=blinding_reported,
        internal_quality_score=round(max(0.0, min(1.0, score)), 3),
        justification=justification,
        evidence_ids=evidence_ids[:30],
        notes=["llm_quick_quality_batches"],
    )
    note = (
        f"LLM quick quality applied: success_batches={success_batches}, "
        f"failed_batches={failed_batches}, total_batches={total_batches}."
    )
    return result, note


def _quality_with_heuristics(
    selected_ids: Sequence[str],
    text_by_id: Dict[str, str],
) -> QuickQualityResult:
    merged = " ".join(normalize_inline_text(text_by_id[evidence_id], normalize_decimal_comma=True) for evidence_id in selected_ids)
    randomization = "yes" if RANDOM_RE.search(merged) else "unclear"
    control_group = "yes" if CONTROL_RE.search(merged) else "unclear"
    sample_size_reported = "yes" if N_RE.search(merged) else "unclear"
    attrition_reported = "yes" if ATTRITION_RE.search(merged) else "unclear"
    blinding_reported = "yes" if BLIND_RE.search(merged) else "unclear"

    score = 0.0
    mapping = {
        "randomization": randomization,
        "control_group": control_group,
        "sample_size_reported": sample_size_reported,
        "attrition_reported": attrition_reported,
        "blinding_reported": blinding_reported,
    }
    for value in mapping.values():
        if value == "yes":
            score += 0.2
        elif value == "unclear":
            score += 0.1
    return QuickQualityResult(
        randomization=randomization,
        control_group=control_group,
        sample_size_reported=sample_size_reported,
        attrition_reported=attrition_reported,
        blinding_reported=blinding_reported,
        internal_quality_score=round(min(1.0, score), 3),
        justification="Heuristic quick quality from methods/bias blocks.",
        evidence_ids=list(selected_ids[:20]),
        notes=[],
    )


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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        response = requests_module.post(endpoint, headers=headers, json=payload, timeout=timeout_seconds)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        return None
    return None


def _yes_no_unclear(value: Any) -> str:
    text = normalize_inline_text(str(value)).lower()
    if text in {"yes", "no", "unclear"}:
        return text
    return "unclear"


def _aggregate_flag(values: Sequence[str]) -> str:
    normalized = [_yes_no_unclear(value) for value in values if value]
    if any(value == "yes" for value in normalized):
        return "yes"
    if any(value == "no" for value in normalized):
        return "no"
    return "unclear"


def _to_float(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = normalize_inline_text(str(value), normalize_decimal_comma=True)
    try:
        return float(text)
    except Exception:
        return default


def _compact_error(exc: Exception) -> str:
    text = normalize_inline_text(str(exc))
    return text[:220] + ("..." if len(text) > 220 else "")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
