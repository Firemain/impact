from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    from ..models import BlockFlagItem, BlockRoutingResult, IngestArtifacts, TextPassage
    from ..text_normalize import normalize_inline_text
except ImportError:  # pragma: no cover
    from src.models import BlockFlagItem, BlockRoutingResult, IngestArtifacts, TextPassage  # type: ignore
    from src.text_normalize import normalize_inline_text  # type: ignore


EFFECT_RE = re.compile(
    r"\b(cohen'?s?\s*d|hedges?\s*g|smd|effect size|standardi[sz]ed mean difference|(?<![a-z])(d|g)\s*[=:])\b",
    re.IGNORECASE,
)
RESULTS_RE = re.compile(r"\b(results?|findings?)\b", re.IGNORECASE)
TABLE_FIG_RE = re.compile(r"\b(table\s*\d+|figure\s*\d+|fig\.\s*\d+)\b", re.IGNORECASE)
STATS_RE = re.compile(
    r"(p\s*[<=>]\s*\.?\d+|t\s*\(|f\s*\(|χ2|chi[- ]?square|ci\b|se\b|bic\b|log[- ]?likelihood)",
    re.IGNORECASE,
)
METHODS_RE = re.compile(r"\b(methods?|procedure|participants?|materials)\b", re.IGNORECASE)
POP_RE = re.compile(r"\b(sample|participants?|patients?|parents?|children|n\s*=)\b", re.IGNORECASE)
BIAS_RE = re.compile(r"\b(randomi[sz]ed|control group|blinded|attrition|drop[- ]out|allocation)\b", re.IGNORECASE)
REFERENCE_RE = re.compile(
    r"(^\s*references?\s*$|\bbibliography\b|\bet al\.\b.*\b(19|20)\d{2}\b)",
    re.IGNORECASE,
)


def run(
    ingest_artifacts: IngestArtifacts,
    output_dir: str | Path,
    use_openai_extraction: bool = False,
    openai_model: str = "gpt-4.1-mini",
    openai_api_base: str = "https://api.openai.com/v1",
    openai_timeout_seconds: int = 45,
    openai_extraction_max_snippets: int = 60,
    openai_snippet_chars: int = 1200,
) -> BlockRoutingResult:
    passages = list(ingest_artifacts.text_index)
    if not passages:
        result = BlockRoutingResult(notes=["No passages to classify."])
        _write_json(Path(output_dir) / "03_block_flags.json", result.model_dump(mode="json"))
        return result

    heuristic_items: Dict[str, BlockFlagItem] = {}
    candidate_passages: List[TextPassage] = []
    for passage in passages:
        text = normalize_inline_text(passage.text, normalize_decimal_comma=True)
        reference_like = _is_reference_like_block(passage=passage, text=text)
        results_candidate = _is_results_candidate(text) and not reference_like
        heuristic_item = _heuristic_flag_item(
            passage=passage,
            results_candidate=results_candidate,
            reference_like=reference_like,
        )
        heuristic_items[passage.evidence_id] = heuristic_item
        if results_candidate:
            candidate_passages.append(passage)

    llm_items: Dict[str, BlockFlagItem] = {}
    llm_note = "LLM block routing disabled."
    if use_openai_extraction and candidate_passages:
        llm_items, llm_note = _route_blocks_with_openai(
            passages=candidate_passages,
            model=openai_model,
            api_base=openai_api_base,
            timeout_seconds=openai_timeout_seconds,
            max_blocks=openai_extraction_max_snippets,
            snippet_chars=openai_snippet_chars,
        )

    items: List[BlockFlagItem] = []
    for passage in passages:
        heuristic_item = heuristic_items[passage.evidence_id]
        llm_item = llm_items.get(passage.evidence_id)
        if llm_item is not None:
            merged = _merge_with_heuristics(
                item=llm_item,
                source_text=passage.text,
                fallback=heuristic_item,
            )
        else:
            merged = heuristic_item
        items.append(merged)

    relevant_effect_blocks = [item.evidence_id for item in items if item.contains_effect_size]
    if not relevant_effect_blocks:
        relevant_effect_blocks = [item.evidence_id for item in items if item.contains_results][:80]
    relevant_quality_blocks = [
        item.evidence_id
        for item in items
        if item.contains_methods or item.contains_bias_info
    ]
    relevant_quality_blocks = relevant_quality_blocks[:120]

    result = BlockRoutingResult(
        items=items,
        relevant_effect_blocks=relevant_effect_blocks,
        relevant_quality_blocks=relevant_quality_blocks,
        notes=[
            llm_note,
            f"blocks_total={len(items)}",
            f"candidate_blocks={len(candidate_passages)}",
            f"effect_blocks={len(relevant_effect_blocks)}",
            f"quality_blocks={len(relevant_quality_blocks)}",
        ],
    )
    _write_json(Path(output_dir) / "03_block_flags.json", result.model_dump(mode="json"))
    return result


def _heuristic_flag_item(
    passage: TextPassage,
    results_candidate: bool,
    reference_like: bool,
) -> BlockFlagItem:
    text = normalize_inline_text(passage.text, normalize_decimal_comma=True)
    section = passage.section_guess.lower().strip()
    contains_effect_size = bool(EFFECT_RE.search(text))
    contains_results = bool(results_candidate or RESULTS_RE.search(text))
    if reference_like:
        contains_results = False
    contains_methods = bool(METHODS_RE.search(text) or section in {"methods", "limitations"})
    contains_population = bool(POP_RE.search(text) or section in {"methods", "participants"})
    contains_bias_info = bool(BIAS_RE.search(text) or section == "limitations")
    confidence = 0.45
    if contains_effect_size:
        confidence += 0.2
    if contains_results:
        confidence += 0.1
    if contains_methods:
        confidence += 0.1
    return BlockFlagItem(
        evidence_id=passage.evidence_id,
        contains_results=contains_results,
        contains_effect_size=contains_effect_size,
        contains_methods=contains_methods,
        contains_population=contains_population,
        contains_bias_info=contains_bias_info,
        confidence=min(0.95, confidence),
        notes=["heuristic_rules"],
    )


def _merge_with_heuristics(
    item: BlockFlagItem,
    source_text: str,
    fallback: BlockFlagItem,
) -> BlockFlagItem:
    text = normalize_inline_text(source_text, normalize_decimal_comma=True)
    reference_like = _is_reference_like_text(text)
    contains_results = item.contains_results or fallback.contains_results or bool(RESULTS_RE.search(text))
    if reference_like:
        contains_results = False
    return item.model_copy(
        update={
            "contains_results": contains_results,
            "contains_effect_size": item.contains_effect_size or fallback.contains_effect_size or bool(EFFECT_RE.search(text)),
            "contains_methods": item.contains_methods or fallback.contains_methods or bool(METHODS_RE.search(text)),
            "contains_population": item.contains_population or fallback.contains_population or bool(POP_RE.search(text)),
            "contains_bias_info": item.contains_bias_info or fallback.contains_bias_info or bool(BIAS_RE.search(text)),
            "notes": list(item.notes) + ["llm_plus_rules_merge"],
        }
    )


def _is_results_candidate(text: str) -> bool:
    return bool(TABLE_FIG_RE.search(text) or STATS_RE.search(text) or EFFECT_RE.search(text))


def _is_reference_like_block(passage: TextPassage, text: str) -> bool:
    if passage.section_guess.lower() == "references":
        return True
    return _is_reference_like_text(text)


def _is_reference_like_text(text: str) -> bool:
    if REFERENCE_RE.search(text):
        return True
    # Heuristic bibliography pattern: many "et al." + years in a short block.
    etal_count = len(re.findall(r"\bet al\.\b", text, flags=re.IGNORECASE))
    year_count = len(re.findall(r"\b(19|20)\d{2}\b", text))
    return etal_count >= 2 and year_count >= 2


def _route_blocks_with_openai(
    passages: Sequence[TextPassage],
    model: str,
    api_base: str,
    timeout_seconds: int,
    max_blocks: int,
    snippet_chars: int,
) -> tuple[Dict[str, BlockFlagItem], str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {}, "LLM block routing requested but OPENAI_API_KEY is missing."
    try:
        import requests  # type: ignore
    except Exception as exc:
        return {}, f"LLM block routing skipped: requests unavailable ({_compact_error(exc)})."

    batch_size = max(1, int(max_blocks))
    routed: Dict[str, BlockFlagItem] = {}
    total_batches = (len(passages) + batch_size - 1) // batch_size
    failed_batches = 0

    for start in range(0, len(passages), batch_size):
        batch = passages[start : start + batch_size]
        payload_blocks = [
            {
                "evidence_id": item.evidence_id,
                "text": normalize_inline_text(item.text, normalize_decimal_comma=True)[:snippet_chars],
            }
            for item in batch
        ]
        parsed = _openai_json_call(
            requests_module=requests,
            api_key=api_key,
            model=model,
            api_base=api_base,
            timeout_seconds=timeout_seconds,
            system_prompt=(
                "You classify scientific text blocks for downstream extraction. "
                "Return strict JSON only."
            ),
            user_prompt=(
                "For each block return booleans only:\n"
                "- contains_results\n"
                "- contains_effect_size\n"
                "- contains_methods\n"
                "- contains_population\n"
                "- contains_bias_info\n"
                "Schema:\n"
                "{\n"
                '  "items": [\n'
                "    {\n"
                '      "evidence_id": "string",\n'
                '      "contains_results": true,\n'
                '      "contains_effect_size": false,\n'
                '      "contains_methods": false,\n'
                '      "contains_population": false,\n'
                '      "contains_bias_info": false,\n'
                '      "confidence": 0.0\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                f"Blocks JSON:\n{json.dumps(payload_blocks, ensure_ascii=False)}"
            ),
        )
        if not isinstance(parsed, dict):
            failed_batches += 1
            continue
        raw_items = parsed.get("items", [])
        if not isinstance(raw_items, list):
            failed_batches += 1
            continue

        known_ids = {item["evidence_id"] for item in payload_blocks}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            evidence_id = normalize_inline_text(str(item.get("evidence_id", "")))
            if evidence_id not in known_ids:
                continue
            routed[evidence_id] = BlockFlagItem(
                evidence_id=evidence_id,
                contains_results=bool(item.get("contains_results", False)),
                contains_effect_size=bool(item.get("contains_effect_size", False)),
                contains_methods=bool(item.get("contains_methods", False)),
                contains_population=bool(item.get("contains_population", False)),
                contains_bias_info=bool(item.get("contains_bias_info", False)),
                confidence=max(0.0, min(1.0, _to_float(item.get("confidence"), default=0.5))),
                notes=["llm_routed"],
            )

    note = (
        f"LLM block routing classified {len(routed)}/{len(passages)} blocks "
        f"across {total_batches} batch(es), failed_batches={failed_batches}."
    )
    return routed, note


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
