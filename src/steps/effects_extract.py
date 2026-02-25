from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from ..effect_labels import (
        DOMAIN_VALUES,
        PREDICTOR_DOMAIN_MAP,
        derive_group_domain_predictor,
        infer_predictor_from_text,
        predictor_prompt_catalog,
    )
    from ..models import (
        BlockRoutingResult,
        EffectResult,
        EffectResultSpec,
        EffectsComputationResult,
        ExtractedTable,
        IngestArtifacts,
        TextPassage,
    )
    from ..text_normalize import normalize_inline_text
except ImportError:  # pragma: no cover
    from src.effect_labels import (  # type: ignore
        DOMAIN_VALUES,
        PREDICTOR_DOMAIN_MAP,
        derive_group_domain_predictor,
        infer_predictor_from_text,
        predictor_prompt_catalog,
    )
    from src.models import (  # type: ignore
        BlockRoutingResult,
        EffectResult,
        EffectResultSpec,
        EffectsComputationResult,
        ExtractedTable,
        IngestArtifacts,
        TextPassage,
    )
    from src.text_normalize import normalize_inline_text  # type: ignore


TEXT_EFFECT_RE = re.compile(
    r"(?P<label>cohen'?s?\s*d|hedges?\s*g|smd|standardi[sz]ed mean difference|\bd\b|\bg\b)\s*(?:=|:)\s*(?P<value>[+-]?\s*(?:\d+\.\d+|\d+|\.\d+))",
    re.IGNORECASE,
)
CI_RE = re.compile(
    r"(?:95\s*%?\s*ci|confidence interval|ci)\s*[:=\[\(\s]*"
    r"(?P<low>[+-]?(?:\d+\.\d+|\d+|\.\d+))\s*(?:,|to|;)\s*(?P<high>[+-]?(?:\d+\.\d+|\d+|\.\d+))",
    re.IGNORECASE,
)
TIMEPOINT_RE = re.compile(r"\b(baseline|pre[- ]?test|post[- ]?intervention|post[- ]?test|follow[- ]?up)\b", re.IGNORECASE)
REFERENCE_LIKE_RE = re.compile(r"\breferences?\b|\bbibliography\b|\bet al\.\b", re.IGNORECASE)
CITATION_RE = re.compile(r"\([A-Z][A-Za-z'`-]+[^)]*\b(19|20)\d{2}\b[^)]*\)")

# ── p-value extraction ──
P_VALUE_RE = re.compile(
    r"(?:p|p[- ]?value)\s*(?:[=<>≤≥])\s*(?P<pval>\.?\d+(?:\.\d+)?(?:\s*[×x]\s*10\s*[−-]\s*\d+)?)",
    re.IGNORECASE,
)
P_THRESHOLD_RE = re.compile(
    r"p\s*<\s*(?P<thresh>\.?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# ── sample size extraction ──
SAMPLE_SIZE_RE = re.compile(
    r"(?:(?:(?:total|overall)\s+)?(?:sample\s+)?[Nn]\s*=\s*(?P<n>\d[\d,]*)"
    r"|\b(?P<n2>\d[\d,]*)\s+(?:participants?|subjects?|individuals?|respondents?|couples?|families|dyads?))",
    re.IGNORECASE,
)

OUTCOME_AROUND_EFFECT_RE = re.compile(
    r"(?P<predictor>[A-Za-z][A-Za-z0-9' \-/]{2,100})\s*\(\s*(?:cohen'?s?\s*d|hedges?\s*g|smd|\bd\b|\bg\b)\s*(?:=|:)",
    re.IGNORECASE,
)
TABLE_HEADER_SPLIT_RE = re.compile(r"[\s_/\\\-]+")
MODEL_STAT_ROW_RE = re.compile(
    r"\b(intercept|linear|quadratic|deviance|log[- ]?likelihood|random effect|df|chi2|χ2|parameter)\b",
    re.IGNORECASE,
)

ALLOWED_HEADER_MAP = {
    "d": "d",
    "cohensd": "d",
    "cohend": "d",
    "hedgesg": "g",
    "g": "g",
    "smd": "SMD",
    "standardizedmeandifference": "SMD",
    "standardisedmeandifference": "SMD",
}
BANNED_HEADER_KEYS = {
    "b",
    "beta",
    "se",
    "sd",
    "r",
    "corr",
    "correlation",
}

MAX_ABS_EFFECT_SIZE = 1.5


def _extract_p_value(text: str) -> Optional[float]:
    """Extract the most relevant p-value from text near an effect size mention."""
    # Try exact p-value first (p = 0.032)
    m = P_VALUE_RE.search(text)
    if m:
        raw = m.group("pval").strip()
        # Handle scientific notation (3.2 × 10−4)
        raw = re.sub(r"\s*[×x]\s*10\s*[−-]\s*", "e-", raw)
        try:
            val = float(raw)
            if 0 < val <= 1:
                return round(val, 6)
        except (ValueError, OverflowError):
            pass
    # Try threshold (p < .05, p < .01, p < .001)
    m = P_THRESHOLD_RE.search(text)
    if m:
        try:
            val = float(m.group("thresh"))
            if 0 < val <= 1:
                return round(val, 6)
        except (ValueError, OverflowError):
            pass
    return None


def _extract_sample_size(text: str) -> Optional[int]:
    """Extract total sample size N from text near an effect."""
    best_n: Optional[int] = None
    for m in SAMPLE_SIZE_RE.finditer(text):
        raw = m.group("n") or m.group("n2")
        if raw:
            try:
                n = int(raw.replace(",", ""))
                if 2 <= n <= 500_000:
                    if best_n is None or n > best_n:
                        best_n = n
            except ValueError:
                pass
    return best_n


def run(
    ingest_artifacts: IngestArtifacts,
    block_routing: BlockRoutingResult,
    output_dir: str | Path,
    paper_id: str,
    use_openai_extraction: bool = False,
    openai_model: str = "gpt-4.1-mini",
    openai_api_base: str = "https://api.openai.com/v1",
    openai_timeout_seconds: int = 45,
    openai_extraction_max_snippets: int = 40,
    openai_effect_snippet_chars: int = 1200,
) -> EffectsComputationResult:
    passage_by_id = {item.evidence_id: item for item in ingest_artifacts.text_index}
    page_to_evidence = _build_page_to_evidence_map(ingest_artifacts.text_index)
    text_context_by_id = _build_neighbor_context_map(ingest_artifacts.text_index)

    selected_ids = [evidence_id for evidence_id in block_routing.relevant_effect_blocks if evidence_id in passage_by_id]

    notes: List[str] = []
    extracted: List[EffectResult] = []

    table_effects = _extract_from_structured_tables(
        tables=ingest_artifacts.tables,
        page_to_evidence=page_to_evidence,
        paper_id=paper_id,
    )
    extracted.extend(table_effects)
    notes.append(f"deterministic_table_effects={len(table_effects)}")

    text_effects = _extract_from_text_passages(
        selected_ids=selected_ids,
        passage_by_id=passage_by_id,
        paper_id=paper_id,
    )
    extracted.extend(text_effects)
    notes.append(f"deterministic_text_effects={len(text_effects)}")

    llm_note = "LLM effects extraction disabled."
    if use_openai_extraction and selected_ids:
        llm_text_effects, llm_note = _extract_with_openai_text(
            selected_ids=selected_ids,
            passage_by_id=passage_by_id,
            paper_id=paper_id,
            model=openai_model,
            api_base=openai_api_base,
            timeout_seconds=openai_timeout_seconds,
            batch_size=openai_extraction_max_snippets,
            snippet_chars=openai_effect_snippet_chars,
        )
        extracted.extend(llm_text_effects)
    notes.append(llm_note)

    vision_note = "LLM table vision disabled."
    if use_openai_extraction:
        vision_effects, vision_note = _extract_with_openai_table_images(
            tables=ingest_artifacts.tables,
            page_to_evidence=page_to_evidence,
            paper_id=paper_id,
            model=openai_model,
            api_base=openai_api_base,
            timeout_seconds=openai_timeout_seconds,
        )
        extracted.extend(vision_effects)
    notes.append(vision_note)

    notes.append(f"agent1_extractor_count={len(extracted)}")

    normalized = _agent_normalize_effects(extracted)
    normalizer_note = "Agent2 deterministic normalizer."
    if use_openai_extraction:
        normalized, normalizer_note = _agent_normalize_effects_with_llm(
            effects=normalized,
            model=openai_model,
            api_base=openai_api_base,
            timeout_seconds=openai_timeout_seconds,
            batch_size=openai_extraction_max_snippets,
        )
    notes.append(normalizer_note)
    notes.append(f"agent2_normalizer_count={len(normalized)}")

    consolidated = _agent_consolidate_effects(normalized, value_tolerance=0.01)
    notes.append(f"agent3_consolidator_count={len(consolidated)}")

    validated = _agent_validate_effects(
        consolidated,
        text_context_by_id=text_context_by_id,
        use_openai_extraction=use_openai_extraction,
        model=openai_model,
        api_base=openai_api_base,
        timeout_seconds=openai_timeout_seconds,
    )
    validated.sort(key=lambda item: abs(item.value) if item.value is not None else -1.0, reverse=True)

    study_count = sum(1 for item in validated if item.effect_scope == "study_effect")
    cited_count = sum(1 for item in validated if item.effect_scope == "literature_cited")
    model_stat_count = sum(1 for item in validated if item.effect_scope == "model_stat")
    notes.append(f"agent4_validator_count={len(validated)}")
    notes.append(f"effects_extracted={len(validated)}")
    notes.append(f"study_effects={study_count}")
    notes.append(f"literature_cited_effects={cited_count}")
    notes.append(f"model_stats_excluded={model_stat_count}")
    if not validated:
        notes.append("no_effect_extracted")

    result = EffectsComputationResult(effects=validated, notes=notes)
    _write_json(Path(output_dir) / "04_effects.json", result.model_dump(mode="json"))
    return result


def _extract_from_structured_tables(
    tables: Sequence[ExtractedTable],
    page_to_evidence: Dict[int, str],
    paper_id: str,
) -> List[EffectResult]:
    effects: List[EffectResult] = []
    for table in tables:
        if table.status != "ok" or not table.structured or not table.rows:
            continue

        header_row_index, effect_columns = _detect_effect_columns(table.rows)
        if not effect_columns:
            continue

        for row in table.rows[header_row_index + 1 :]:
            row_text = _row_text(row)
            if not row_text:
                continue
            for column_index, effect_type in effect_columns.items():
                if column_index >= len(row.cells):
                    continue
                value = _extract_numeric_value(row.cells[column_index])
                if value is None:
                    continue
                scope = "study_effect"
                note_label = "deterministic_table_column"
                if _looks_like_model_stat_context(row_text):
                    scope = "model_stat"
                    note_label = "model_stat_row_context"
                if abs(value) > MAX_ABS_EFFECT_SIZE:
                    scope = "model_stat"
                    note_label = "model_stat_out_of_range"
                outcome = _infer_outcome_from_row(row, skip_indexes={column_index})
                comparison = _infer_comparison(row_text)
                group = _infer_group(row_text, comparison)
                timepoint = _infer_timepoint(row_text)
                group_label, domain_label, predictor_label = derive_group_domain_predictor(
                    group_raw=group,
                    predictor_raw=outcome,
                    context=row_text,
                )
                if predictor_label == "unknown":
                    continue
                ci_low, ci_high = _extract_ci(row_text)
                p_val = _extract_p_value(row_text)
                n_size = _extract_sample_size(row_text)
                quote = row_text[:260]
                source_ref = f"{table.table_id}:row_{row.row_index}"
                evidence_id = page_to_evidence.get(table.page)
                spec = EffectResultSpec(
                    outcome=predictor_label,
                    timepoint=timepoint,
                    comparison=comparison,
                    groups=group_label,
                    analysis_set="reported",
                )
                result_id = _build_result_id(
                    paper_id=paper_id,
                    seed=f"{source_ref}|{effect_type}|{value:.4f}",
                    effect_type=effect_type,
                    value=value,
                    outcome=predictor_label,
                    comparison=comparison,
                    timepoint=timepoint,
                    scope=scope,
                )
                effects.append(
                    EffectResult(
                        result_id=result_id,
                        result_spec=spec,
                        design_level="unknown",
                        effect_role="unclear",
                        grouping_label=group_label,
                        outcome_label_normalized=domain_label,
                        timepoint_label_normalized=_normalize_timepoint(timepoint),
                        canonical_key=result_id,
                        stat_consistency="unknown",
                        dedup_sources=1,
                        effect_type=effect_type,
                        effect_scope=scope,
                        origin="reported",
                        source_kind="table",
                        source_page=table.page,
                        source_ref=source_ref,
                        quote=quote,
                        value=value,
                        ci_low=ci_low,
                        ci_high=ci_high,
                        p_value=p_val,
                        sample_size=n_size,
                        derivation_method="reported",
                        calc_confidence="exact",
                        evidence_ids=[evidence_id] if evidence_id else [],
                        notes=[note_label],
                    )
                )
    return effects


def _extract_from_text_passages(
    selected_ids: Sequence[str],
    passage_by_id: Dict[str, TextPassage],
    paper_id: str,
) -> List[EffectResult]:
    effects: List[EffectResult] = []
    for evidence_id in selected_ids:
        passage = passage_by_id[evidence_id]
        text = normalize_inline_text(passage.text, normalize_decimal_comma=True)
        if not text:
            continue

        for index, match in enumerate(TEXT_EFFECT_RE.finditer(text), start=1):
            effect_type = _normalize_effect_type(match.group("label"))
            if effect_type == "unknown":
                continue
            value = _to_optional_float(match.group("value"))
            if value is None:
                continue

            window_left = max(0, match.start() - 150)
            window_right = min(len(text), match.end() + 180)
            window = text[window_left:window_right]
            anchor_in_window = match.start() - window_left

            sentence_left, sentence_right = _sentence_bounds(text, match.start(), match.end())
            sentence = text[sentence_left:sentence_right]
            anchor_in_sentence = match.start() - sentence_left

            outcome = infer_predictor_from_text(sentence, anchor_char=anchor_in_sentence)
            if outcome == "unknown":
                outcome = infer_predictor_from_text(window, anchor_char=anchor_in_window)
            if outcome == "unknown":
                outcome = _infer_outcome(sentence)
            if outcome == "unknown":
                outcome = _infer_outcome(window)
            if outcome == "unknown":
                outcome = _infer_outcome(text)
            comparison = _infer_comparison(window)
            group = _infer_group_from_match_context(text=text, match_start=match.start(), fallback=_infer_group(window, comparison))
            timepoint = _infer_timepoint(window)
            group_label, domain_label, predictor_label = derive_group_domain_predictor(
                group_raw=group,
                predictor_raw=outcome,
                context=window,
            )
            if predictor_label == "unknown":
                continue
            ci_low, ci_high = _extract_ci(window)
            p_val = _extract_p_value(window)
            n_size = _extract_sample_size(window) or _extract_sample_size(text)
            scope = _infer_effect_scope(passage, window)
            note_label = "deterministic_text_label"
            if abs(value) > MAX_ABS_EFFECT_SIZE:
                scope = "model_stat"
                note_label = "model_stat_out_of_range"
            quote = _build_quote(text, match.start(), match.end())

            spec = EffectResultSpec(
                outcome=predictor_label,
                timepoint=timepoint,
                comparison=comparison,
                groups=group_label,
                analysis_set="reported",
            )
            result_id = _build_result_id(
                paper_id=paper_id,
                seed=f"{evidence_id}_{index}",
                effect_type=effect_type,
                value=value,
                outcome=predictor_label,
                comparison=comparison,
                timepoint=timepoint,
                scope=scope,
            )
            effects.append(
                EffectResult(
                    result_id=result_id,
                    result_spec=spec,
                    design_level="unknown",
                    effect_role="unclear",
                    grouping_label=group_label,
                    outcome_label_normalized=domain_label,
                    timepoint_label_normalized=_normalize_timepoint(timepoint),
                    canonical_key=result_id,
                    stat_consistency="unknown",
                    dedup_sources=1,
                    effect_type=effect_type,
                    effect_scope=scope,
                    origin="reported",
                    source_kind="text",
                    source_page=passage.page,
                    source_ref=evidence_id,
                    quote=quote,
                    value=value,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    p_value=p_val,
                    sample_size=n_size,
                    derivation_method="reported",
                    calc_confidence="exact",
                    evidence_ids=[evidence_id],
                    notes=[note_label],
                )
            )
    return effects


def _extract_with_openai_text(
    selected_ids: Sequence[str],
    passage_by_id: Dict[str, TextPassage],
    paper_id: str,
    model: str,
    api_base: str,
    timeout_seconds: int,
    batch_size: int,
    snippet_chars: int,
) -> tuple[List[EffectResult], str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return [], "LLM effects extraction requested but OPENAI_API_KEY is missing."
    try:
        import requests  # type: ignore
    except Exception as exc:
        return [], f"LLM effects extraction skipped: requests unavailable ({_compact_error(exc)})."

    per_call = max(1, int(batch_size))
    total_batches = (len(selected_ids) + per_call - 1) // per_call
    failed_batches = 0
    rejected = 0
    extracted: List[EffectResult] = []
    predictor_catalog = predictor_prompt_catalog()

    for start in range(0, len(selected_ids), per_call):
        batch_ids = selected_ids[start : start + per_call]
        payload_blocks = [
            {
                "evidence_id": evidence_id,
                "page": passage_by_id[evidence_id].page,
                "section_guess": passage_by_id[evidence_id].section_guess,
                "text": normalize_inline_text(passage_by_id[evidence_id].text, normalize_decimal_comma=True)[:snippet_chars],
            }
            for evidence_id in batch_ids
        ]
        parsed = _openai_json_call(
            requests_module=requests,
            api_key=api_key,
            model=model,
            api_base=api_base,
            timeout_seconds=timeout_seconds,
            system_prompt="You extract effect sizes from scientific snippets. Return strict JSON only.",
            user_prompt=(
                "Extract only explicit standardized effect sizes from snippets.\n"
                "Keep precision high: when uncertain, return no item.\n"
                "Do NOT treat regression coefficients (B), correlations (r), SE, SD, chi-square, t, F, p-values as effect sizes.\n"
                "Only accept an item when the value is explicitly labeled as d, Cohen's d, Hedges' g, or SMD, OR from a table column with an exact d/g/SMD header.\n"
                "If text looks tabular, map each value to its header and keep only d/g/SMD columns.\n"
                "Classify effect_scope as study_effect, literature_cited, or model_stat.\n"
                "If abs(value) > 1.5, prefer model_stat unless the snippet clearly states this is an effect size.\n"
                "Output at most one row per unique key (group + predictor + timepoint + effect_scope + effect_type).\n"
                "If several candidate values exist for the same key, keep the best-supported one.\n"
                "Define labels:\n"
                "- group: participant/beneficiary label (not just 'A vs B').\n"
                "- predictor: choose ONLY from the allowed predictor catalog below.\n"
                "- if no allowed predictor matches, set predictor to unknown.\n"
                "- domain: one of Intra-personal | Extra-personal | Material environment and education | Socio-political environment | Work and Activities | Health | unknown.\n"
                "When predictor is unknown, set domain to unknown.\n"
                "Allowed predictor catalog (exact canonical labels):\n"
                f"{predictor_catalog}\n"
                "Positive examples:\n"
                '- "problem intensity (d = 0.77, p < .001)" -> valid effect.\n'
                '- "parents reported lower relationship satisfaction than non-parents (d = -.19)" -> valid effect.\n'
                "Negative examples:\n"
                '- "B = 0.43, SE = 0.10" -> reject.\n'
                '- "r = .31, p < .05" -> reject.\n'
                '- "intercept = 3.98" or a model-parameter row -> model_stat or reject.\n'
                "Schema:\n"
                "{\n"
                '  "effects": [\n'
                "    {\n"
                '      "evidence_id": "string",\n'
                '      "effect_scope": "study_effect|literature_cited|model_stat",\n'
                '      "effect_type": "d|g|SMD|unknown",\n'
                '      "value": 0.0,\n'
                '      "group": "string",\n'
                '      "domain": "Intra-personal|Extra-personal|Material environment and education|Socio-political environment|Work and Activities|Health|unknown",\n'
                '      "predictor": "string",\n'
                '      "outcome": "string",\n'
                '      "group_or_comparator": "string",\n'
                '      "timepoint": "string|unknown",\n'
                '      "ci_low": null,\n'
                '      "ci_high": null,\n'
                '      "p_value": null,\n'
                '      "sample_size": null,\n'
                '      "quote": "exact short quote"\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                f"Blocks JSON:\n{json.dumps(payload_blocks, ensure_ascii=False)}"
            ),
        )
        if not isinstance(parsed, dict):
            failed_batches += 1
            continue
        raw_effects = parsed.get("effects", [])
        if not isinstance(raw_effects, list):
            failed_batches += 1
            continue

        known_ids = {item["evidence_id"] for item in payload_blocks}
        for item in raw_effects:
            effect = _effect_from_llm_text_item(
                item=item,
                known_ids=known_ids,
                passage_by_id=passage_by_id,
                paper_id=paper_id,
            )
            if effect is None:
                rejected += 1
                continue
            extracted.append(effect)

    note = (
        f"LLM text effects extracted={len(extracted)}, rejected={rejected}, "
        f"batches={total_batches}, failed_batches={failed_batches}."
    )
    return extracted, note


def _extract_with_openai_table_images(
    tables: Sequence[ExtractedTable],
    page_to_evidence: Dict[int, str],
    paper_id: str,
    model: str,
    api_base: str,
    timeout_seconds: int,
) -> tuple[List[EffectResult], str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return [], "LLM table vision requested but OPENAI_API_KEY is missing."
    try:
        import requests  # type: ignore
    except Exception as exc:
        return [], f"LLM table vision skipped: requests unavailable ({_compact_error(exc)})."

    candidate_tables = [table for table in tables if table.image_path and table.status == "image_only"]
    if not candidate_tables:
        return [], "LLM table vision skipped: no non-structured table images."

    extracted: List[EffectResult] = []
    failed = 0
    rejected = 0
    predictor_catalog = predictor_prompt_catalog()

    for table in candidate_tables:
        image_path = Path(table.image_path or "")
        if not image_path.exists():
            failed += 1
            continue
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
        parsed = _openai_json_call_vision(
            requests_module=requests,
            api_key=api_key,
            model=model,
            api_base=api_base,
            timeout_seconds=timeout_seconds,
            system_prompt="You extract effect sizes from a table image. Return strict JSON only.",
            user_prompt=(
                "Extract only explicit standardized effect sizes from this table image.\n"
                "High precision required: if uncertain, return no item.\n"
                "Do NOT treat B, beta, r, SE, SD, chi-square, t, F, p-values, intercepts, slopes, or model parameters as effect sizes.\n"
                "Only accept values from headers exactly d / Cohen's d / Hedges g / SMD.\n"
                "If a header mixes metrics (example: 'd r with level at birth') or includes B/SE/SD/r terms, reject it.\n"
                "If no clear d/g/SMD header exists, return an empty effects list.\n"
                "If abs(value) > 1.5, classify as model_stat unless the table clearly states this is a standardized effect size.\n"
                "Output at most one row per unique key (group + predictor + timepoint + effect_scope + effect_type).\n"
                "If several candidate values exist for the same key, keep the best-supported one.\n"
                "Add labels:\n"
                "- group: participant/beneficiary label.\n"
                "- predictor: choose ONLY from the allowed predictor catalog below.\n"
                "- domain: one of Intra-personal | Extra-personal | Material environment and education | Socio-political environment | Work and Activities | Health | unknown.\n"
                "If predictor is unknown, set domain to unknown.\n"
                "Allowed predictor catalog (exact canonical labels):\n"
                f"{predictor_catalog}\n"
                "Schema:\n"
                "{\n"
                '  "effects": [\n'
                "    {\n"
                '      "header_label": "string",\n'
                '      "effect_scope": "study_effect|literature_cited|model_stat",\n'
                '      "effect_type": "d|g|SMD|unknown",\n'
                '      "value": 0.0,\n'
                '      "group": "string",\n'
                '      "domain": "Intra-personal|Extra-personal|Material environment and education|Socio-political environment|Work and Activities|Health|unknown",\n'
                '      "predictor": "string",\n'
                '      "outcome": "string",\n'
                '      "group_or_comparator": "string",\n'
                '      "timepoint": "string|unknown",\n'
                '      "ci_low": null,\n'
                '      "ci_high": null,\n'
                '      "p_value": null,\n'
                '      "sample_size": null,\n'
                '      "quote": "exact short quote from image"\n'
                "    }\n"
                "  ]\n"
                "}\n"
            ),
            image_base64=encoded_image,
        )
        if not isinstance(parsed, dict):
            failed += 1
            continue
        raw_effects = parsed.get("effects", [])
        if not isinstance(raw_effects, list):
            failed += 1
            continue

        evidence_id = page_to_evidence.get(table.page)
        for index, item in enumerate(raw_effects, start=1):
            effect = _effect_from_llm_image_item(
                item=item,
                paper_id=paper_id,
                table=table,
                row_index=index,
                evidence_id=evidence_id,
            )
            if effect is None:
                rejected += 1
                continue
            extracted.append(effect)

    note = f"LLM table vision effects extracted={len(extracted)}, rejected={rejected}, failed={failed}."
    return extracted, note


def _effect_from_llm_text_item(
    item: Any,
    known_ids: set[str],
    passage_by_id: Dict[str, TextPassage],
    paper_id: str,
) -> Optional[EffectResult]:
    if not isinstance(item, dict):
        return None
    evidence_id = normalize_inline_text(str(item.get("evidence_id", "")))
    if evidence_id not in known_ids:
        return None

    passage = passage_by_id[evidence_id]
    source_text = normalize_inline_text(passage.text, normalize_decimal_comma=True)
    effect_type = _normalize_effect_type(str(item.get("effect_type", "unknown")))
    if effect_type == "unknown":
        return None
    value = _to_optional_float(item.get("value"))
    if value is None:
        return None

    quote = normalize_inline_text(str(item.get("quote", "")))
    if quote and quote.lower() not in source_text.lower():
        return None
    if not TEXT_EFFECT_RE.search(source_text):
        return None

    predictor_candidates = [
        normalize_inline_text(str(item.get("outcome", ""))),
        normalize_inline_text(str(item.get("predictor", ""))),
        infer_predictor_from_text(quote) if quote else "unknown",
        infer_predictor_from_text(source_text),
        _infer_outcome(quote) if quote else "unknown",
        _infer_outcome(source_text),
    ]
    predictor_raw = "unknown"
    for candidate in predictor_candidates:
        candidate_clean = normalize_inline_text(candidate)
        if candidate_clean and candidate_clean.lower() != "unknown":
            predictor_raw = candidate_clean
            break
    comparison = normalize_inline_text(str(item.get("group_or_comparator", ""))) or _infer_comparison(source_text)
    group_raw = normalize_inline_text(str(item.get("group", ""))) or _infer_group(source_text, comparison)
    if comparison == "unknown" and group_raw != "unknown":
        comparison = group_raw
    group_label, domain_label, predictor_label = derive_group_domain_predictor(
        group_raw=group_raw,
        predictor_raw=predictor_raw,
        domain_raw=normalize_inline_text(str(item.get("domain", ""))),
        context=source_text,
    )
    if predictor_label == "unknown":
        return None
    timepoint = normalize_inline_text(str(item.get("timepoint", ""))) or _infer_timepoint(source_text)
    ci_low = _to_optional_float(item.get("ci_low"))
    ci_high = _to_optional_float(item.get("ci_high"))
    if ci_low is not None and ci_high is not None and ci_low > ci_high:
        ci_low, ci_high = ci_high, ci_low
    scope = _normalize_scope(str(item.get("effect_scope", ""))) or _infer_effect_scope(passage, source_text)
    note_label = "llm_text_extraction"
    if abs(value) > MAX_ABS_EFFECT_SIZE:
        scope = "model_stat"
        note_label = "model_stat_out_of_range"
    p_val = _to_optional_float(item.get("p_value")) or _extract_p_value(source_text)
    n_size = _to_optional_int(item.get("sample_size")) or _extract_sample_size(source_text)

    spec = EffectResultSpec(
        outcome=predictor_label,
        timepoint=timepoint,
        comparison=comparison,
        groups=group_label,
        analysis_set="reported",
    )
    result_id = _build_result_id(
        paper_id=paper_id,
        seed=f"{evidence_id}|llm|{effect_type}|{value:.4f}",
        effect_type=effect_type,
        value=value,
        outcome=predictor_label,
        comparison=comparison,
        timepoint=timepoint,
        scope=scope,
    )
    return EffectResult(
        result_id=result_id,
        result_spec=spec,
        design_level="unknown",
        effect_role="unclear",
        grouping_label=group_label,
        outcome_label_normalized=domain_label,
        timepoint_label_normalized=_normalize_timepoint(timepoint),
        canonical_key=result_id,
        stat_consistency="unknown",
        dedup_sources=1,
        effect_type=effect_type,
        effect_scope=scope,
        origin="reported",
        source_kind="text",
        source_page=passage.page,
        source_ref=evidence_id,
        quote=quote or _build_quote(source_text, 0, min(80, len(source_text))),
        value=value,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_val,
        sample_size=n_size,
        derivation_method="reported",
        calc_confidence="exact",
        evidence_ids=[evidence_id],
        notes=[note_label],
    )


def _effect_from_llm_image_item(
    item: Any,
    paper_id: str,
    table: ExtractedTable,
    row_index: int,
    evidence_id: Optional[str],
) -> Optional[EffectResult]:
    if not isinstance(item, dict):
        return None
    header_label_raw = normalize_inline_text(str(item.get("header_label", "")))
    if not header_label_raw:
        return None
    header_effect_type = _effect_type_from_header(header_label_raw)
    if not header_effect_type:
        return None
    effect_type = header_effect_type
    value = _to_optional_float(item.get("value"))
    if value is None:
        return None
    quote = normalize_inline_text(str(item.get("quote", "")))[:260]
    predictor_candidates = [
        normalize_inline_text(str(item.get("outcome", ""))),
        normalize_inline_text(str(item.get("predictor", ""))),
        infer_predictor_from_text(quote),
        _infer_outcome(quote),
    ]
    predictor_raw = "unknown"
    for candidate in predictor_candidates:
        candidate_clean = normalize_inline_text(candidate)
        if candidate_clean and candidate_clean.lower() != "unknown":
            predictor_raw = candidate_clean
            break
    comparison = normalize_inline_text(str(item.get("group_or_comparator", ""))) or "unknown"
    group_raw = normalize_inline_text(str(item.get("group", ""))) or _infer_group(comparison, comparison)
    if comparison == "unknown" and group_raw != "unknown":
        comparison = group_raw
    group_label, domain_label, predictor_label = derive_group_domain_predictor(
        group_raw=group_raw,
        predictor_raw=predictor_raw,
        domain_raw=normalize_inline_text(str(item.get("domain", ""))),
        context=f"{header_label_raw} {quote}",
    )
    if predictor_label == "unknown":
        return None
    timepoint = normalize_inline_text(str(item.get("timepoint", ""))) or "unknown"
    source_ref = f"{table.table_id}:vision_{row_index}"
    raw_scope = _normalize_scope(str(item.get("effect_scope", "")))
    scope = raw_scope or _llm_image_scope(outcome=predictor_label, comparison=comparison, quote=quote, value=value)

    spec = EffectResultSpec(
        outcome=predictor_label,
        timepoint=timepoint,
        comparison=comparison,
        groups=group_label,
        analysis_set="reported",
    )
    result_id = _build_result_id(
        paper_id=paper_id,
        seed=f"{source_ref}|{effect_type}|{value:.4f}",
        effect_type=effect_type,
        value=value,
        outcome=predictor_label,
        comparison=comparison,
        timepoint=timepoint,
        scope=scope,
    )
    p_val = _to_optional_float(item.get("p_value"))
    n_size = _to_optional_int(item.get("sample_size"))
    if abs(value) > MAX_ABS_EFFECT_SIZE:
        note_label = "model_stat_out_of_range"
        scope = "model_stat"
    elif scope == "model_stat":
        note_label = "model_stat_row_context"
    else:
        note_label = "llm_table_vision"
    return EffectResult(
        result_id=result_id,
        result_spec=spec,
        design_level="unknown",
        effect_role="unclear",
        grouping_label=group_label,
        outcome_label_normalized=domain_label,
        timepoint_label_normalized=_normalize_timepoint(timepoint),
        canonical_key=result_id,
        stat_consistency="unknown",
        dedup_sources=1,
        effect_type=effect_type,
        effect_scope=scope,
        origin="reported",
        source_kind="table_image_vision",
        source_page=table.page,
        source_ref=source_ref,
        quote=quote,
        value=value,
        ci_low=_to_optional_float(item.get("ci_low")),
        ci_high=_to_optional_float(item.get("ci_high")),
        p_value=p_val,
        sample_size=n_size,
        derivation_method="reported",
        calc_confidence="exact",
        evidence_ids=[evidence_id] if evidence_id else [],
        notes=[note_label],
    )


def _detect_effect_columns(rows: Sequence[Any]) -> Tuple[int, Dict[int, str]]:
    best_header_index = 0
    best_columns: Dict[int, str] = {}
    for header_idx in range(min(3, len(rows))):
        header_row = rows[header_idx]
        columns: Dict[int, str] = {}
        for cell_idx, cell in enumerate(header_row.cells):
            effect_type = _effect_type_from_header(cell)
            if effect_type:
                columns[cell_idx] = effect_type
        if len(columns) > len(best_columns):
            best_columns = columns
            best_header_index = header_idx
    return best_header_index, best_columns


def _effect_type_from_header(cell: str) -> str:
    normalized = _normalize_header_key(cell)
    if not normalized:
        return ""
    if normalized in BANNED_HEADER_KEYS:
        return ""
    return ALLOWED_HEADER_MAP.get(normalized, "")


def _normalize_header_key(value: str) -> str:
    text = normalize_inline_text(value, normalize_decimal_comma=False).lower()
    text = text.replace("\u2019", "'").replace("`", "'")
    text = re.sub(r"[^a-z0-9\s']", "", text)
    parts = [part for part in TABLE_HEADER_SPLIT_RE.split(text) if part]
    return "".join(parts)


def _extract_numeric_value(value: str) -> Optional[float]:
    text = normalize_inline_text(value, normalize_decimal_comma=True)
    match = re.search(r"[+-]?(?:\d+\.\d+|\d+|\.\d+)", text)
    if not match:
        return None
    return _to_optional_float(match.group(0))


def _extract_ci(text: str) -> tuple[Optional[float], Optional[float]]:
    match = CI_RE.search(text)
    if not match:
        return None, None
    low = _to_optional_float(match.group("low"))
    high = _to_optional_float(match.group("high"))
    if low is not None and high is not None and low > high:
        low, high = high, low
    return low, high


def _row_text(row: Any) -> str:
    return normalize_inline_text(" | ".join(cell for cell in row.cells if cell))


def _infer_outcome_from_row(row: Any, skip_indexes: set[int]) -> str:
    for idx, cell in enumerate(row.cells):
        if idx in skip_indexes:
            continue
        clean = normalize_inline_text(cell)
        if not clean:
            continue
        if re.fullmatch(r"[+-]?(?:\d+\.\d+|\d+|\.\d+)", clean):
            continue
        if len(clean) >= 3:
            return clean
    return "unknown"


def _build_page_to_evidence_map(passages: Sequence[TextPassage]) -> Dict[int, str]:
    page_to_evidence: Dict[int, str] = {}
    for passage in passages:
        if passage.page not in page_to_evidence:
            page_to_evidence[passage.page] = passage.evidence_id
    return page_to_evidence


def _build_neighbor_context_map(passages: Sequence[TextPassage]) -> Dict[str, str]:
    contexts: Dict[str, str] = {}
    cleaned = [
        normalize_inline_text(item.text, normalize_decimal_comma=True)
        for item in passages
    ]
    total = len(passages)
    for idx, passage in enumerate(passages):
        prev_text = cleaned[idx - 1] if idx > 0 else ""
        curr_text = cleaned[idx]
        next_text = cleaned[idx + 1] if (idx + 1) < total else ""
        parts: List[str] = []
        if prev_text:
            parts.append(f"[PREV] {prev_text[:700]}")
        if curr_text:
            parts.append(f"[CURR] {curr_text[:900]}")
        if next_text:
            parts.append(f"[NEXT] {next_text[:700]}")
        contexts[passage.evidence_id] = "\n".join(parts)[:2400]
    return contexts


def _infer_effect_scope(passage: TextPassage, text: str) -> str:
    section = passage.section_guess.lower().strip()
    if section in {"introduction", "background", "references"}:
        return "literature_cited"
    lower = text.lower()
    if REFERENCE_LIKE_RE.search(text) and CITATION_RE.search(text):
        return "literature_cited"
    if any(term in lower for term in ["meta-analysis", "cross-sectional", "previous studies", "reported by"]):
        return "literature_cited"
    return "study_effect"


def _normalize_scope(scope: str) -> str:
    lowered = normalize_inline_text(scope).lower()
    if lowered in {"study_effect", "literature_cited", "model_stat"}:
        return lowered
    return ""


def _looks_like_model_stat_context(text: str) -> bool:
    return bool(MODEL_STAT_ROW_RE.search(text))


def _llm_image_scope(outcome: str, comparison: str, quote: str, value: float) -> str:
    if abs(value) > MAX_ABS_EFFECT_SIZE:
        return "model_stat"
    merged = " | ".join([outcome, comparison, quote])
    if _looks_like_model_stat_context(merged):
        return "model_stat"
    return "study_effect"


def _build_quote(text: str, start: int, end: int) -> str:
    left_bound = max(0, start - 180)
    right_bound = min(len(text), end + 180)
    window = text[left_bound:right_bound]

    rel_start = max(0, start - left_bound)
    rel_end = min(len(window), end - left_bound)

    left_cut = 0
    for marker in [". ", "; ", ": ", "\n"]:
        idx = window.rfind(marker, 0, rel_start)
        if idx > left_cut:
            left_cut = idx + len(marker)

    right_cut = len(window)
    for marker in [". ", "; ", ": ", "\n"]:
        idx = window.find(marker, rel_end)
        if idx != -1:
            right_cut = min(right_cut, idx + len(marker))

    quote = normalize_inline_text(window[left_cut:right_cut])
    if len(quote) > 240:
        quote = quote[:240].rstrip()
    return quote


def _sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    left = 0
    right = len(text)
    boundary_re = re.compile(r"[.!?](?:\d+)?\s+(?=[A-Z])|\n+")
    for match in boundary_re.finditer(text):
        if match.end() <= start:
            left = match.end()
            continue
        if match.start() >= end:
            right = match.start()
            break

    if right <= left:
        return max(0, start - 120), min(len(text), end + 120)
    return left, right


def _normalize_effect_type(raw: str) -> str:
    text = normalize_inline_text(raw).lower()
    if "hedges" in text or text.strip() == "g":
        return "g"
    if "cohen" in text or text.strip() == "d":
        return "d"
    if "smd" in text or "standardized mean difference" in text or "standardised mean difference" in text:
        return "SMD"
    return "unknown"


def _normalize_label(raw: str) -> str:
    text = normalize_inline_text(raw).lower()
    if "depress" in text:
        return "depression"
    if "anxiety" in text:
        return "anxiety"
    if "stress" in text:
        return "stress"
    if "satisfaction" in text:
        return "satisfaction"
    if "confidence" in text:
        return "confidence"
    if "conflict" in text:
        return "conflict"
    return text if text else "unknown"


def _normalize_timepoint(raw: str) -> str:
    text = normalize_inline_text(raw).lower()
    if ("post-birth" in text and "pre-birth" in text) or ("post birth" in text and "pre birth" in text):
        return "transition effect"
    if "transition to parenthood" in text or "transition effect" in text:
        return "transition effect"
    if "follow" in text:
        return "follow-up"
    if "post" in text:
        return "post"
    if "baseline" in text or "pre" in text:
        return "baseline"
    return text if text else "unknown"


def _infer_outcome(text: str) -> str:
    normalized = normalize_inline_text(text, normalize_decimal_comma=True)
    candidates: List[str] = []
    for match in OUTCOME_AROUND_EFFECT_RE.finditer(normalized):
        candidate = _clean_outcome_candidate(match.group("predictor"))
        if candidate != "unknown":
            candidates.append(candidate)
    if candidates:
        return candidates[-1]
    lower = normalized.lower()
    for key in [
        "relationship satisfaction",
        "problem intensity",
        "negative communication",
        "relationship confidence",
        "relationship dedication",
        "depression",
        "anxiety",
        "stress",
        "self-efficacy",
        "conflict",
    ]:
        if key in lower:
            return key
    return "unknown"


def _clean_outcome_candidate(raw: str) -> str:
    text = normalize_inline_text(raw)
    text = re.sub(r"^[^A-Za-z]+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;:.")
    if not text:
        return "unknown"
    fragments = [frag.strip() for frag in re.split(r"[;,:]", text) if frag.strip()]
    if fragments:
        text = fragments[-1]
    text = re.sub(
        r"^(reported|showed|demonstrated|found|observed|indicated|revealed|significant(?:ly)?|sudden(?:ly)?)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    marker_match = re.search(r"\b(in|on|for)\s+([A-Za-z][A-Za-z0-9' \-/]{2,80})$", text, flags=re.IGNORECASE)
    if marker_match:
        text = marker_match.group(2)
    words = text.split()
    if len(words) > 8:
        text = " ".join(words[-8:])
    text = text.strip(" ,;:.")
    if not text:
        return "unknown"
    lowered = text.lower()
    if lowered in {"overall", "effect", "effect size", "study", "result"}:
        return "unknown"
    return text


def _infer_comparison(text: str) -> str:
    lower = text.lower()
    if "control" in lower and "intervention" in lower:
        return "intervention vs control"
    if "mothers" in lower and "fathers" in lower:
        return "mothers vs fathers"
    if "control" in lower:
        return "vs control"
    return "unknown"


def _infer_group(text: str, comparison: str) -> str:
    lower = text.lower()
    has_mothers = bool(re.search(r"\bmothers?\b", lower))
    has_fathers = bool(re.search(r"\bfathers?\b", lower))
    has_parents = bool(re.search(r"\bparents?\b", lower))
    has_women = bool(re.search(r"\bwomen\b", lower))
    has_men = bool(re.search(r"\bmen\b", lower))
    has_participants = bool(re.search(r"\bparticipants?\b", lower))

    if has_mothers and has_fathers:
        return "mothers vs fathers"
    if has_parents:
        return "parents"
    if has_women and has_men:
        return "women and men"
    if has_women:
        return "women"
    if has_men:
        return "men"
    if has_participants:
        return "participants"
    if comparison and comparison != "unknown":
        return comparison
    return "unknown"


def _infer_group_from_match_context(text: str, match_start: int, fallback: str) -> str:
    left = normalize_inline_text(text[max(0, match_start - 260) : match_start]).lower()
    if re.search(r"\bmothers?\s*,?\s*but not\s*fathers?\b", left):
        return "mothers"
    if re.search(r"\bfathers?\s*,?\s*but not\s*mothers?\b", left):
        return "fathers"

    direct_matches = list(
        re.finditer(
        r"\b(mothers?|fathers?|parents?|women|men|participants)\b[^A-Za-z0-9]{0,10}\(\s*$",
        left,
        flags=re.IGNORECASE,
        )
    )
    if direct_matches:
        return normalize_inline_text(direct_matches[-1].group(1)).lower()

    pattern = re.compile(r"\b(?:for|among|in)\s+(mothers?|fathers?|parents?|women|men|participants)\b")
    matches = list(pattern.finditer(left))
    if matches:
        return normalize_inline_text(matches[-1].group(1)).lower()

    explicit = re.compile(r"\b(mothers?|fathers?|parents?|women|men|participants)\s*\(\s*(?:cohen|hedges|d|g|smd)", re.IGNORECASE)
    matches_explicit = list(explicit.finditer(left))
    if matches_explicit:
        return normalize_inline_text(matches_explicit[-1].group(1)).lower()

    labels = ["mothers", "fathers", "parents", "women", "men", "participants"]
    present = [label for label in labels if re.search(rf"\b{re.escape(label)}\b", left)]
    if len(present) == 1:
        return present[0]
    return fallback


def _infer_timepoint(text: str) -> str:
    match = TIMEPOINT_RE.search(text)
    if not match:
        return "unknown"
    return normalize_inline_text(match.group(1)).lower()


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = normalize_inline_text(str(value), normalize_decimal_comma=True).replace(" ", "")
    if text.startswith("."):
        text = "0" + text
    if text.startswith("-."):
        text = "-0" + text[1:]
    try:
        return float(text)
    except Exception:
        return None


def _to_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == int(value) else None
    text = normalize_inline_text(str(value)).replace(" ", "").replace(",", "")
    try:
        return int(float(text))
    except Exception:
        return None


def _agent_normalize_effects(effects: Sequence[EffectResult]) -> List[EffectResult]:
    normalized: List[EffectResult] = []
    for effect in effects:
        effect_type = _normalize_effect_type(effect.effect_type)
        if effect_type == "unknown":
            continue

        value = float(effect.value) if effect.value is not None else None
        ci_low = float(effect.ci_low) if effect.ci_low is not None else None
        ci_high = float(effect.ci_high) if effect.ci_high is not None else None
        if ci_low is not None and ci_high is not None and ci_low > ci_high:
            ci_low, ci_high = ci_high, ci_low

        quote = normalize_inline_text(effect.quote)
        if len(quote) > 260:
            quote = quote[:260].rstrip()

        group, domain, predictor = derive_group_domain_predictor(
            group_raw=effect.grouping_label or effect.result_spec.groups,
            predictor_raw=effect.result_spec.outcome,
            domain_raw=effect.outcome_label_normalized,
            context=quote,
        )

        comparison = normalize_inline_text(effect.result_spec.comparison)
        if _is_unknown_token(comparison) and not _is_unknown_token(group):
            comparison = group
        if _is_unknown_token(comparison):
            comparison = "unknown"

        timepoint = _normalize_timepoint(effect.timepoint_label_normalized or effect.result_spec.timepoint)

        spec = effect.result_spec.model_copy(
            update={
                "outcome": predictor,
                "groups": group,
                "comparison": comparison,
                "timepoint": timepoint,
            }
        )
        normalized.append(
            effect.model_copy(
                update={
                    "effect_type": effect_type,
                    "grouping_label": group,
                    "outcome_label_normalized": domain,
                    "timepoint_label_normalized": timepoint,
                    "quote": quote,
                    "value": value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "result_spec": spec,
                }
            )
        )
    return normalized


def _agent_normalize_effects_with_llm(
    effects: Sequence[EffectResult],
    model: str,
    api_base: str,
    timeout_seconds: int,
    batch_size: int,
) -> tuple[List[EffectResult], str]:
    if not effects:
        return [], "Agent2 LLM normalizer skipped: no effects."

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return list(effects), "Agent2 LLM normalizer skipped: OPENAI_API_KEY missing."
    try:
        import requests  # type: ignore
    except Exception as exc:
        return list(effects), f"Agent2 LLM normalizer skipped: requests unavailable ({_compact_error(exc)})."

    normalized = list(effects)
    per_call = max(8, min(int(batch_size), 32))
    total_batches = (len(normalized) + per_call - 1) // per_call
    failed_batches = 0
    updates_count = 0
    predictor_catalog = predictor_prompt_catalog()

    for start in range(0, len(normalized), per_call):
        batch = normalized[start : start + per_call]
        payload = []
        for offset, effect in enumerate(batch):
            payload.append(
                {
                    "idx": start + offset,
                    "effect_scope": effect.effect_scope,
                    "source_kind": effect.source_kind,
                    "effect_type": effect.effect_type,
                    "value": effect.value,
                    "group": effect.grouping_label,
                    "domain": effect.outcome_label_normalized,
                    "predictor": effect.result_spec.outcome,
                    "timepoint": effect.timepoint_label_normalized or effect.result_spec.timepoint,
                    "quote": effect.quote[:220],
                }
            )

        parsed = _openai_json_call(
            requests_module=requests,
            api_key=api_key,
            model=model,
            api_base=api_base,
            timeout_seconds=timeout_seconds,
            system_prompt="You normalize scientific effect labels. Return strict JSON only.",
            user_prompt=(
                "Normalize labels without changing the scientific meaning.\n"
                "Do not invent new effects and do not modify value/effect_type/scope.\n"
                "Goals:\n"
                "- Standardize group labels: example mothers -> Mothers.\n"
                "- Standardize predictor labels: example observed negative communication -> Negative Communication.\n"
                "- Standardize timepoint labels: example post-birth vs pre-birth -> Transition effect.\n"
                "- Predictor must be chosen only from the allowed predictor catalog.\n"
                "- If no valid predictor exists in the catalog, use predictor=unknown and domain=unknown.\n"
                "- Keep at most one canonical row per key (group + predictor + timepoint + effect_scope + effect_type).\n"
                "- Keep domain in: Intra-personal | Extra-personal | Material environment and education | "
                "Socio-political environment | Work and Activities | Health | unknown.\n"
                "Allowed predictor catalog (exact canonical labels):\n"
                f"{predictor_catalog}\n"
                "If uncertain, keep unknown.\n"
                "Schema:\n"
                "{\n"
                '  "items": [\n'
                "    {\n"
                '      "idx": 0,\n'
                '      "group": "string",\n'
                '      "domain": "Intra-personal|Extra-personal|Material environment and education|Socio-political environment|Work and Activities|Health|unknown",\n'
                '      "predictor": "string",\n'
                '      "timepoint": "string"\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                f"Rows JSON:\n{json.dumps(payload, ensure_ascii=False)}"
            ),
        )
        if not isinstance(parsed, dict):
            failed_batches += 1
            continue
        raw_items = parsed.get("items", [])
        if not isinstance(raw_items, list):
            failed_batches += 1
            continue

        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            idx_value = raw.get("idx")
            if not isinstance(idx_value, int):
                continue
            if idx_value < 0 or idx_value >= len(normalized):
                continue

            current = normalized[idx_value]
            raw_group = normalize_inline_text(str(raw.get("group", "")))
            raw_domain = normalize_inline_text(str(raw.get("domain", "")))
            raw_predictor = normalize_inline_text(str(raw.get("predictor", "")))
            raw_timepoint = normalize_inline_text(str(raw.get("timepoint", "")))

            group, domain, predictor = derive_group_domain_predictor(
                group_raw=raw_group or current.grouping_label,
                predictor_raw=raw_predictor or current.result_spec.outcome,
                domain_raw=raw_domain or current.outcome_label_normalized,
                context=current.quote,
            )
            if predictor == "unknown":
                continue
            if domain not in DOMAIN_VALUES:
                domain = "unknown"

            timepoint = _normalize_timepoint(raw_timepoint or current.timepoint_label_normalized or current.result_spec.timepoint)
            comparison = normalize_inline_text(current.result_spec.comparison)
            if _is_unknown_token(comparison) and not _is_unknown_token(group):
                comparison = group
            if _is_unknown_token(comparison):
                comparison = "unknown"

            old_signature = (
                current.grouping_label,
                current.outcome_label_normalized,
                current.result_spec.outcome,
                current.timepoint_label_normalized,
                current.result_spec.comparison,
            )

            updated_spec = current.result_spec.model_copy(
                update={
                    "groups": group,
                    "outcome": predictor,
                    "timepoint": timepoint,
                    "comparison": comparison,
                }
            )
            updated = current.model_copy(
                update={
                    "grouping_label": group,
                    "outcome_label_normalized": domain,
                    "timepoint_label_normalized": timepoint,
                    "result_spec": updated_spec,
                }
            )
            new_signature = (
                updated.grouping_label,
                updated.outcome_label_normalized,
                updated.result_spec.outcome,
                updated.timepoint_label_normalized,
                updated.result_spec.comparison,
            )
            if new_signature != old_signature:
                updates_count += 1
            normalized[idx_value] = updated

    note = (
        f"Agent2 LLM normalizer applied: updated={updates_count}, "
        f"batches={total_batches}, failed_batches={failed_batches}."
    )
    return normalized, note


def _agent_consolidate_effects(
    effects: Sequence[EffectResult],
    value_tolerance: float = 0.01,
) -> List[EffectResult]:
    clusters: List[List[EffectResult]] = []
    ordered = sorted(effects, key=_effect_rank_tuple)
    for effect in ordered:
        placed = False
        for cluster in clusters:
            if _is_effect_compatible(effect, cluster[0], value_tolerance=value_tolerance):
                cluster.append(effect)
                placed = True
                break
        if not placed:
            clusters.append([effect])
    return [_merge_effect_cluster(cluster) for cluster in clusters]


def _agent_validate_effects(
    effects: Sequence[EffectResult],
    text_context_by_id: Dict[str, str],
    use_openai_extraction: bool,
    model: str,
    api_base: str,
    timeout_seconds: int,
) -> List[EffectResult]:
    validated: List[EffectResult] = []
    for effect in effects:
        current = effect
        if current.value is None:
            continue

        predictor = normalize_inline_text(current.result_spec.outcome)
        if predictor not in PREDICTOR_DOMAIN_MAP:
            continue
        expected_domain = PREDICTOR_DOMAIN_MAP.get(predictor, "unknown")
        if current.outcome_label_normalized != expected_domain:
            current = current.model_copy(
                update={
                    "outcome_label_normalized": expected_domain,
                    "notes": _dedupe_preserve(current.notes + ["validator_domain_from_predictor"]),
                }
            )

        if current.effect_scope == "study_effect" and abs(current.value) > MAX_ABS_EFFECT_SIZE:
            current = current.model_copy(
                update={
                    "effect_scope": "model_stat",
                    "notes": _dedupe_preserve(current.notes + ["model_stat_out_of_range"]),
                }
            )

        if current.effect_scope == "study_effect":
            if _is_unknown_token(current.grouping_label) or _is_unknown_token(current.result_spec.outcome):
                continue
            if _is_incomplete_quote(current.quote, current.source_kind):
                continue
            if _is_unknown_token(current.result_spec.comparison):
                current = current.model_copy(
                    update={"result_spec": current.result_spec.model_copy(update={"comparison": current.grouping_label})}
                )

        validated.append(current)
    return _collapse_validator_duplicates(
        validated,
        text_context_by_id=text_context_by_id,
        use_openai_extraction=use_openai_extraction,
        model=model,
        api_base=api_base,
        timeout_seconds=timeout_seconds,
    )


def _collapse_validator_duplicates(
    effects: Sequence[EffectResult],
    text_context_by_id: Dict[str, str],
    use_openai_extraction: bool,
    model: str,
    api_base: str,
    timeout_seconds: int,
) -> List[EffectResult]:
    grouped: Dict[tuple[str, str, str, str], List[EffectResult]] = {}
    for effect in effects:
        key = (
            effect.effect_scope,
            effect.effect_type,
            _normalize_match_key(effect.grouping_label),
            _normalize_match_key(effect.result_spec.outcome),
        )
        grouped.setdefault(key, []).append(effect)

    collapsed: List[EffectResult] = []
    for cluster in grouped.values():
        if len(cluster) == 1:
            collapsed.append(cluster[0])
            continue
        chosen, selector_note = _select_validator_cluster_winner(
            cluster=cluster,
            text_context_by_id=text_context_by_id,
            use_openai_extraction=use_openai_extraction,
            model=model,
            api_base=api_base,
            timeout_seconds=timeout_seconds,
        )
        merged_ids = _dedupe_preserve([evidence_id for item in cluster for evidence_id in item.evidence_ids])
        merged_rows = _dedupe_preserve([row for item in cluster for row in item.table_row_refs])
        merged_notes = _dedupe_preserve(
            [note for item in cluster for note in item.notes]
            + [f"validator_same_key_cluster={len(cluster)}"]
            + ([selector_note] if selector_note else [])
        )
        collapsed.append(
            chosen.model_copy(
                update={
                    "evidence_ids": merged_ids,
                    "table_row_refs": merged_rows,
                    "notes": merged_notes,
                    "dedup_sources": max(chosen.dedup_sources, len(cluster)),
                }
            )
        )
    return collapsed


def _select_validator_cluster_winner(
    cluster: Sequence[EffectResult],
    text_context_by_id: Dict[str, str],
    use_openai_extraction: bool,
    model: str,
    api_base: str,
    timeout_seconds: int,
) -> tuple[EffectResult, str]:
    llm_choice = _validator_pick_with_openai(
        cluster=cluster,
        text_context_by_id=text_context_by_id,
        use_openai_extraction=use_openai_extraction,
        model=model,
        api_base=api_base,
        timeout_seconds=timeout_seconds,
    )
    if llm_choice is not None:
        return llm_choice, "validator_llm_winner"

    anchor = _cluster_anchor_value(cluster)
    ordered = sorted(
        cluster,
        key=lambda item: (
            _effect_rank_tuple(item),
            -_predictor_context_relevance(item, text_context_by_id),
            abs((item.value or 0.0) - anchor) if anchor is not None else 0.0,
            -abs(item.value or 0.0),
            item.source_page,
        ),
    )
    return ordered[0], "validator_deterministic_winner"


def _validator_pick_with_openai(
    cluster: Sequence[EffectResult],
    text_context_by_id: Dict[str, str],
    use_openai_extraction: bool,
    model: str,
    api_base: str,
    timeout_seconds: int,
) -> Optional[EffectResult]:
    if not use_openai_extraction or len(cluster) < 2:
        return None

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        import requests  # type: ignore
    except Exception:
        return None

    payload_items: List[Dict[str, Any]] = []
    for idx, effect in enumerate(cluster):
        context = _effect_context_text(effect, text_context_by_id)
        payload_items.append(
            {
                "idx": idx,
                "result_id": effect.result_id,
                "group": effect.grouping_label,
                "predictor": effect.result_spec.outcome,
                "timepoint": _normalize_timepoint(effect.timepoint_label_normalized or effect.result_spec.timepoint),
                "effect_scope": effect.effect_scope,
                "effect_type": effect.effect_type,
                "value": effect.value,
                "source_kind": effect.source_kind,
                "source_page": effect.source_page,
                "quote": normalize_inline_text(effect.quote)[:260],
                "context_prev_curr_next": context,
            }
        )

    parsed = _openai_json_call(
        requests_module=requests,
        api_key=api_key,
        model=model,
        api_base=api_base,
        timeout_seconds=timeout_seconds,
        system_prompt="You resolve duplicate effect candidates. Return strict JSON only.",
        user_prompt=(
            "All rows below refer to the SAME group and predictor and represent competing extracted effect sizes.\n"
            "Choose the single best supported candidate using quote + surrounding context.\n"
            "Prefer explicit and complete evidence. If two rows conflict and one is less specific or noisier, discard it.\n"
            "Schema:\n"
            "{\n"
            '  "winner_idx": 0,\n'
            '  "reason": "short reason"\n'
            "}\n\n"
            f"Candidates JSON:\n{json.dumps(payload_items, ensure_ascii=False)}"
        ),
    )
    if not isinstance(parsed, dict):
        return None
    winner_idx = parsed.get("winner_idx")
    if not isinstance(winner_idx, int):
        return None
    if winner_idx < 0 or winner_idx >= len(cluster):
        return None
    return cluster[winner_idx]


def _effect_context_text(effect: EffectResult, text_context_by_id: Dict[str, str]) -> str:
    chunks: List[str] = []
    for evidence_id in effect.evidence_ids[:2]:
        context = normalize_inline_text(text_context_by_id.get(evidence_id, ""))
        if context:
            chunks.append(context[:900])
    if not chunks:
        return ""
    merged = "\n\n".join(chunks)
    return merged[:1500]


def _predictor_context_relevance(effect: EffectResult, text_context_by_id: Dict[str, str]) -> int:
    predictor_tokens = [token for token in _normalize_match_key(effect.result_spec.outcome).split() if len(token) >= 4]
    if not predictor_tokens:
        return 0
    context = _effect_context_text(effect, text_context_by_id)
    merged = _normalize_match_key(f"{effect.quote} {context}")
    if not merged:
        return 0
    score = 0
    for token in predictor_tokens:
        if re.search(rf"\b{re.escape(token)}\b", merged):
            score += 3
    return score


def _cluster_anchor_value(cluster: Sequence[EffectResult]) -> Optional[float]:
    values = sorted(item.value for item in cluster if item.value is not None)
    if not values:
        return None
    middle = len(values) // 2
    if len(values) % 2 == 1:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def _merge_effect_cluster(cluster: Sequence[EffectResult]) -> EffectResult:
    ordered = sorted(cluster, key=_effect_rank_tuple)
    chosen = ordered[0]

    merged_ids = _dedupe_preserve([evidence_id for item in cluster for evidence_id in item.evidence_ids])
    merged_notes = _dedupe_preserve([note for item in cluster for note in item.notes])
    merged_table_rows = _dedupe_preserve([row for item in cluster for row in item.table_row_refs])
    merged_quote = _best_quote(cluster) or chosen.quote
    merged_value = _cluster_anchor_value(cluster)
    merged_group = _best_group_label(cluster) or chosen.grouping_label
    merged_predictor = _best_predictor_label(cluster) or chosen.result_spec.outcome
    merged_domain = _best_domain_label(cluster) or chosen.outcome_label_normalized
    merged_timepoint = _best_timepoint_label(cluster) or chosen.timepoint_label_normalized

    merged_comparison = chosen.result_spec.comparison
    if _is_unknown_token(merged_comparison) and not _is_unknown_token(merged_group):
        merged_comparison = merged_group

    updates: Dict[str, Any] = {
        "quote": merged_quote,
        "value": merged_value,
        "grouping_label": merged_group,
        "outcome_label_normalized": merged_domain,
        "timepoint_label_normalized": merged_timepoint,
        "evidence_ids": merged_ids,
        "table_row_refs": merged_table_rows,
        "dedup_sources": len(merged_ids) if merged_ids else max(1, chosen.dedup_sources),
        "result_spec": chosen.result_spec.model_copy(
            update={
                "comparison": merged_comparison,
                "groups": merged_group,
                "outcome": merged_predictor,
                "timepoint": merged_timepoint,
            }
        ),
    }
    if len(cluster) > 1:
        updates["notes"] = _dedupe_preserve(merged_notes + [f"consolidated_cluster={len(cluster)}"])
    else:
        updates["notes"] = merged_notes
    return chosen.model_copy(update=updates)


def _is_effect_compatible(left: EffectResult, right: EffectResult, value_tolerance: float) -> bool:
    if left.effect_scope != right.effect_scope:
        return False
    if left.effect_type != right.effect_type:
        return False
    if left.value is None or right.value is None:
        return False
    if abs(left.value - right.value) > value_tolerance:
        return False

    left_predictor = _predictor_cluster_key(left.result_spec.outcome)
    right_predictor = _predictor_cluster_key(right.result_spec.outcome)
    if left_predictor != right_predictor:
        return False

    if not _group_labels_compatible(left.grouping_label, right.grouping_label):
        return False
    if not _timepoints_compatible(left.timepoint_label_normalized, right.timepoint_label_normalized):
        return False
    if not _domains_compatible(left.outcome_label_normalized, right.outcome_label_normalized):
        return False

    return True


def _predictor_cluster_key(value: str) -> str:
    text = _normalize_match_key(value)
    text = re.sub(r"\b(full|reported|observed|unspecified|measure|scale|score|mat)\b", " ", text)
    text = re.sub(r"\b(relationship|marital)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or "unknown"


def _group_labels_compatible(left: str, right: str) -> bool:
    left_key = _normalize_match_key(left)
    right_key = _normalize_match_key(right)
    if left_key == right_key:
        return True
    if _is_unknown_token(left_key) or _is_unknown_token(right_key):
        return True
    if _is_generic_group_label(left_key) or _is_generic_group_label(right_key):
        left_tokens = set(_group_tokens(left_key))
        right_tokens = set(_group_tokens(right_key))
        return bool(left_tokens & right_tokens)
    return False


def _is_generic_group_label(group_key: str) -> bool:
    return " vs " in group_key or " and " in group_key


def _group_tokens(group_key: str) -> List[str]:
    text = group_key.replace(" vs ", " ").replace(" and ", " ")
    return [token for token in text.split() if token]


def _timepoints_compatible(left: str, right: str) -> bool:
    left_key = _normalize_match_key(left)
    right_key = _normalize_match_key(right)
    if left_key == right_key:
        return True
    if _is_unknown_token(left_key) or _is_unknown_token(right_key):
        return True
    return False


def _domains_compatible(left: str, right: str) -> bool:
    left_key = _normalize_match_key(left)
    right_key = _normalize_match_key(right)
    if left_key == right_key:
        return True
    if _is_unknown_token(left_key) or _is_unknown_token(right_key):
        return True
    return False


def _best_group_label(cluster: Sequence[EffectResult]) -> str:
    best = ""
    best_rank = (9, 9, 9)
    for item in cluster:
        label = normalize_inline_text(item.grouping_label)
        if not label:
            continue
        rank = (
            1 if _is_unknown_token(label) else 0,
            1 if _is_generic_group_label(_normalize_match_key(label)) else 0,
            len(_group_tokens(_normalize_match_key(label))),
        )
        if rank < best_rank:
            best = label
            best_rank = rank
    return best


def _best_predictor_label(cluster: Sequence[EffectResult]) -> str:
    best = ""
    best_score = -999
    for item in cluster:
        label = normalize_inline_text(item.result_spec.outcome)
        if not label:
            continue
        score = 0
        if not _is_unknown_token(label):
            score += 20
        if "unspecified" in label.lower():
            score -= 4
        score += min(len(label), 80) // 8
        if score > best_score:
            best_score = score
            best = label
    return best


def _best_domain_label(cluster: Sequence[EffectResult]) -> str:
    counts: Dict[str, int] = {}
    for item in cluster:
        domain = normalize_inline_text(item.outcome_label_normalized)
        if not domain or _is_unknown_token(domain):
            continue
        counts[domain] = counts.get(domain, 0) + 1
    if not counts:
        return "unknown"
    return max(counts, key=counts.get)


def _best_timepoint_label(cluster: Sequence[EffectResult]) -> str:
    priority = {"follow-up": 0, "post": 1, "baseline": 2, "unknown": 9}
    best = "unknown"
    best_rank = 99
    for item in cluster:
        label = _normalize_timepoint(item.timepoint_label_normalized or item.result_spec.timepoint)
        rank = priority.get(label, 5)
        if rank < best_rank:
            best = label
            best_rank = rank
    return best


def _effect_rank_tuple(effect: EffectResult) -> tuple[int, int, int, int]:
    return (
        _source_priority(effect),
        1 if _is_incomplete_quote(effect.quote, effect.source_kind) else 0,
        1 if _is_unknown_token(effect.grouping_label) or _is_unknown_token(effect.result_spec.outcome) else 0,
        -_quote_quality(effect.quote, effect.source_kind),
    )


def _source_priority(effect: EffectResult) -> int:
    notes = {normalize_inline_text(note).lower() for note in effect.notes}
    if effect.source_kind == "table":
        return 0
    if effect.source_kind == "text":
        if any(note.startswith("deterministic_") for note in notes):
            return 1
        return 2
    if effect.source_kind == "table_image_vision":
        return 3
    return 4


def _best_quote(effects: Sequence[EffectResult]) -> str:
    best = ""
    best_score = -1
    for effect in effects:
        score = _quote_quality(effect.quote, effect.source_kind)
        if score > best_score:
            best = effect.quote
            best_score = score
    return best


def _quote_quality(quote: str, source_kind: str = "text") -> int:
    text = normalize_inline_text(quote)
    if not text:
        return 0
    score = min(len(text), 220)
    if _is_incomplete_quote(text, source_kind):
        score -= 80
    return score


def _is_incomplete_quote(quote: str, source_kind: str) -> bool:
    text = normalize_inline_text(quote)
    if not text:
        return True
    if source_kind in {"table", "table_image_vision"}:
        return len(text) < 3
    if len(text) < 12:
        return True
    if source_kind == "text" and len(text) < 20:
        return True
    if re.search(r"\b(?:d|g|smd)\s*[=:]?\s*$", text, re.IGNORECASE):
        return True
    if text.endswith(("...", ",", ";", "(", "[", "=", "vs", "and")):
        return True
    return False


def _is_unknown_token(value: str) -> bool:
    lowered = normalize_inline_text(value).lower()
    return lowered in {"", "unknown", "n/a", "na", "none", "null", "unclear", "unspecified", "overall"}


def _normalize_match_key(value: str) -> str:
    return normalize_inline_text(value, normalize_decimal_comma=True).lower()


def _dedupe_preserve(values: Sequence[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _build_result_id(
    paper_id: str,
    seed: str,
    effect_type: str,
    value: float,
    outcome: str,
    comparison: str,
    timepoint: str,
    scope: str,
) -> str:
    raw = f"{paper_id}|{seed}|{scope}|{effect_type}|{value:.4f}|{outcome}|{comparison}|{timepoint}"
    return "res_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


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


def _openai_json_call_vision(
    requests_module,
    api_key: str,
    model: str,
    api_base: str,
    timeout_seconds: int,
    system_prompt: str,
    user_prompt: str,
    image_base64: str,
) -> Optional[Dict[str, Any]]:
    endpoint = f"{api_base.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                    },
                ],
            },
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


def _compact_error(exc: Exception) -> str:
    text = normalize_inline_text(str(exc))
    return text[:220] + ("..." if len(text) > 220 else "")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
