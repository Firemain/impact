from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

from .effect_labels import derive_group_domain_predictor
from .models import (
    BlockRoutingResult,
    EffectsComputationResult,
    ExternalCredibilityResult,
    IngestArtifacts,
    IngestConfig,
    PipelineRunSummary,
    PipelineStepDefinition,
    PipelineStepResult,
    QuickQualityResult,
    ReliabilityResult,
)
from .steps.block_router import run as run_block_router
from .steps.effects_extract import run as run_effects_extract
from .steps.external_credibility import run as run_external_credibility
from .steps.ingest import run as run_ingest
from .steps.quality_quick import run as run_quality_quick
from .steps.reliability import run as run_reliability
from .storage.sqlite import persist_pipeline_outputs

PIPELINE_STEPS = [
    PipelineStepDefinition(
        step_id="01",
        key="ingest",
        name="Ingestion PDF",
        description="Extraction texte + tables et creation des blocs.",
    ),
    PipelineStepDefinition(
        step_id="02",
        key="block_router",
        name="Classification blocs",
        description="LLM yes/no par bloc: results/effects/methods/population/bias.",
    ),
    PipelineStepDefinition(
        step_id="03",
        key="effects_extract",
        name="Extraction effets",
        description="Extraction d/g/SMD sur blocs pertinents uniquement.",
    ),
    PipelineStepDefinition(
        step_id="04",
        key="quality_quick",
        name="Qualite interne rapide",
        description="Mini grille methodologique et score simple.",
    ),
    PipelineStepDefinition(
        step_id="05",
        key="external_credibility",
        name="Credibilite externe",
        description="OpenAlex: citations, venue, auteurs, score simple.",
    ),
    PipelineStepDefinition(
        step_id="06",
        key="final_score",
        name="Score global",
        description="Combine effets + qualite interne + credibilite externe.",
    ),
]

ProgressCallback = Callable[[str, str, str, Dict[str, object]], None]


def run_pipeline(
    pdf_path: str | Path,
    output_root: str | Path = "outputs",
    paper_id: Optional[str] = None,
    ingest_config: Optional[IngestConfig] = None,
) -> IngestArtifacts:
    return run_ingest(
        pdf_path=pdf_path,
        output_root=output_root,
        paper_id=paper_id,
        config=ingest_config,
    )


def run_full_pipeline(
    pdf_path: str | Path,
    output_root: str | Path = "outputs",
    paper_id: Optional[str] = None,
    ingest_config: Optional[IngestConfig] = None,
    progress_callback: Optional[ProgressCallback] = None,
    visualization_delay_seconds: float = 0.0,
) -> PipelineRunSummary:
    _ = visualization_delay_seconds  # kept for API compatibility
    config = ingest_config or IngestConfig()

    results: list[PipelineStepResult] = []
    ingest_artifacts: Optional[IngestArtifacts] = None
    block_routing: Optional[BlockRoutingResult] = None
    effects_result: Optional[EffectsComputationResult] = None
    quick_quality: Optional[QuickQualityResult] = None
    external_credibility: Optional[ExternalCredibilityResult] = None
    reliability_result: Optional[ReliabilityResult] = None
    output_path: Optional[Path] = None
    resolved_paper_id = paper_id or "unknown"

    for step in PIPELINE_STEPS:
        started_at = datetime.utcnow()
        _emit_progress(progress_callback, step, "running", f"{step.name} demarree")
        try:
            if step.key == "ingest":
                ingest_artifacts = run_ingest(
                    pdf_path=pdf_path,
                    output_root=output_root,
                    paper_id=paper_id,
                    config=config,
                )
                resolved_paper_id = ingest_artifacts.metadata.paper_id
                output_path = Path(output_root).resolve() / resolved_paper_id
                message = (
                    f"{ingest_artifacts.metadata.num_pages} pages, "
                    f"{len(ingest_artifacts.text_index)} blocs."
                )
                results.append(_completed_step(step, started_at, message, ["00_metadata.json", "01_text_index.json", "02_tables.json"]))
                _emit_progress(progress_callback, step, "completed", message)
                continue

            if output_path is None or ingest_artifacts is None:
                raise RuntimeError("Pipeline state invalide: ingestion manquante.")

            if step.key == "block_router":
                block_routing = run_block_router(
                    ingest_artifacts=ingest_artifacts,
                    output_dir=output_path,
                    use_openai_extraction=config.use_openai_extraction,
                    openai_model=config.openai_model,
                    openai_api_base=config.openai_api_base,
                    openai_timeout_seconds=config.openai_timeout_seconds,
                    openai_extraction_max_snippets=config.openai_extraction_max_snippets,
                    openai_snippet_chars=config.openai_effect_snippet_chars,
                )
                message = (
                    f"blocs effets={len(block_routing.relevant_effect_blocks)}, "
                    f"blocs qualite={len(block_routing.relevant_quality_blocks)}"
                )
                results.append(_completed_step(step, started_at, message, ["03_block_flags.json"]))
                _emit_progress(progress_callback, step, "completed", message)
                continue

            if step.key == "effects_extract":
                if block_routing is None:
                    raise RuntimeError("Block routing manquant.")
                effects_result = run_effects_extract(
                    ingest_artifacts=ingest_artifacts,
                    block_routing=block_routing,
                    output_dir=output_path,
                    paper_id=resolved_paper_id,
                    use_openai_extraction=config.use_openai_extraction,
                    openai_model=config.openai_model,
                    openai_api_base=config.openai_api_base,
                    openai_timeout_seconds=config.openai_timeout_seconds,
                    openai_extraction_max_snippets=config.openai_extraction_max_snippets,
                    openai_effect_snippet_chars=config.openai_effect_snippet_chars,
                )
                study_count = len([effect for effect in effects_result.effects if effect.effect_scope == "study_effect"])
                cited_count = len([effect for effect in effects_result.effects if effect.effect_scope == "literature_cited"])
                model_stats_count = len([effect for effect in effects_result.effects if effect.effect_scope == "model_stat"])
                message = (
                    f"effets etude={study_count}, cites={cited_count}, "
                    f"stats_modele={model_stats_count}"
                )
                results.append(_completed_step(step, started_at, message, ["04_effects.json"]))
                _emit_progress(progress_callback, step, "completed", message)
                continue

            if step.key == "quality_quick":
                if block_routing is None:
                    raise RuntimeError("Block routing manquant.")
                quick_quality = run_quality_quick(
                    ingest_artifacts=ingest_artifacts,
                    block_routing=block_routing,
                    output_dir=output_path,
                    use_openai_extraction=config.use_openai_extraction,
                    openai_model=config.openai_model,
                    openai_api_base=config.openai_api_base,
                    openai_timeout_seconds=config.openai_timeout_seconds,
                    openai_extraction_max_snippets=config.openai_extraction_max_snippets,
                    openai_snippet_chars=config.openai_effect_snippet_chars,
                )
                message = f"score qualite interne={quick_quality.internal_quality_score:.2f}"
                results.append(_completed_step(step, started_at, message, ["05_quality_quick.json"]))
                _emit_progress(progress_callback, step, "completed", message)
                continue

            if step.key == "external_credibility":
                external_credibility = run_external_credibility(
                    ingest_artifacts=ingest_artifacts,
                    output_dir=output_path,
                    openai_model=config.openai_model,
                    openai_api_base=config.openai_api_base,
                    openai_timeout_seconds=config.openai_timeout_seconds,
                    use_openai_extraction=False,
                )
                message = (
                    f"credibilite={external_credibility.credibility_level}, "
                    f"score={external_credibility.external_score:.2f}"
                )
                results.append(_completed_step(step, started_at, message, ["06_external_credibility.json"]))
                _emit_progress(progress_callback, step, "completed", message)
                continue

            if step.key == "final_score":
                if effects_result is None or quick_quality is None or external_credibility is None:
                    raise RuntimeError("Etapes precedentes manquantes pour le score final.")
                reliability_result = run_reliability(
                    effects_result=effects_result,
                    quick_quality_result=quick_quality,
                    external_credibility_result=external_credibility,
                    output_dir=output_path,
                )
                message = (
                    f"score global={reliability_result.global_score:.2f}, "
                    f"conclusion={reliability_result.conclusion}"
                )
                results.append(_completed_step(step, started_at, message, ["07_summary_score.json", "12_reliability.json"]))
                _emit_progress(progress_callback, step, "completed", message)
                continue

            results.append(_failed_step(step, started_at, "Etape non reconnue"))
            _emit_progress(progress_callback, step, "failed", "Etape non reconnue")
        except Exception as exc:
            results.append(_failed_step(step, started_at, str(exc)))
            _emit_progress(progress_callback, step, "failed", str(exc))
            raise

    summary = PipelineRunSummary(
        paper_id=resolved_paper_id,
        output_dir=str((Path(output_root).resolve() / resolved_paper_id) if output_path is None else output_path),
        steps=results,
    )

    if ingest_artifacts is not None:
        try:
            persist_pipeline_outputs(
                ingest_artifacts=ingest_artifacts,
                effects_result=effects_result,
                reliability_result=reliability_result,
            )
        except Exception:
            pass

    if output_path is not None:
        _write_report(
            output_dir=output_path,
            summary=summary,
            effects_result=effects_result,
            quick_quality=quick_quality,
            external_credibility=external_credibility,
            reliability_result=reliability_result,
        )
    return summary


def _completed_step(
    step: PipelineStepDefinition,
    started_at: datetime,
    message: str,
    output_files: list[str],
) -> PipelineStepResult:
    finished_at = datetime.utcnow()
    return PipelineStepResult(
        step_id=step.step_id,
        key=step.key,
        name=step.name,
        status="completed",
        message=message,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=(finished_at - started_at).total_seconds(),
        output_files=output_files,
    )


def _failed_step(step: PipelineStepDefinition, started_at: datetime, message: str) -> PipelineStepResult:
    finished_at = datetime.utcnow()
    return PipelineStepResult(
        step_id=step.step_id,
        key=step.key,
        name=step.name,
        status="failed",
        message=message,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=(finished_at - started_at).total_seconds(),
        output_files=[],
    )


def _emit_progress(
    callback: Optional[ProgressCallback],
    step: PipelineStepDefinition,
    state: str,
    message: str,
) -> None:
    if callback is None:
        return
    callback(step.step_id, state, message, {})


def _write_report(
    output_dir: Path,
    summary: PipelineRunSummary,
    effects_result: Optional[EffectsComputationResult],
    quick_quality: Optional[QuickQualityResult],
    external_credibility: Optional[ExternalCredibilityResult],
    reliability_result: Optional[ReliabilityResult],
) -> None:
    metadata = _load_metadata_payload(output_dir / "00_metadata.json")

    lines: list[str] = []
    lines.append("# Rapport court")
    lines.append("")
    lines.append(f"- paper_id: `{summary.paper_id}`")
    lines.append(f"- generated_at_utc: `{datetime.utcnow().isoformat()}`")
    if metadata is not None:
        title = _safe_text(metadata.get("title", "unknown"))
        year = _safe_text(metadata.get("year", "unknown"))
        doi = _safe_text(metadata.get("doi", "unknown"))
        authors_raw = metadata.get("authors", [])
        authors = ", ".join(str(item) for item in authors_raw) if isinstance(authors_raw, list) and authors_raw else "unknown"
        lines.append(f"- titre: {title}")
        lines.append(f"- auteurs: {authors}")
        lines.append(f"- annee: {year}")
        lines.append(f"- doi: {doi}")
    lines.append("")
    lines.append("## Effets de l'etude")
    if effects_result is None or not effects_result.effects:
        lines.append("- Aucun effet extrait.")
    else:
        sorted_effects = sorted(
            effects_result.effects,
            key=lambda effect: abs(effect.value) if effect.value is not None else -1.0,
            reverse=True,
        )
        study_effects = [effect for effect in sorted_effects if effect.effect_scope == "study_effect"]
        cited_effects = [effect for effect in sorted_effects if effect.effect_scope == "literature_cited"]
        model_stats = [effect for effect in sorted_effects if effect.effect_scope == "model_stat"]

        _append_effect_table(lines, study_effects, empty_message="- Aucun effet de l'etude extrait.")
        lines.append("")
        lines.append("## Effets cites dans la litterature")
        _append_effect_table(lines, cited_effects, empty_message="- Aucun effet cite extrait.")
        lines.append("")
        lines.append("## Stats de modele (exclues des effect sizes)")
        _append_effect_table(lines, model_stats, empty_message="- Aucune stat de modele detectee.")
    lines.append("")
    lines.append("## Qualite interne")
    if quick_quality is not None:
        lines.append(f"- score={quick_quality.internal_quality_score:.2f}")
        lines.append(f"- randomization={quick_quality.randomization}")
        lines.append(f"- control_group={quick_quality.control_group}")
        lines.append(f"- sample_size_reported={quick_quality.sample_size_reported}")
        lines.append(f"- attrition_reported={quick_quality.attrition_reported}")
        lines.append(f"- blinding_reported={quick_quality.blinding_reported}")
        lines.append(f"- justification: {quick_quality.justification}")
    else:
        lines.append("- non disponible")
    lines.append("")
    lines.append("## Credibilite externe")
    if external_credibility is not None:
        lines.append(f"- score={external_credibility.external_score:.2f}")
        lines.append(f"- niveau={external_credibility.credibility_level}")
        lines.append(f"- venue={external_credibility.venue}")
        lines.append(f"- citations={external_credibility.citation_count}")
    else:
        lines.append("- non disponible")
    lines.append("")
    lines.append("## Score global")
    if reliability_result is not None:
        lines.append(f"- global_score={reliability_result.global_score:.2f}")
        lines.append(f"- conclusion={reliability_result.conclusion}")
    else:
        lines.append("- non disponible")
    lines.append("")
    lines.append("## Etapes")
    for step in summary.steps:
        lines.append(f"- `{step.step_id}` {step.name}: **{step.status}** - {step.message}")
    lines.append("")

    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _load_metadata_payload(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _safe_text(value: object) -> str:
    text = str(value).replace("\n", " ").strip()
    text = text.replace("|", "/")
    return text if text else "unknown"


def _report_effect_size(effect) -> str:
    if effect.value is None:
        return f"{effect.effect_type}=n/a"
    if effect.ci_low is not None and effect.ci_high is not None:
        return f"{effect.effect_type}={effect.value:.3f} (CI [{effect.ci_low:.3f}; {effect.ci_high:.3f}])"
    return f"{effect.effect_type}={effect.value:.3f}"


def _append_effect_table(lines: list[str], effects: list, empty_message: str) -> None:
    if not effects:
        lines.append(empty_message)
        return
    lines.append("| Groups | Domain | Predictor | Effect size | Effect duration | Quote | Page | Source |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for effect in effects[:40]:
        group, domain, predictor = derive_group_domain_predictor(
            group_raw=str(effect.grouping_label or effect.result_spec.groups or ""),
            predictor_raw=str(effect.result_spec.outcome or ""),
            domain_raw=str(effect.outcome_label_normalized or ""),
            context=str(effect.quote or ""),
        )
        group = _safe_text(group)
        domain = _safe_text(domain)
        predictor = _safe_text(predictor)
        duration = _safe_text(effect.timepoint_label_normalized or effect.result_spec.timepoint or "unknown")
        effect_size = _report_effect_size(effect)
        quote = _safe_text(effect.quote[:120] if effect.quote else "")
        page = _safe_text(effect.source_page if effect.source_page else "unknown")
        source = _safe_text(effect.source_kind)
        lines.append(f"| {group} | {domain} | {predictor} | {effect_size} | {duration} | {quote} | {page} | {source} |")
