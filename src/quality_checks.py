from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from .models import (
    BlockRoutingResult,
    EffectsComputationResult,
    ExternalCredibilityResult,
    PaperMetadata,
    QuickQualityResult,
    ReliabilityResult,
    TextPassage,
)


@dataclass
class CheckResult:
    ok: bool
    message: str


def run_checks(output_dir: str | Path) -> List[CheckResult]:
    directory = Path(output_dir).resolve()
    results: List[CheckResult] = []
    if not directory.exists():
        return [CheckResult(ok=False, message=f"Dossier introuvable: {directory}")]

    required_files = [
        "00_metadata.json",
        "01_text_index.json",
        "02_tables.json",
        "03_block_flags.json",
        "04_effects.json",
        "05_quality_quick.json",
        "06_external_credibility.json",
        "07_summary_score.json",
        "12_reliability.json",
        "report.md",
    ]
    for filename in required_files:
        path = directory / filename
        results.append(CheckResult(ok=path.exists(), message=f"{filename} {'present' if path.exists() else 'manquant'}"))

    metadata = _load_model(directory / "00_metadata.json", PaperMetadata, results)
    passages = _load_models(directory / "01_text_index.json", TextPassage, results)
    block_flags = _load_model(directory / "03_block_flags.json", BlockRoutingResult, results)
    effects = _load_model(directory / "04_effects.json", EffectsComputationResult, results)
    quality = _load_model(directory / "05_quality_quick.json", QuickQualityResult, results)
    external = _load_model(directory / "06_external_credibility.json", ExternalCredibilityResult, results)
    summary = _load_model(directory / "07_summary_score.json", ReliabilityResult, results)
    reliability = _load_model(directory / "12_reliability.json", ReliabilityResult, results)

    evidence_pool = {item.evidence_id for item in passages}
    if evidence_pool:
        results.append(CheckResult(ok=True, message=f"evidence pool charge ({len(evidence_pool)} ids)"))
    else:
        results.append(CheckResult(ok=False, message="01_text_index.json vide"))

    if metadata is not None:
        results.append(CheckResult(ok=metadata.title != "unknown", message="metadata.title renseigne" if metadata.title != "unknown" else "metadata.title = unknown"))

    if block_flags is not None:
        unknown_ids = [item.evidence_id for item in block_flags.items if item.evidence_id not in evidence_pool]
        results.append(CheckResult(ok=not unknown_ids, message="block_flags evidence_ids coherents" if not unknown_ids else f"block_flags ids inconnus: {unknown_ids[:5]}"))

    if effects is not None:
        has_effect_value = any(item.value is not None for item in effects.effects)
        results.append(
            CheckResult(
                ok=True,
                message=(
                    "au moins un effet avec valeur"
                    if has_effect_value
                    else "aucun effet numerique extrait (cas acceptable)"
                ),
            )
        )
        unknown_effect_ids = [evidence_id for item in effects.effects for evidence_id in item.evidence_ids if evidence_id not in evidence_pool]
        results.append(CheckResult(ok=not unknown_effect_ids, message="effects evidence_ids coherents" if not unknown_effect_ids else f"effects ids inconnus: {unknown_effect_ids[:5]}"))

    if quality is not None:
        valid_score = 0.0 <= quality.internal_quality_score <= 1.0
        results.append(CheckResult(ok=valid_score, message="quality score dans [0,1]" if valid_score else "quality score hors bornes"))

    if external is not None:
        valid_ext = 0.0 <= external.external_score <= 1.0
        results.append(CheckResult(ok=valid_ext, message="external score dans [0,1]" if valid_ext else "external score hors bornes"))

    if summary is not None and reliability is not None:
        aligned = summary.global_score == reliability.global_score and summary.conclusion == reliability.conclusion
        results.append(CheckResult(ok=aligned, message="07_summary_score et 12_reliability alignes" if aligned else "07_summary_score et 12_reliability differents"))

    return results


def summarize(results: Sequence[CheckResult]) -> str:
    ok_count = sum(1 for result in results if result.ok)
    fail_count = len(results) - ok_count
    lines = [f"Checks OK={ok_count} FAIL={fail_count}"]
    for result in results:
        lines.append(f"{'[OK]' if result.ok else '[FAIL]'} {result.message}")
    return "\n".join(lines)


def _load_model(path: Path, model_type, results: List[CheckResult]):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        instance = model_type.model_validate(payload)
        results.append(CheckResult(ok=True, message=f"{path.name} schema valide"))
        return instance
    except Exception as exc:
        results.append(CheckResult(ok=False, message=f"{path.name} schema invalide: {exc}"))
        return None


def _load_models(path: Path, model_type, results: List[CheckResult]):
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise TypeError("payload n est pas une liste")
        instances = [model_type.model_validate(entry) for entry in payload]
        results.append(CheckResult(ok=True, message=f"{path.name} schema liste valide ({len(instances)} elements)"))
        return instances
    except Exception as exc:
        results.append(CheckResult(ok=False, message=f"{path.name} schema invalide: {exc}"))
        return []
