from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

try:
    from ..models import (
        EffectResult,
        EffectsComputationResult,
        ExternalCredibilityResult,
        QuickQualityResult,
        ReliabilityItem,
        ReliabilityResult,
    )
except ImportError:  # pragma: no cover
    from src.models import (  # type: ignore
        EffectResult,
        EffectsComputationResult,
        ExternalCredibilityResult,
        QuickQualityResult,
        ReliabilityItem,
        ReliabilityResult,
    )


def run(
    effects_result: EffectsComputationResult,
    quick_quality_result: QuickQualityResult,
    external_credibility_result: ExternalCredibilityResult,
    output_dir: str | Path,
    article_evaluation_result: dict | None = None,
) -> ReliabilityResult:
    items: List[ReliabilityItem] = []
    study_effects = [effect for effect in effects_result.effects if effect.effect_scope == "study_effect"]
    literature_effects_count = len([effect for effect in effects_result.effects if effect.effect_scope == "literature_cited"])
    model_stats_count = len([effect for effect in effects_result.effects if effect.effect_scope == "model_stat"])

    # Extract article evaluation global score (0-1) if available
    article_eval_score = _extract_article_eval_score(article_evaluation_result)

    for effect in study_effects:
        calc_score = _calc_score(effect)
        internal_score = _clamp(quick_quality_result.internal_quality_score)
        external_score = _clamp(external_credibility_result.external_score)
        if article_eval_score is not None:
            # New formula: 40% calc + 20% internal + 15% external + 25% article evaluation
            total = _clamp(
                (0.40 * calc_score)
                + (0.20 * internal_score)
                + (0.15 * external_score)
                + (0.25 * article_eval_score)
            )
            justification = (
                f"calc={calc_score:.2f}, qualite_interne={internal_score:.2f}, "
                f"credibilite_externe={external_score:.2f}, eval_article={article_eval_score:.2f}"
            )
        else:
            # Fallback: original formula without article evaluation
            total = _clamp((0.5 * calc_score) + (0.3 * internal_score) + (0.2 * external_score))
            justification = (
                f"calc={calc_score:.2f}, qualite_interne={internal_score:.2f}, "
                f"credibilite_externe={external_score:.2f}"
            )
        verdict = _verdict(total, effect)
        items.append(
            ReliabilityItem(
                result_id=effect.result_id,
                calc_score=round(calc_score, 3),
                bias_score=round(internal_score, 3),
                review_quality_score=round(article_eval_score, 3) if article_eval_score is not None else 0.0,
                reporting_score=round(internal_score, 3),
                external_score=round(external_score, 3),
                consistency_score=round(_consistency_score(effect), 3),
                reliability_score_total=round(total, 3),
                verdict=verdict,
                justification=justification,
                evidence_ids=list(effect.evidence_ids[:30]),
            )
        )

    global_score = _global_score(items)
    conclusion = _global_conclusion(global_score=global_score, effects=study_effects)

    notes = [
        f"effects_count={len(effects_result.effects)}",
        f"study_effects_count={len(study_effects)}",
        f"literature_effects_count={literature_effects_count}",
        f"model_stats_count={model_stats_count}",
        f"internal_quality_score={quick_quality_result.internal_quality_score:.3f}",
        f"external_score={external_credibility_result.external_score:.3f}",
    ]
    if article_eval_score is not None:
        notes.append(f"article_evaluation_score={article_eval_score:.3f}")

    result = ReliabilityResult(
        items=items,
        global_score=round(global_score, 3),
        conclusion=conclusion,
        notes=notes,
    )
    _write_json(Path(output_dir) / "07_summary_score.json", result.model_dump(mode="json"))
    _write_json(Path(output_dir) / "12_reliability.json", result.model_dump(mode="json"))
    return result


def _extract_article_eval_score(article_evaluation_result: dict | None) -> float | None:
    """Extract the global score (0-1) from the article evaluation dict."""
    if not article_evaluation_result or not isinstance(article_evaluation_result, dict):
        return None
    scores = article_evaluation_result.get("scores", {})
    if not isinstance(scores, dict):
        return None
    g = scores.get("global", {})
    if isinstance(g, dict):
        val = g.get("value")
        if isinstance(val, (int, float)):
            return _clamp(float(val))
    if isinstance(g, (int, float)):
        return _clamp(float(g))
    return None


def _calc_score(effect: EffectResult) -> float:
    """Score the statistical quality of an individual effect (0–1).

    Bonuses for:
      - recognised effect type (d/g/SMD)
      - confidence interval reported
      - CI is reasonably narrow (width ≤ 1.0)
      - p-value reported
      - p-value < 0.05
      - sample size reported
      - sample size ≥ 30 (reasonable minimum)
      - consistency pass

    Penalties for:
      - not derivable confidence
      - very small sample (< 15)
    """
    if effect.value is None:
        return 0.0
    score = 0.35
    # Effect type bonus
    if effect.effect_type in {"d", "g", "SMD"}:
        score += 0.15
    # Confidence interval bonus
    if effect.ci_low is not None and effect.ci_high is not None:
        score += 0.10
        ci_width = abs(effect.ci_high - effect.ci_low)
        if ci_width <= 1.0:
            score += 0.05  # narrow CI bonus
    # P-value bonus
    if effect.p_value is not None:
        score += 0.05
        if effect.p_value < 0.05:
            score += 0.05
    # Sample size bonus/penalty
    if effect.sample_size is not None:
        score += 0.05
        if effect.sample_size >= 30:
            score += 0.05
        elif effect.sample_size < 15:
            score -= 0.10  # very small sample penalty
    # Consistency
    if effect.stat_consistency == "pass":
        score += 0.10
    # Derivability penalty
    if effect.calc_confidence == "not_derivable":
        score -= 0.20
    return _clamp(score)


def _consistency_score(effect: EffectResult) -> float:
    if effect.stat_consistency == "pass":
        return 1.0
    if effect.stat_consistency == "failed":
        return 0.0
    return 0.5


def _verdict(total: float, effect: EffectResult) -> str:
    if effect.value is None:
        return "Not usable"
    if total >= 0.75:
        return "High"
    if total >= 0.55:
        return "Moderate"
    if total >= 0.4:
        return "Low"
    return "Not usable"


def _global_score(items: Sequence[ReliabilityItem]) -> float:
    usable = [item.reliability_score_total for item in items if item.verdict != "Not usable"]
    if not usable:
        return 0.0
    return sum(usable) / len(usable)


def _global_conclusion(global_score: float, effects: Sequence[EffectResult]) -> str:
    has_effect = any(effect.value is not None for effect in effects)
    if not has_effect:
        return "non utilisable"
    if global_score >= 0.65:
        return "utilisable"
    if global_score >= 0.45:
        return "utilisable avec prudence"
    return "non utilisable"


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
