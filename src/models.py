from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DecisionLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    level: Literal["info", "warning", "error"] = "info"
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PaperMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    source_path: str
    pdf_hash: str
    title: str = "unknown"
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    num_pages: int = 0
    parser: str = "unknown"
    ingested_at: datetime = Field(default_factory=datetime.utcnow)


class TextPassage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    page: int
    section_guess: str = "unknown"
    text: str
    source: Literal["pdf_text", "grobid_aligned"] = "pdf_text"


class TableRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    row_index: int
    cells: List[str]


class ExtractedTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    table_id: str
    page: int
    title_guess: str = "unknown"
    rows: List[TableRow] = Field(default_factory=list)
    structured: bool = False
    image_path: Optional[str] = None
    extraction_method: Literal["pdfplumber", "camelot", "none"] = "none"
    status: Literal["ok", "empty", "failed", "image_only"] = "empty"


class IngestArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: PaperMetadata
    text_index: List[TextPassage] = Field(default_factory=list)
    tables: List[ExtractedTable] = Field(default_factory=list)
    logs: List[DecisionLog] = Field(default_factory=list)


class IngestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_chars: int = 1800
    chunk_overlap: int = 220
    min_chunk_chars: int = 120
    use_grobid: bool = False
    grobid_url: str = "http://localhost:8070"
    request_timeout_seconds: int = 60
    metadata_pages_scan: int = 4
    use_openai_metadata: bool = True
    use_openai_extraction: bool = True
    openai_extraction_max_snippets: int = 40
    openai_effect_snippet_chars: int = 1200
    openai_model: str = "gpt-4.1-mini"
    openai_api_base: str = "https://api.openai.com/v1"
    openai_timeout_seconds: int = 45


class BlockFlagItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    contains_results: bool = False
    contains_effect_size: bool = False
    contains_methods: bool = False
    contains_population: bool = False
    contains_bias_info: bool = False
    confidence: float = 0.0
    notes: List[str] = Field(default_factory=list)


class BlockRoutingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[BlockFlagItem] = Field(default_factory=list)
    relevant_effect_blocks: List[str] = Field(default_factory=list)
    relevant_quality_blocks: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class EffectResultSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: str = "unknown"
    timepoint: str = "unknown"
    comparison: str = "unknown"
    groups: str = "unknown"
    analysis_set: str = "unknown"


class EffectResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_id: str
    result_spec: EffectResultSpec
    design_level: Literal["meta_analysis", "single_study", "unknown"] = "unknown"
    effect_role: Literal[
        "pooled_overall",
        "subgroup",
        "followup",
        "sensitivity",
        "individual_study_effect",
        "unclear",
    ] = "unclear"
    grouping_label: str = "overall"
    outcome_label_normalized: str = "unknown"
    timepoint_label_normalized: str = "unknown"
    canonical_key: str = ""
    stat_consistency: Literal["pass", "failed", "unknown"] = "unknown"
    dedup_sources: int = 1
    effect_type: str = "unknown"
    effect_scope: Literal["study_effect", "literature_cited", "model_stat"] = "study_effect"
    origin: Literal["reported", "computed", "external_transfer"] = "reported"
    source_kind: Literal["text", "table", "table_image_vision"] = "text"
    source_page: int = 0
    source_ref: str = ""
    quote: str = ""
    value: Optional[float] = None
    se: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    derivation_method: Literal[
        "reported",
        "computed_from_means",
        "converted_from_t",
        "assumption_based",
        "external_transfer",
        "not_derivable",
    ] = "not_derivable"
    assumptions: List[str] = Field(default_factory=list)
    calc_confidence: Literal["exact", "assumption_based", "not_derivable"] = "not_derivable"
    is_primary: bool = True
    exclusion_reason: str = ""
    evidence_ids: List[str] = Field(default_factory=list)
    table_row_refs: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class EffectsComputationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effects: List[EffectResult] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class QuickQualityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    randomization: Literal["yes", "no", "unclear"] = "unclear"
    control_group: Literal["yes", "no", "unclear"] = "unclear"
    sample_size_reported: Literal["yes", "no", "unclear"] = "unclear"
    attrition_reported: Literal["yes", "no", "unclear"] = "unclear"
    blinding_reported: Literal["yes", "no", "unclear"] = "unclear"
    internal_quality_score: float = 0.0
    justification: str = "unknown"
    evidence_ids: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ExternalCredibilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title_match_found: bool = False
    venue: str = "unknown"
    publisher: str = "unknown"
    citation_count: Optional[int] = None
    authors_found: int = 0
    external_score: float = 0.0
    credibility_level: Literal["High", "Moderate", "Low", "Unknown"] = "Unknown"
    notes: List[str] = Field(default_factory=list)


class ReliabilityItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result_id: str
    calc_score: float = 0.0
    bias_score: float = 0.0
    review_quality_score: float = 0.0
    reporting_score: float = 0.0
    external_score: float = 0.0
    consistency_score: float = 0.0
    reliability_score_total: float = 0.0
    verdict: Literal["High", "Moderate", "Low", "Not usable"] = "Not usable"
    justification: str = "unknown"
    evidence_ids: List[str] = Field(default_factory=list)


class ReliabilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: List[ReliabilityItem] = Field(default_factory=list)
    global_score: float = 0.0
    conclusion: str = "non utilisable"
    notes: List[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class PipelineStepDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    key: str
    name: str
    description: str


class PipelineStepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str
    key: str
    name: str
    status: Literal["completed", "skipped", "failed"]
    message: str
    started_at: datetime
    finished_at: datetime
    duration_seconds: float
    output_files: List[str] = Field(default_factory=list)


class PipelineRunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str
    output_dir: str
    steps: List[PipelineStepResult] = Field(default_factory=list)
