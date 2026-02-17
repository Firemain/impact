from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Sequence

try:
    from ..models import (
        EffectsComputationResult,
        IngestArtifacts,
        ReliabilityResult,
    )
except ImportError:  # pragma: no cover
    from src.models import (  # type: ignore
        EffectsComputationResult,
        IngestArtifacts,
        ReliabilityResult,
    )


DEFAULT_DB_PATH = Path("outputs") / "impact_index.sqlite"


def persist_pipeline_outputs(
    ingest_artifacts: IngestArtifacts,
    effects_result: Optional[EffectsComputationResult],
    reliability_result: Optional[ReliabilityResult],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Path:
    path = Path(db_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        _create_schema(connection)
        _upsert_paper(connection, ingest_artifacts)
        _upsert_evidence(connection, ingest_artifacts)
        if effects_result is not None and reliability_result is not None:
            _upsert_results(
                connection=connection,
                paper_id=ingest_artifacts.metadata.paper_id,
                effects_result=effects_result,
                reliability_result=reliability_result,
            )
        connection.commit()
    return path


def _create_schema(connection: sqlite3.Connection) -> None:
    cursor = connection.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS papers (
            paper_id TEXT PRIMARY KEY,
            title TEXT,
            year INTEGER,
            doi TEXT,
            design TEXT,
            pdf_hash TEXT,
            created_at TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            result_id TEXT PRIMARY KEY,
            paper_id TEXT,
            outcome TEXT,
            timepoint TEXT,
            comparison TEXT,
            effect_type TEXT,
            effect_value REAL,
            ci_low REAL,
            ci_high REAL,
            calc_confidence TEXT,
            bias_overall TEXT,
            reliability_score REAL,
            created_at TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence (
            evidence_id TEXT PRIMARY KEY,
            paper_id TEXT,
            type TEXT,
            page INTEGER,
            section TEXT,
            text TEXT,
            table_id TEXT,
            row_ref TEXT
        )
        """
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_results_paper_id ON results(paper_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_evidence_paper_id ON evidence(paper_id)")


def _upsert_paper(
    connection: sqlite3.Connection,
    ingest_artifacts: IngestArtifacts,
) -> None:
    metadata = ingest_artifacts.metadata
    design_value = "unknown"
    created_at = datetime.utcnow().isoformat()
    connection.execute(
        """
        INSERT OR REPLACE INTO papers(
            paper_id, title, year, doi, design, pdf_hash, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metadata.paper_id,
            metadata.title,
            metadata.year,
            metadata.doi,
            design_value,
            metadata.pdf_hash,
            created_at,
        ),
    )


def _upsert_results(
    connection: sqlite3.Connection,
    paper_id: str,
    effects_result: EffectsComputationResult,
    reliability_result: ReliabilityResult,
) -> None:
    bias_overall = "not_scored"
    reliability_map = {
        item.result_id: item.reliability_score_total for item in reliability_result.items
    }
    created_at = datetime.utcnow().isoformat()
    for effect in effects_result.effects:
        reliability_score = reliability_map.get(effect.result_id)
        connection.execute(
            """
            INSERT OR REPLACE INTO results(
                result_id, paper_id, outcome, timepoint, comparison,
                effect_type, effect_value, ci_low, ci_high, calc_confidence,
                bias_overall, reliability_score, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                effect.result_id,
                paper_id,
                effect.result_spec.outcome,
                effect.result_spec.timepoint,
                effect.result_spec.comparison,
                effect.effect_type,
                effect.value,
                effect.ci_low,
                effect.ci_high,
                effect.calc_confidence,
                bias_overall,
                reliability_score,
                created_at,
            ),
        )


def _upsert_evidence(connection: sqlite3.Connection, ingest_artifacts: IngestArtifacts) -> None:
    paper_id = ingest_artifacts.metadata.paper_id
    for passage in ingest_artifacts.text_index:
        connection.execute(
            """
            INSERT OR REPLACE INTO evidence(
                evidence_id, paper_id, type, page, section, text, table_id, row_ref
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                passage.evidence_id,
                paper_id,
                "text_passage",
                passage.page,
                passage.section_guess,
                passage.text,
                None,
                None,
            ),
        )

    for table in ingest_artifacts.tables:
        table_row_ids = _table_row_ids(paper_id=paper_id, table_id=table.table_id, row_indices=[row.row_index for row in table.rows])
        for row, generated_id in zip(table.rows, table_row_ids):
            connection.execute(
                """
                INSERT OR REPLACE INTO evidence(
                    evidence_id, paper_id, type, page, section, text, table_id, row_ref
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generated_id,
                    paper_id,
                    "table_row",
                    table.page,
                    "table",
                    " | ".join(cell for cell in row.cells if cell),
                    table.table_id,
                    f"{table.table_id}:row_{row.row_index}",
                ),
            )


def _table_row_ids(paper_id: str, table_id: str, row_indices: Sequence[int]) -> Iterable[str]:
    for row_index in row_indices:
        yield f"evd_tbl_{paper_id}_{table_id}_r{row_index}"
