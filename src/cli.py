from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import IngestConfig
from .orchestrator import run_full_pipeline
from .quality_checks import run_checks, summarize
from .steps.ingest import run as run_ingest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Impact pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Run PDF ingestion")
    ingest_parser.add_argument("pdf_path", help="Path to PDF file")
    ingest_parser.add_argument("--paper-id", default=None, help="Optional explicit paper_id")
    ingest_parser.add_argument("--output-root", default="outputs", help="Output folder root")
    ingest_parser.add_argument("--use-grobid", action="store_true", help="Enable optional GROBID enrichment")
    ingest_parser.add_argument("--grobid-url", default=None, help="GROBID base URL")
    ingest_parser.add_argument("--chunk-chars", type=int, default=1800, help="Chunk size in characters")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=220, help="Chunk overlap in characters")
    ingest_parser.add_argument("--min-chunk-chars", type=int, default=120, help="Minimum chunk length")
    ingest_parser.add_argument(
        "--metadata-pages-scan",
        type=int,
        default=4,
        help="Number of initial pages scanned for metadata extraction",
    )
    ingest_parser.add_argument(
        "--use-openai-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use OpenAI to refine metadata from first pages (default: enabled)",
    )
    ingest_parser.add_argument(
        "--use-openai-extraction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use OpenAI to refine structured extraction (default: enabled)",
    )
    ingest_parser.add_argument(
        "--openai-extraction-max-snippets",
        type=int,
        default=40,
        help="Batch size of snippets per OpenAI extraction call",
    )
    ingest_parser.add_argument(
        "--openai-effect-snippet-chars",
        type=int,
        default=1200,
        help="Maximum characters per snippet sent to OpenAI effect extractor",
    )
    ingest_parser.add_argument(
        "--openai-model",
        default="gpt-4.1-mini",
        help="OpenAI model for metadata refinement",
    )

    pipeline_parser = subparsers.add_parser("pipeline", help="Run full pipeline")
    pipeline_parser.add_argument("pdf_path", help="Path to PDF file")
    pipeline_parser.add_argument("--paper-id", default=None, help="Optional explicit paper_id")
    pipeline_parser.add_argument("--output-root", default="outputs", help="Output folder root")
    pipeline_parser.add_argument("--use-grobid", action="store_true", help="Enable optional GROBID enrichment")
    pipeline_parser.add_argument("--grobid-url", default=None, help="GROBID base URL")
    pipeline_parser.add_argument("--chunk-chars", type=int, default=1800, help="Chunk size in characters")
    pipeline_parser.add_argument("--chunk-overlap", type=int, default=220, help="Chunk overlap in characters")
    pipeline_parser.add_argument("--min-chunk-chars", type=int, default=120, help="Minimum chunk length")
    pipeline_parser.add_argument("--metadata-pages-scan", type=int, default=4, help="Number of pages for metadata extraction")
    pipeline_parser.add_argument(
        "--use-openai-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use OpenAI metadata refinement (default: enabled)",
    )
    pipeline_parser.add_argument(
        "--use-openai-extraction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use OpenAI extraction refinement (default: enabled)",
    )
    pipeline_parser.add_argument("--openai-extraction-max-snippets", type=int, default=40, help="Batch size of snippets per OpenAI extraction call")
    pipeline_parser.add_argument(
        "--openai-effect-snippet-chars",
        type=int,
        default=1200,
        help="Maximum characters per snippet sent to OpenAI effect extractor",
    )
    pipeline_parser.add_argument(
        "--openai-model", default="gpt-4.1-mini", help="OpenAI model for metadata refinement"
    )
    pipeline_parser.add_argument("--ui-delay", type=float, default=0.0, help="Optional delay between steps for UI readability")
    pipeline_parser.add_argument(
        "--strict-check",
        action="store_true",
        help="Return non-zero exit code if quality checks fail.",
    )

    check_parser = subparsers.add_parser("check", help="Run quality checks on a pipeline output folder")
    check_parser.add_argument("output_dir", help="Path to outputs/<paper_id>")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "ingest":
        config_updates = {
            "chunk_chars": args.chunk_chars,
            "chunk_overlap": args.chunk_overlap,
            "min_chunk_chars": args.min_chunk_chars,
            "use_grobid": args.use_grobid,
            "metadata_pages_scan": args.metadata_pages_scan,
            "use_openai_metadata": args.use_openai_metadata,
            "use_openai_extraction": args.use_openai_extraction,
            "openai_extraction_max_snippets": args.openai_extraction_max_snippets,
            "openai_effect_snippet_chars": args.openai_effect_snippet_chars,
            "openai_model": args.openai_model,
        }
        if args.grobid_url:
            config_updates["grobid_url"] = args.grobid_url

        config = IngestConfig(**config_updates)
        artifacts = run_ingest(
            pdf_path=args.pdf_path,
            output_root=args.output_root,
            paper_id=args.paper_id,
            config=config,
        )
        summary = {
            "paper_id": artifacts.metadata.paper_id,
            "pages": artifacts.metadata.num_pages,
            "passages": len(artifacts.text_index),
            "tables": len(artifacts.tables),
        }
        print(json.dumps(summary, indent=2))
        return

    if args.command == "check":
        results = run_checks(args.output_dir)
        print(summarize(results))
        if any(not result.ok for result in results):
            raise SystemExit(1)
        return

    if args.command == "pipeline":
        config_updates = {
            "chunk_chars": args.chunk_chars,
            "chunk_overlap": args.chunk_overlap,
            "min_chunk_chars": args.min_chunk_chars,
            "use_grobid": args.use_grobid,
            "metadata_pages_scan": args.metadata_pages_scan,
            "use_openai_metadata": args.use_openai_metadata,
            "use_openai_extraction": args.use_openai_extraction,
            "openai_extraction_max_snippets": args.openai_extraction_max_snippets,
            "openai_effect_snippet_chars": args.openai_effect_snippet_chars,
            "openai_model": args.openai_model,
        }
        if args.grobid_url:
            config_updates["grobid_url"] = args.grobid_url
        config = IngestConfig(**config_updates)
        summary = run_full_pipeline(
            pdf_path=args.pdf_path,
            output_root=args.output_root,
            paper_id=args.paper_id,
            ingest_config=config,
            visualization_delay_seconds=args.ui_delay,
        )
        print(summary.model_dump_json(indent=2))
        print("")
        check_results = run_checks(Path(summary.output_dir))
        print(summarize(check_results))
        if args.strict_check and any(not result.ok for result in check_results):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
