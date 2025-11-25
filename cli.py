import argparse
import json
from pathlib import Path

from data_pipeline import IngestionPipeline
from orchestrator import CrewManager


def create_pipeline() -> IngestionPipeline:
    """Instantiate the ingestion pipeline."""
    return IngestionPipeline()


def create_crew_manager() -> CrewManager:
    """Create and initialize the crew manager for ad-hoc queries."""
    manager = CrewManager()
    manager.initialize()
    return manager


def cmd_ingest_dir(args):
    pipeline = create_pipeline()
    result = pipeline.ingest_directory(
        directory=args.directory,
        force=args.force,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))


def cmd_ingest_file(args):
    pipeline = create_pipeline()
    result = pipeline.process_file(
        file_path=args.file,
        force=args.force,
    )
    print(json.dumps(result, indent=2))


def cmd_status(_args):
    pipeline = create_pipeline()
    status = pipeline.get_ingestion_status()
    print(json.dumps(status, indent=2))


def cmd_reset(args):
    pipeline = create_pipeline()
    pipeline.reset(clear_vector_store=args.clear_vector_store)
    print("Ingestion state reset.")


def cmd_query(args):
    manager = create_crew_manager()
    result = manager.execute_query(args.prompt)
    print(json.dumps(result.to_dict(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DAY10 utility CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    dir_parser = sub.add_parser("ingest-dir", help="Ingest an entire directory")
    dir_parser.add_argument("--directory", default="./data/raw")
    dir_parser.add_argument("--batch-size", type=int, default=10)
    dir_parser.add_argument("--force", action="store_true")
    dir_parser.set_defaults(func=cmd_ingest_dir)

    file_parser = sub.add_parser("ingest-file", help="Ingest a single file")
    file_parser.add_argument("file", help="File path to ingest")
    file_parser.add_argument("--force", action="store_true")
    file_parser.set_defaults(func=cmd_ingest_file)

    status_parser = sub.add_parser("status", help="Show ingestion status")
    status_parser.set_defaults(func=cmd_status)

    reset_parser = sub.add_parser("reset", help="Reset ingestion state")
    reset_parser.add_argument("--clear-vector-store", action="store_true", default=False)
    reset_parser.set_defaults(func=cmd_reset)

    query_parser = sub.add_parser("query", help="Run a quick agent query")
    query_parser.add_argument("prompt", help="Prompt for the crew to answer")
    query_parser.set_defaults(func=cmd_query)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

