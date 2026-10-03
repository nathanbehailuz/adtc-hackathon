"""Command-line entrypoint for the v8 foundation and supply census."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .census import DEFAULT_OUTPUT, run_census
from .identity import DEFAULT_REGISTRY, initialize_registry
from .schema import canonical_json_schema
from .sources import DEFAULT_SOURCES, load_sources
from .validate import validate_jsonl

ADTC_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = (
    ADTC_ROOT / "data" / "manifests" / "v8" / "tebeb-v8.1.schema.json"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def export_schema(output: Path) -> dict[str, Any]:
    schema = canonical_json_schema()
    _write_json(output, schema)
    return {"schema": str(output), "title": schema["title"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    schema_parser = subparsers.add_parser("export-schema")
    schema_parser.add_argument("--output", type=Path, default=DEFAULT_SCHEMA_PATH)

    registry_parser = subparsers.add_parser("init-registry")
    registry_parser.add_argument("--path", type=Path, default=DEFAULT_REGISTRY)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument("--report", type=Path)

    census_parser = subparsers.add_parser("census")
    census_parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    census_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    census_parser.add_argument("--inventory", type=Path, action="append", default=[])

    gate_parser = subparsers.add_parser("gate-a")
    gate_parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    gate_parser.add_argument("--schema-output", type=Path, default=DEFAULT_SCHEMA_PATH)
    gate_parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    gate_parser.add_argument("--census-output", type=Path, default=DEFAULT_OUTPUT)
    gate_parser.add_argument("--inventory", type=Path, action="append", default=[])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "export-schema":
        result = export_schema(args.output)
    elif args.command == "init-registry":
        result = initialize_registry(args.path)
    elif args.command == "validate":
        report = validate_jsonl(args.path)
        result = report.to_dict()
        if args.report:
            _write_json(args.report, result)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if report.valid else 1)
    elif args.command == "census":
        result = run_census(
            sources_path=args.sources,
            output_path=args.output,
            extra_inventory_paths=args.inventory,
        )
    elif args.command == "gate-a":
        manifest = load_sources(args.sources)
        schema_result = export_schema(args.schema_output)
        registry_result = initialize_registry(args.registry)
        census_result = run_census(
            sources_path=args.sources,
            output_path=args.census_output,
            extra_inventory_paths=args.inventory,
        )
        result = {
            "phase": "A",
            "schema": schema_result,
            "sources": {
                "path": str(args.sources),
                "count": len(manifest.sources),
            },
            "registry": registry_result,
            "census": {
                "path": str(args.census_output),
                **census_result["summary"],
                **census_result["gate_a"],
            },
        }
    else:  # pragma: no cover
        raise AssertionError(args.command)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
