"""
Agentic TDD Generation Pipeline.

Flow:

ZIP
  ↓
Safe Extraction
  ↓
Deterministic Scanner
  ↓
Source-aware Semantic Chunking
  ↓
Diverse Evidence Retrieval
  ↓
Architect Agent
  ↓
Critic / Quality Gate
  ↓
Manager / TDD Renderer
  ↓
Traceable Markdown + Manifest
"""

from __future__ import annotations

import json
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.architect import ArchitectAgent
from agents.critic import CriticAgent
from agents.manager import ManagerAgent
from agents.scanner import ScannerAgent
from parser.chunker import SemanticChunker
from rag.vector_store import CodeVectorStore


# ================================================================
# GENERAL HELPERS
# ================================================================


def utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )


def safe_project_name(
    zip_path: str,
) -> str:
    return Path(zip_path).stem


# ================================================================
# SAFE ZIP EXTRACTION
# ================================================================


def safe_extract_zip(
    zip_path: str,
    destination: str | None = None,
) -> tuple[str, str]:
    """
    Safely extract a ZIP archive.

    Prevents ZIP Slip/path traversal attacks.

    Returns:
        (project_name, project_root)
    """

    archive_path = Path(zip_path)

    if not archive_path.is_file():
        raise FileNotFoundError(
            f"ZIP file not found: {archive_path}"
        )

    if not zipfile.is_zipfile(
        archive_path
    ):
        raise ValueError(
            "Input file is not a valid ZIP archive."
        )

    project_name = safe_project_name(
        str(archive_path)
    )

    if destination:
        extraction_root = Path(
            destination
        )
        extraction_root.mkdir(
            parents=True,
            exist_ok=True,
        )
    else:
        extraction_root = Path(
            tempfile.mkdtemp(
                prefix="agentic_tdd_zip_"
            )
        )

    extraction_root = (
        extraction_root.resolve()
    )

    with zipfile.ZipFile(
        archive_path,
        "r",
    ) as archive:

        for member in archive.infolist():

            target = (
                extraction_root
                / member.filename
            ).resolve()

            # ZIP Slip protection.
            if (
                target != extraction_root
                and extraction_root
                not in target.parents
            ):
                raise ValueError(
                    "Unsafe ZIP entry detected: "
                    f"{member.filename}"
                )

            if member.is_dir():
                target.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                continue

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with archive.open(
                member,
                "r",
            ) as source:

                with open(
                    target,
                    "wb",
                ) as destination_file:

                    destination_file.write(
                        source.read()
                    )

    # Ignore macOS metadata.
    entries = [
        item
        for item in extraction_root.iterdir()
        if item.name != "__MACOSX"
    ]

    # If ZIP contains one project folder,
    # use that folder as the project root.
    if (
        len(entries) == 1
        and entries[0].is_dir()
    ):
        project_root = entries[0]
    else:
        project_root = extraction_root

    return (
        project_name,
        str(project_root),
    )


# ================================================================
# SOURCE STATISTICS
# ================================================================


def build_source_statistics(
    scanner_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Preserve deterministic scanner statistics.
    """

    return {
        "characters": scanner_data.get(
            "character_count",
            0,
        ),
        "lines": scanner_data.get(
            "line_count",
            0,
        ),
        "non_empty_lines": scanner_data.get(
            "non_empty_lines",
            0,
        ),
        "files": scanner_data.get(
            "file_count",
            0,
        ),
        "test_files": scanner_data.get(
            "test_file_count",
            0,
        ),
        "languages": scanner_data.get(
            "language_counts",
            {},
        ),
    }


# ================================================================
# DIVERSE EVIDENCE RETRIEVAL
# ================================================================


def retrieve_evidence(
    store: CodeVectorStore,
    maximum: int = 23,
) -> list[dict[str, Any]]:
    """
    Retrieve diverse architecture-relevant evidence.

    Multiple independent queries are used so the Architect
    does not construct the TDD around a single source file.
    """

    queries = [
        "application entry point main UI screen workflow",
        "core business logic processing algorithm",
        "network HTTP API endpoint request response",
        "dependencies packages configuration environment",
        "file upload download storage archive audio",
        "state management user interaction error handling",
        "platform Android iOS permissions native integration",
        "tests test cases validation",
        "README setup installation usage architecture",
    ]

    retrieved: list[
        dict[str, Any]
    ] = []

    seen_ids: set[str] = set()

    # ------------------------------------------------------------
    # First pass:
    # Get evidence from every architectural topic.
    # ------------------------------------------------------------

    for query in queries:

        results = store.search_with_metadata(
            query,
            top_k=4,
        )

        for result in results:

            if not isinstance(
                result,
                dict,
            ):
                continue

            evidence_id = str(
                result.get(
                    "id",
                    "",
                )
            ).strip()

            if not evidence_id:
                continue

            if evidence_id in seen_ids:
                continue

            seen_ids.add(
                evidence_id
            )

            retrieved.append(
                result
            )

            if len(retrieved) >= maximum:
                return retrieved

    return retrieved


# ================================================================
# MANIFEST
# ================================================================


def save_manifest(
    output_dir: Path,
    manifest: dict[str, Any],
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        output_dir
        / "generation_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ================================================================
# MAIN PIPELINE
# ================================================================


def main() -> int:

    if len(sys.argv) < 2:

        print(
            'Usage: python pipeline.py "path\\to\\project.zip"'
        )

        return 1

    zip_path = sys.argv[1]

    output_dir = Path(
        "output"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        # ========================================================
        # 1. SAFE EXTRACTION
        # ========================================================

        project_name, project_root = (
            safe_extract_zip(
                zip_path
            )
        )

        # ========================================================
        # 2. DETERMINISTIC SCANNING
        # ========================================================

        scanner = ScannerAgent()

        scanner_data = scanner.scan_files(
            Path(project_root)
        )

        if not isinstance(
            scanner_data,
            dict,
        ):
            raise ValueError(
                "Scanner returned invalid data."
            )

        files = scanner_data.get(
            "files",
            [],
        )

        if not isinstance(
            files,
            list,
        ):
            files = []

        source_statistics = (
            build_source_statistics(
                scanner_data
            )
        )

        # ========================================================
        # 3. SOURCE-AWARE CHUNKING
        # ========================================================

        chunker = SemanticChunker(
            chunk_size=4500,
            chunk_overlap=500,
        )

        chunks = chunker.split_files(
            files
        )

        if not isinstance(
            chunks,
            list,
        ):
            chunks = []

        # ========================================================
        # 4. RETRIEVAL STORE
        # ========================================================

        store = CodeVectorStore(
            persist_directory=(
                "output/rag_store"
            ),
            mode="lexical",
        )

        store.add_chunks(
            chunks
        )

        # ========================================================
        # 5. DIVERSE EVIDENCE RETRIEVAL
        # ========================================================

        evidence = retrieve_evidence(
            store,
            maximum=23,
        )

        # ========================================================
        # 6. ARCHITECT
        # ========================================================

        architect = ArchitectAgent()

        blueprint = (
            architect.build_design_blueprint(
                filename=project_name,
                scanner_data=scanner_data,
                relevant_chunks=evidence,
            )
        )

        if not isinstance(
            blueprint,
            dict,
        ):
            raise ValueError(
                "Architect returned invalid blueprint."
            )

        # Keep deterministic traceability metadata.
        blueprint[
            "project_file_count"
        ] = len(files)

        blueprint[
            "retrieved_evidence_ids"
        ] = [
            str(
                item.get(
                    "id",
                    "",
                )
            )
            for item in evidence
            if item.get("id")
        ]

        # ========================================================
        # 7. CRITIC
        # ========================================================

        critic = CriticAgent()

        quality = critic.review(
            blueprint,
            evidence,
        )

        if not isinstance(
            quality,
            dict,
        ):
            raise ValueError(
                "Critic returned invalid quality result."
            )

        # Manager reads quality_review.
        blueprint[
            "quality_review"
        ] = quality

        # ========================================================
        # 8. MANAGER
        # ========================================================

        manager = ManagerAgent(
            output_directory=str(
                output_dir
            )
        )

        document = manager.generate_tdd(
            blueprint
        )

        if not isinstance(
            document,
            str,
        ):
            document = str(
                document
            )

        # ========================================================
        # 9. SAVE TDD
        # ========================================================

        tdd_path = manager.save_tdd(
            document,
            "technical_design_document.md",
        )

        # ========================================================
        # 10. GENERATION MANIFEST
        # ========================================================

        manifest = {
            "project": project_name,
            "project_root": project_root,
            "files_analyzed": len(
                files
            ),
            "chunks_created": len(
                chunks
            ),
            "evidence_retrieved": len(
                evidence
            ),
            "source_statistics": (
                source_statistics
            ),
            "quality": quality,
            "evidence_ids": [
                str(
                    item.get(
                        "id",
                        "",
                    )
                )
                for item in evidence
                if item.get("id")
            ],
            "output": str(
                tdd_path.resolve()
            ),
            "generated_at": utc_now(),
        }

        save_manifest(
            output_dir,
            manifest,
        )

        # ========================================================
        # 11. FINAL SUMMARY
        # ========================================================

        print()
        print(
            "=" * 64
        )

        print(
            "AGENTIC TDD GENERATION COMPLETED"
        )

        print(
            "=" * 64
        )

        print(
            f"Project files analyzed : "
            f"{len(files)}"
        )

        print(
            f"Semantic chunks        : "
            f"{len(chunks)}"
        )

        print(
            f"Evidence chunks used   : "
            f"{len(evidence)}"
        )

        print(
            f"Quality score          : "
            f"{quality.get('score', 'N/A')}/10"
        )

        print(
            f"Quality gate           : "
            f"{'PASS' if quality.get('passed') else 'REVIEW'}"
        )

        print(
            f"TDD saved to           : "
            f"{tdd_path.resolve()}"
        )

        warnings = quality.get(
            "warnings",
            [],
        )

        if warnings:

            print()
            print(
                "Quality warnings:"
            )

            for warning in warnings[:5]:

                print(
                    f"- {warning}"
                )

        issues = quality.get(
            "issues",
            [],
        )

        if issues:

            print()
            print(
                "Quality issues:"
            )

            for issue in issues[:5]:

                print(
                    f"- {issue}"
                )

        return 0

    except Exception as exc:

        print()
        print(
            "=" * 64
        )

        print(
            "PIPELINE FAILED"
        )

        print(
            "=" * 64
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )