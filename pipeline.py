"""
Agentic TDD Generation Pipeline

Pipeline:

    Project / ZIP
        |
        v
    Safe Extraction
        |
        v
    Deterministic Scanner
        |
        v
    Semantic MAP
        |
        v
    Context Injection
        |
        v
    Semantic REDUCE
        |
        v
    Evidence Retrieval
        |
        v
    Architect Agent
        |
        v
    Critic / Quality Gate
        |
        v
    Manager / TDD Renderer
        |
        v
    Markdown TDD + Generation Manifest

The pipeline is deliberately evidence-grounded:
- Source facts come from deterministic scanning.
- Evidence chunks retain file and line provenance.
- Semantic chunking is deterministic.
- Unsupported claims should be rejected by the Architect/Critic layers.
"""

from __future__ import annotations

import json
import shutil
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


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"

TDD_FILENAME = "technical_design_document.md"
MANIFEST_FILENAME = "generation_manifest.json"

CHUNK_SIZE = 4500
CHUNK_OVERLAP = 500
CONTEXT_LINES = 3
MAX_CONTEXT_CHARS = 900

MAX_RETRIEVED_EVIDENCE = 23

MAX_FILE_CHARS = 200_000


SUPPORTED_EXTENSIONS = {
    ".py",
    ".dart",
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".kt",
    ".kts",
    ".cs",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".swift",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sql",
    ".yaml",
    ".yml",
    ".json",
    ".xml",
    ".md",
    ".txt",
}


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "node_modules",
    ".idea",
    ".vscode",
    "build",
    "dist",
    "target",
    ".dart_tool",
    "Pods",
    "bin",
    "obj",
    ".gradle",
    ".cache",
    "coverage",
    "output",
    "uploads",
    "uploaded_project",
    "final_tdd_files",
}


IMPORTANT_NAMES = {
    "readme.md",
    "pubspec.yaml",
    "pubspec.yml",
    "requirements.txt",
    "package.json",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "main.py",
    "main.dart",
}


# ============================================================
# TIME
# ============================================================

def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# ZIP SECURITY
# ============================================================

def safe_extract_zip(
    zip_path: str | Path,
    destination: str | Path,
) -> Path:
    """
    Safely extract a ZIP while preventing:
    - ZIP Slip / path traversal
    - Windows path-length failures
    - extraction of dependency/generated directories
    """

    zip_path = Path(zip_path).resolve()
    destination = Path(destination).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(
            f"ZIP file not found: {zip_path}"
        )

    if zip_path.suffix.lower() != ".zip":
        raise ValueError(
            "Input file must be a ZIP file."
        )

    if destination.exists():
        shutil.rmtree(destination)

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    ignored_dirs = {
        ".git",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "node_modules",
        ".idea",
        ".vscode",
        "build",
        "dist",
        "target",
        ".dart_tool",
        "pods",
        "bin",
        "obj",
        ".gradle",
        ".cache",
        "coverage",
        "output",
        "uploads",
        "uploaded_project",
    }

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:

        destination_root = destination.resolve()

        for member in archive.infolist():

            raw_name = str(
                member.filename
            ).replace(
                "\\",
                "/",
            )

            if not raw_name.strip():
                continue

            parts = [
                part
                for part in raw_name.split("/")
                if part not in {
                    "",
                    ".",
                }
            ]

            if not parts:
                continue

            # Prevent ZIP Slip / parent traversal.
            if any(
                part == ".."
                for part in parts
            ):
                raise RuntimeError(
                    "Unsafe ZIP file detected: "
                    f"{member.filename}"
                )

            # Prevent absolute Windows/Unix paths.
            if (
                raw_name.startswith("/")
                or raw_name.startswith("\\")
                or (
                    len(raw_name) >= 2
                    and raw_name[1] == ":"
                )
            ):
                raise RuntimeError(
                    "Unsafe ZIP file detected: "
                    f"{member.filename}"
                )

            # Skip dependency/generated/cache directories.
            if any(
                part.lower() in ignored_dirs
                for part in parts[:-1]
            ):
                continue

            relative_path = Path(
                *parts
            )

            target_path = (
                destination_root
                / relative_path
            ).resolve()

            # Final ZIP-Slip protection.
            try:
                target_path.relative_to(
                    destination_root
                )
            except ValueError as exc:
                raise RuntimeError(
                    "Unsafe ZIP file detected: "
                    f"{member.filename}"
                ) from exc

            # Windows MAX_PATH safety margin.
            if len(str(target_path)) > 240:
                print(
                    "Skipping overly long ZIP path: "
                    f"{member.filename}"
                )
                continue

            if member.is_dir():
                target_path.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                continue

            target_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with archive.open(
                member,
                "r",
            ) as source:
                with open(
                    target_path,
                    "wb",
                ) as destination_file:
                    shutil.copyfileobj(
                        source,
                        destination_file,
                    )

    return destination


# ============================================================
# SOURCE FILE DISCOVERY
# ============================================================

def find_source_files(
    project_root: str | Path,
) -> list[Path]:
    """
    Find supported source/documentation files while excluding
    generated and dependency directories.
    """

    project_root = Path(project_root).resolve()

    if not project_root.exists():
        return []

    result: list[Path] = []

    for path in project_root.rglob("*"):

        if not path.is_file():
            continue

        try:
            relative = path.relative_to(
                project_root
            )
        except ValueError:
            continue

        if any(
            part in IGNORED_DIRECTORIES
            for part in relative.parts[:-1]
        ):
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        result.append(path)

    return sorted(
        result,
        key=lambda p: str(p).lower(),
    )


# ============================================================
# PROJECT ROOT DETECTION
# ============================================================

def find_project_root(
    directory: str | Path,
) -> Path:
    """
    Detect a nested project directory inside an extracted ZIP.
    """

    directory = Path(directory).resolve()

    if not directory.exists():
        raise FileNotFoundError(
            f"Directory not found: {directory}"
        )

    current_files = find_source_files(
        directory
    )

    if current_files:
        return directory

    children = [
        child
        for child in directory.iterdir()
        if child.is_dir()
        and child.name not in IGNORED_DIRECTORIES
    ]

    best_directory: Path | None = None
    best_count = 0

    for child in children:

        child_files = find_source_files(
            child
        )

        if len(child_files) > best_count:
            best_directory = child
            best_count = len(child_files)

    if best_directory is not None:
        return best_directory

    return directory


# ============================================================
# FILE PRIORITY
# ============================================================

def priority_score(
    path: Path,
    project_root: Path,
) -> int:
    """
    Give important project files higher retrieval priority.
    """

    relative = path.relative_to(
        project_root
    )

    relative_string = (
        str(relative)
        .replace("\\", "/")
        .lower()
    )

    filename = path.name.lower()

    score = 0

    if filename in IMPORTANT_NAMES:
        score += 100

    if relative_string == "readme.md":
        score += 250

    if relative_string == "pubspec.yaml":
        score += 250

    if relative_string == "lib/main.dart":
        score += 300

    if relative_string == "main.py":
        score += 200

    if "/lib/" in f"/{relative_string}":
        score += 100

    if "/src/" in f"/{relative_string}":
        score += 80

    if "/app/" in f"/{relative_string}":
        score += 60

    if (
        "/test/" in f"/{relative_string}"
        or filename.startswith("test_")
        or filename.endswith("_test.py")
        or filename.endswith("_test.dart")
    ):
        score += 30

    platform_directories = (
        "/android/",
        "/ios/",
        "/linux/",
        "/macos/",
        "/windows/",
    )

    for platform in platform_directories:
        if platform in f"/{relative_string}":
            score -= 20

    if "generated_plugin" in relative_string:
        score -= 50

    return score


def sort_source_files(
    files: list[Path],
    project_root: Path,
) -> list[Path]:

    return sorted(
        files,
        key=lambda path: (
            -priority_score(
                path,
                project_root,
            ),
            str(path).lower(),
        ),
    )


# ============================================================
# READ PROJECT
# ============================================================

def read_project(
    project_root: Path,
) -> list[dict[str, Any]]:
    """
    Read supported project files.

    Files exceeding MAX_FILE_CHARS are skipped rather than
    silently truncated.
    """

    files = sort_source_files(
        find_source_files(project_root),
        project_root,
    )

    records: list[dict[str, Any]] = []

    for path in files:

        try:
            content = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except Exception as exc:
            print(
                f"WARNING: Could not read "
                f"{path}: {exc}"
            )
            continue

        if not content.strip():
            continue

        if len(content) > MAX_FILE_CHARS:
            print(
                f"WARNING: Skipping oversized file "
                f"{path} ({len(content)} chars)"
            )
            continue

        relative = path.relative_to(
            project_root
        )

        records.append(
            {
                "path": str(relative).replace(
                    "\\",
                    "/",
                ),
                "name": path.name,
                "content": content,
            }
        )

    return records


# ============================================================
# SOURCE STATISTICS
# ============================================================

def build_source_statistics(
    scanner_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Preserve deterministic scanner statistics in the manifest.
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


# ============================================================
# DETERMINISTIC SCANNER
# ============================================================

def scan_project(
    project_root: Path,
) -> dict[str, Any]:

    print()
    print("=" * 64)
    print("1. DETERMINISTIC SOURCE SCANNING")
    print("=" * 64)

    scanner = ScannerAgent()

    scanner_data = scanner.scan_files(
        project_root
    )

    if not isinstance(
        scanner_data,
        dict,
    ):
        raise ValueError(
            "Scanner returned invalid data."
        )

    print(
        f"Files analyzed     : "
        f"{scanner_data.get('file_count', 0)}"
    )

    print(
        f"Characters         : "
        f"{scanner_data.get('character_count', 0)}"
    )

    print(
        f"Lines              : "
        f"{scanner_data.get('line_count', 0)}"
    )

    print(
        f"Non-empty lines    : "
        f"{scanner_data.get('non_empty_lines', 0)}"
    )

    return scanner_data


# ============================================================
# SEMANTIC MAP-REDUCE CHUNKING
# ============================================================

def create_semantic_chunks(
    scanner_data: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Execute:

        MAP
        CONTEXT INJECTION
        REDUCE

    The chunker itself owns the actual deterministic semantic
    boundary logic.
    """

    print()
    print("=" * 64)
    print("2. SEMANTIC MAP-REDUCE CHUNKING")
    print("=" * 64)

    files = scanner_data.get(
        "files",
        [],
    )

    if not isinstance(
        files,
        list,
    ):
        files = []

    chunker = SemanticChunker(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        context_lines=CONTEXT_LINES,
        max_context_chars=MAX_CONTEXT_CHARS,
    )

    # Explicit MAP stage for measurable pipeline evidence.
    mapped_units = chunker.map_files(
        files
    )

    # Complete MAP -> Context -> REDUCE operation.
    chunks = chunker.split_files(
        files
    )

    if not isinstance(
        chunks,
        list,
    ):
        chunks = []

    semantic_types = sorted(
        {
            str(
                chunk.get(
                    "kind",
                    "unknown",
                )
            )
            for chunk in chunks
        }
    )

    context_injection = bool(chunks) and all(
        "[CONTEXT]" in str(
            chunk.get(
                "text",
                "",
            )
        )
        and "[/CONTEXT]" in str(
            chunk.get(
                "text",
                "",
            )
        )
        for chunk in chunks
    )

    source_wrapping = bool(chunks) and all(
        "[SOURCE]" in str(
            chunk.get(
                "text",
                "",
            )
        )
        and "[/SOURCE]" in str(
            chunk.get(
                "text",
                "",
            )
        )
        for chunk in chunks
    )

    evidence_ids = [
        str(
            chunk.get(
                "id",
                "",
            )
        )
        for chunk in chunks
        if chunk.get("id")
    ]

    unique_evidence_ids = (
        len(evidence_ids)
        == len(set(evidence_ids))
    )

    chunking_metrics = {
        "strategy": "semantic-map-reduce",
        "map_units": len(mapped_units),
        "reduced_chunks": len(chunks),
        "context_injection": context_injection,
        "source_provenance_wrapping": source_wrapping,
        "stable_evidence_ids": unique_evidence_ids,
        "semantic_types": semantic_types,
        "chunk_size_chars": CHUNK_SIZE,
        "chunk_overlap_chars": CHUNK_OVERLAP,
        "context_lines": CONTEXT_LINES,
        "max_context_chars": MAX_CONTEXT_CHARS,
        "source_characters": scanner_data.get(
            "character_count",
            0,
        ),
        "source_lines": scanner_data.get(
            "line_count",
            0,
        ),
        "over_150k_character_input": (
            scanner_data.get(
                "character_count",
                0,
            )
            >= 150_000
        ),
    }

    print(
        f"MAP units          : "
        f"{len(mapped_units)}"
    )

    print(
        f"Reduced chunks     : "
        f"{len(chunks)}"
    )

    print(
        f"Context injection  : "
        f"{context_injection}"
    )

    print(
        f"Stable evidence IDs: "
        f"{unique_evidence_ids}"
    )

    print(
        f"Semantic types     : "
        f"{', '.join(semantic_types) or 'none'}"
    )

    print(
        f"150k+ input        : "
        f"{chunking_metrics['over_150k_character_input']}"
    )

    if not chunks:
        raise RuntimeError(
            "Semantic chunking produced no chunks."
        )

    return chunks, chunking_metrics


# ============================================================
# EVIDENCE RETRIEVAL
# ============================================================

def retrieve_evidence(
    store: CodeVectorStore,
    maximum: int = MAX_RETRIEVED_EVIDENCE,
) -> list[dict[str, Any]]:
    """
    Retrieve diverse evidence across architectural concerns.

    Multiple queries reduce the chance that the Architect is
    dominated by a single source file.
    """

    print()
    print("=" * 64)
    print("3. DIVERSE EVIDENCE RETRIEVAL")
    print("=" * 64)

    queries = [
        "project purpose overview requirements",
        "application entry point main workflow",
        "architecture components modules services",
        "core business logic processing algorithm",
        "API endpoints routes requests responses",
        "database storage persistence models",
        "authentication authorization security permissions",
        "configuration environment variables dependencies",
        "errors exceptions validation retry handling",
        "tests test cases testing framework coverage",
        "deployment platforms build runtime",
        "performance memory latency concurrency",
    ]

    retrieved: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for query in queries:

        try:
            results = store.search_with_metadata(
                query,
                top_k=4,
            )
        except Exception as exc:
            print(
                f"WARNING: Retrieval failed for "
                f"'{query}': {exc}"
            )
            continue

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
                print(
                    f"Retrieved {len(retrieved)} evidence chunks."
                )
                return retrieved

    print(
        f"Retrieved {len(retrieved)} evidence chunks."
    )

    return retrieved


# ============================================================
# IMPORTANT FILE EVIDENCE
# ============================================================

def add_important_file_evidence(
    records: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    maximum: int = MAX_RETRIEVED_EVIDENCE,
) -> list[dict[str, Any]]:
    """
    Ensure important project files can remain visible to the
    Architect even when lexical retrieval scores them poorly.
    """

    important_paths = {
        str(
            record.get(
                "path",
                "",
            )
        ).replace(
            "\\",
            "/",
        )
        for record in records
        if str(
            record.get(
                "name",
                "",
            )
        ).lower()
        in IMPORTANT_NAMES
    }

    if not important_paths:
        return evidence

    selected: dict[str, dict[str, Any]] = {}

    for item in evidence:

        evidence_id = str(
            item.get(
                "id",
                "",
            )
        ).strip()

        if evidence_id:
            selected[evidence_id] = item

    for chunk in chunks:

        path = str(
            chunk.get(
                "path",
                "",
            )
        ).replace(
            "\\",
            "/",
        ).strip()
        
        # Never re-introduce generated TDD artifacts
        # into the source evidence set.
        if (
            path == "final_tdd_files"
            or path.startswith("final_tdd_files/")
            or "/final_tdd_files/" in path
        ):
            continue

        if path not in important_paths:
            continue

        evidence_id = str(
            chunk.get(
                "id",
                "",
            )
        ).strip()

        if not evidence_id:
            continue

        selected[evidence_id] = {
            "id": evidence_id,
            "text": chunk.get(
                "text",
                "",
            ),
            "metadata": {
                "id": evidence_id,
                "path": path,
                "start_line": chunk.get(
                    "start_line"
                ),
                "end_line": chunk.get(
                    "end_line"
                ),
                "language": chunk.get(
                    "language",
                    "",
                ),
                "kind": chunk.get(
                    "kind",
                    "",
                ),
                "symbol": chunk.get(
                    "symbol",
                    "",
                ),
            },
            "score": 1.0,
        }

    result = sorted(
        selected.values(),
        key=lambda item: float(
            item.get(
                "score",
                0.0,
            )
        ),
        reverse=True,
    )

    return result[:maximum]


# ============================================================
# ARCHITECT AGENT
# ============================================================

def run_architect(
    project_name: str,
    scanner_data: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:

    print()
    print("=" * 64)
    print("4. ARCHITECT AGENT")
    print("=" * 64)

    architect = ArchitectAgent()

    blueprint = architect.build_design_blueprint(
        filename=project_name,
        scanner_data=scanner_data,
        relevant_chunks=evidence,
    )

    if not isinstance(
        blueprint,
        dict,
    ):
        raise ValueError(
            "Architect returned invalid blueprint."
        )

    blueprint["project_name"] = (
        project_name
    )

    blueprint["project_file_count"] = (
        scanner_data.get(
            "file_count",
            0,
        )
    )

    blueprint["retrieved_evidence_ids"] = [
        str(
            item.get(
                "id",
                "",
            )
        )
        for item in evidence
        if item.get("id")
    ]

    architecture = blueprint.get(
        "architecture",
        {},
    )

    if not isinstance(
        architecture,
        dict,
    ):
        architecture = {}

    components = architecture.get(
        "components",
        [],
    )

    if not isinstance(
        components,
        list,
    ):
        components = []

    requirements = blueprint.get(
        "functional_requirements",
        [],
    )

    if not isinstance(
        requirements,
        list,
    ):
        requirements = []

    print(
        f"Architecture components : "
        f"{len(components)}"
    )

    print(
        f"Functional requirements  : "
        f"{len(requirements)}"
    )

    return blueprint


# ============================================================
# CRITIC / QUALITY GATE
# ============================================================

def run_critic(
    blueprint: dict[str, Any],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:

    print()
    print("=" * 64)
    print("5. CRITIC / QUALITY GATE")
    print("=" * 64)

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

    print(
        f"Quality score : "
        f"{quality.get('score', 'N/A')}/10"
    )

    print(
        "Quality gate  : "
        + (
            "PASS"
            if quality.get("passed")
            else "REVIEW"
        )
    )

    warnings = quality.get(
        "warnings",
        [],
    )

    if warnings:
        print(
            f"Warnings      : {len(warnings)}"
        )

    return quality


# ============================================================
# MANAGER / TDD RENDERER
# ============================================================

def generate_tdd(
    blueprint: dict[str, Any],
) -> Path:

    print()
    print("=" * 64)
    print("6. MANAGER / TDD RENDERER")
    print("=" * 64)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manager = ManagerAgent(
        output_directory=str(
            OUTPUT_DIR
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

    output_file = manager.save_tdd(
        document,
        filename=TDD_FILENAME,
    )

    output_file = Path(
        output_file
    ).resolve()

    print(
        f"TDD saved to: {output_file}"
    )

    return output_file


# ============================================================
# MANIFEST
# ============================================================

def save_manifest(
    manifest: dict[str, Any],
) -> Path:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        OUTPUT_DIR
        / MANIFEST_FILENAME
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return manifest_path


# ============================================================
# PROJECT PIPELINE
# ============================================================

def run_project_pipeline(
    project_directory: str | Path,
) -> dict[str, Any]:
    """
    Run the complete project pipeline.

    This is the primary internal entry point.
    """

    project_directory = (
        Path(project_directory)
        .resolve()
    )

    if not project_directory.exists():
        raise FileNotFoundError(
            f"Project directory not found: "
            f"{project_directory}"
        )

    project_root = find_project_root(
        project_directory
    )

    project_name = project_root.name

    print()
    print("=" * 64)
    print("AGENTIC TDD GENERATION PIPELINE")
    print("=" * 64)
    print(
        f"Project root: {project_root}"
    )

    # --------------------------------------------------------
    # Read records
    # --------------------------------------------------------

    records = read_project(
        project_root
    )

    if not records:
        raise RuntimeError(
            "No readable supported source files found."
        )

    print(
        f"Readable files: {len(records)}"
    )

    # --------------------------------------------------------
    # Deterministic Scanner
    # --------------------------------------------------------

    scanner_data = scan_project(
        project_root
    )

    source_statistics = (
        build_source_statistics(
            scanner_data
        )
    )

    # --------------------------------------------------------
    # Semantic Map-Reduce
    # --------------------------------------------------------

    chunks, chunking_metrics = (
        create_semantic_chunks(
            scanner_data
        )
    )

    # --------------------------------------------------------
    # Retrieval Store
    # --------------------------------------------------------

    print()
    print("=" * 64)
    print("BUILDING RETRIEVAL STORE")
    print("=" * 64)

    store = CodeVectorStore(
        persist_directory=str(
            OUTPUT_DIR
            / "rag_store"
        ),
        model_name="all-MiniLM-L6-v2",
        mode="lexical",
    )

    store.add_chunks(
        chunks
    )

    # --------------------------------------------------------
    # Diverse retrieval
    # --------------------------------------------------------

    evidence = retrieve_evidence(
        store,
        maximum=MAX_RETRIEVED_EVIDENCE,
    )

    evidence = add_important_file_evidence(
        records,
        chunks,
        evidence,
        maximum=MAX_RETRIEVED_EVIDENCE,
    )

    # --------------------------------------------------------
    # Architect
    # --------------------------------------------------------

    blueprint = run_architect(
        project_name,
        scanner_data,
        evidence,
    )

    # --------------------------------------------------------
    # Attach deterministic pipeline metadata
    # --------------------------------------------------------

    blueprint[
        "project_name"
    ] = project_name

    blueprint[
        "project_file_count"
    ] = len(records)

    blueprint[
        "project_files"
    ] = [
        record["path"]
        for record in records
    ]

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

    blueprint[
        "chunking"
    ] = chunking_metrics

    # --------------------------------------------------------
    # Critic
    # --------------------------------------------------------

    quality = run_critic(
        blueprint,
        evidence,
    )

    blueprint[
        "quality_review"
    ] = quality

    # --------------------------------------------------------
    # Manager
    # --------------------------------------------------------

    output_file = generate_tdd(
        blueprint
    )

    # --------------------------------------------------------
    # Manifest
    # --------------------------------------------------------

    evidence_ids = [
        str(
            item.get(
                "id",
                "",
            )
        )
        for item in evidence
        if item.get("id")
    ]

    manifest = {
        "pipeline_version": "v2-semantic-agents",
        "project": project_name,
        "project_root": str(
            project_root
        ),
        "generated_at": utc_now(),

        "source_statistics": (
            source_statistics
        ),

        "files_analyzed": len(
            records
        ),

        "chunks_created": len(
            chunks
        ),

        "evidence_retrieved": len(
            evidence
        ),

        "chunking": (
            chunking_metrics
        ),

        "evidence_ids": evidence_ids,

        "quality": quality,

        "output": str(
            output_file
        ),
    }

    manifest_path = save_manifest(
        manifest
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = {
        "success": True,
        "project": project_name,
        "project_root": str(
            project_root
        ),
        "file_count": len(
            records
        ),
        "chunk_count": len(
            chunks
        ),
        "rag_count": len(
            evidence
        ),
        "quality": quality,
        "output_file": str(
            output_file
        ),
        "manifest_file": str(
            manifest_path
        ),
        "source_statistics": (
            source_statistics
        ),
        "chunking": (
            chunking_metrics
        ),
        "evidence_ids": evidence_ids,
    }

    print()
    print("=" * 64)
    print("PIPELINE COMPLETED")
    print("=" * 64)
    print(
        f"Files analyzed : {result['file_count']}"
    )
    print(
        f"MAP units      : "
        f"{chunking_metrics['map_units']}"
    )
    print(
        f"Reduced chunks : "
        f"{chunking_metrics['reduced_chunks']}"
    )
    print(
        f"Evidence used  : {result['rag_count']}"
    )
    print(
        f"Quality        : "
        f"{quality.get('score', 'N/A')}/10"
    )
    print(
        f"Quality gate   : "
        f"{'PASS' if quality.get('passed') else 'REVIEW'}"
    )
    print(
        f"TDD            : {output_file}"
    )
    print(
        f"Manifest       : {manifest_path}"
    )
    print("=" * 64)

    return result


# ============================================================
# ZIP PIPELINE
# ============================================================

def run_zip_pipeline(
    zip_path: str | Path,
) -> dict[str, Any]:

    zip_path = Path(
        zip_path
    ).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(
            f"ZIP file not found: "
            f"{zip_path}"
        )

    temp_root = Path(
        tempfile.mkdtemp(
            prefix="agentic_tdd_zip_"
        )
    )

    try:

        extracted_root = (
            temp_root
            / "project"
        )

        safe_extract_zip(
            zip_path,
            extracted_root,
        )

        project_root = find_project_root(
            extracted_root
        )

        return run_project_pipeline(
            project_root
        )

    finally:

        shutil.rmtree(
            temp_root,
            ignore_errors=True,
        )


# ============================================================
# FLASK / WEB COMPATIBILITY
# ============================================================

def run_pipeline(
    source_file: str | Path | None = None,
):
    """
    Compatibility function imported by app.py.

    Supports:
    - ZIP
    - Project directory
    - Single source file

    Returns the complete pipeline result dictionary so the
    Flask web layer can expose quality, statistics, history,
    and download information.
    """

    if source_file is None:
        source_file = (
            BASE_DIR
            / "sample.py"
        )

    source_path = Path(
        source_file
    ).resolve()

    if not source_path.exists():
        raise FileNotFoundError(
            f"Input not found: "
            f"{source_path}"
        )

    if (
        source_path.is_file()
        and source_path.suffix.lower()
        == ".zip"
    ):
        return run_zip_pipeline(
            source_path
        )

    if source_path.is_dir():
        return run_project_pipeline(
            source_path
        )

    if source_path.is_file():
        return run_project_pipeline(
            source_path.parent
        )

    raise ValueError(
        "Unsupported input type."
    )


# ============================================================
# CLI
# ============================================================

def main() -> int:

    if len(sys.argv) < 2:

        print()
        print(
            "Usage:"
        )
        print(
            "  python pipeline.py project.zip"
        )
        print(
            "  python pipeline.py project_folder"
        )
        print(
            "  python pipeline.py source.py"
        )
        print()

        return 1

    input_path = Path(
        sys.argv[1]
    ).resolve()

    if not input_path.exists():

        print(
            f"ERROR: Input not found: "
            f"{input_path}"
        )

        return 1

    try:

        if (
            input_path.is_file()
            and input_path.suffix.lower()
            == ".zip"
        ):

            result = run_zip_pipeline(
                input_path
            )

        elif input_path.is_dir():

            result = run_project_pipeline(
                input_path
            )

        elif input_path.is_file():

            result = run_project_pipeline(
                input_path.parent
            )

        else:

            print(
                "ERROR: Unsupported input type."
            )

            return 1

        print()
        print("=" * 64)
        print(
            "FINAL RESULT"
        )
        print("=" * 64)

        print(
            f"Project files analyzed : "
            f"{result['file_count']}"
        )

        print(
            f"MAP units              : "
            f"{result['chunking']['map_units']}"
        )

        print(
            f"Semantic chunks        : "
            f"{result['chunk_count']}"
        )

        print(
            f"Evidence chunks used   : "
            f"{result['rag_count']}"
        )

        print(
            f"Quality score           : "
            f"{result['quality'].get('score', 'N/A')}/10"
        )

        print(
            f"Quality gate            : "
            f"{'PASS' if result['quality'].get('passed') else 'REVIEW'}"
        )

        print(
            f"TDD                     : "
            f"{result['output_file']}"
        )

        print(
            f"Manifest                : "
            f"{result['manifest_file']}"
        )

        print(
            f"150k+ input             : "
            f"{result['chunking']['over_150k_character_input']}"
        )

        print("=" * 64)

        return 0

    except Exception as exc:

        print()
        print("=" * 64)
        print(
            "PIPELINE FAILED"
        )
        print("=" * 64)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print("=" * 64)

        return 1


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )