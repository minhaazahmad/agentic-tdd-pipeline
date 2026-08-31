import sys
import shutil
import zipfile
from pathlib import Path

from agents.scanner import ScannerAgent
from agents.architect import ArchitectAgent
from agents.manager import ManagerAgent
from parser.chunker import SemanticChunker
from rag.vector_store import CodeVectorStore


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

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

    # TDD pipeline folders
    "output",
    "uploads",
    "uploaded_project",
}


# ============================================================
# IMPORTANT PROJECT FILES
# ============================================================

PRIORITY_FILE_NAMES = {
    "readme.md",
    "pubspec.yaml",
    "pubspec.yml",
    "main.dart",
    "main.py",
    "requirements.txt",
    "package.json",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
}


# ============================================================
# PROJECT-SPECIFIC PRIORITY PATHS
# ============================================================

PRIORITY_PATH_HINTS = [
    "lib/main.dart",
    "README.md",
    "pubspec.yaml",
    "pubspec.yml",
    "src/main",
    "src",
    "app",
]


# ============================================================
# ZIP EXTRACTION
# ============================================================

def extract_zip(
    zip_path: Path,
    extract_directory: Path
):

    print()
    print("=" * 60)
    print("PROJECT EXTRACTION")
    print("=" * 60)

    if extract_directory.exists():

        shutil.rmtree(
            extract_directory
        )

    extract_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"ZIP file      : {zip_path}"
    )

    print(
        f"Extracting to : {extract_directory}"
    )

    print()

    with zipfile.ZipFile(
        zip_path,
        "r"
    ) as archive:

        # ----------------------------------------------------
        # ZIP SECURITY CHECK
        # ----------------------------------------------------

        for member in archive.infolist():

            target_path = (
                extract_directory
                / member.filename
            ).resolve()

            try:

                target_path.relative_to(
                    extract_directory.resolve()
                )

            except ValueError:

                raise RuntimeError(
                    "Unsafe ZIP file detected."
                )

        archive.extractall(
            extract_directory
        )

    print(
        "Project extracted successfully."
    )

    return extract_directory


# ============================================================
# FIND SOURCE FILES
# ============================================================

def find_source_files(
    project_directory: Path
):

    project_directory = (
        Path(project_directory)
        .resolve()
    )

    files = []

    if not project_directory.exists():

        return files

    for path in project_directory.rglob("*"):

        if not path.is_file():
            continue

        try:

            relative_path = (
                path.relative_to(
                    project_directory
                )
            )

        except ValueError:

            continue

        relative_parts = (
            relative_path.parts[:-1]
        )

        # ----------------------------------------------------
        # Ignore generated/cache directories.
        # ----------------------------------------------------

        if any(
            part in IGNORED_DIRECTORIES
            for part in relative_parts
        ):

            continue

        # ----------------------------------------------------
        # Supported extensions.
        # ----------------------------------------------------

        if (
            path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        ):

            files.append(path)

    return sorted(files)


# ============================================================
# FIND PROJECT ROOT
# ============================================================

def find_project_root(
    directory: Path
):

    directory = (
        Path(directory)
        .resolve()
    )

    # --------------------------------------------------------
    # Check current directory.
    # --------------------------------------------------------

    current_files = (
        find_source_files(
            directory
        )
    )

    if current_files:

        return directory

    # --------------------------------------------------------
    # Check immediate child directories.
    # --------------------------------------------------------

    children = [
        child
        for child in directory.iterdir()
        if child.is_dir()
        and child.name not in
        IGNORED_DIRECTORIES
    ]

    best_directory = None
    best_count = 0

    for child in children:

        child_files = (
            find_source_files(
                child
            )
        )

        if len(child_files) > best_count:

            best_directory = child
            best_count = len(
                child_files
            )

    if best_directory is not None:

        return best_directory

    return directory


# ============================================================
# PRIORITY SORTING
# ============================================================

def priority_score(
    file_path: Path,
    project_root: Path
):

    relative = (
        file_path
        .relative_to(project_root)
    )

    relative_string = (
        str(relative)
        .replace("\\", "/")
        .lower()
    )

    filename = (
        file_path.name.lower()
    )

    score = 0

    # --------------------------------------------------------
    # Highest priority: exact important files.
    # --------------------------------------------------------

    if filename in PRIORITY_FILE_NAMES:

        score += 100

    # --------------------------------------------------------
    # Highest priority: SonicSplit-style main files.
    # --------------------------------------------------------

    if relative_string == "lib/main.dart":

        score += 300

    if relative_string == "pubspec.yaml":

        score += 250

    if relative_string == "readme.md":

        score += 250

    # --------------------------------------------------------
    # Application source directories.
    # --------------------------------------------------------

    if "/lib/" in (
        "/" + relative_string
    ):

        score += 100

    if "/src/" in (
        "/" + relative_string
    ):

        score += 80

    if "/app/" in (
        "/" + relative_string
    ):

        score += 60

    # --------------------------------------------------------
    # Tests.
    # --------------------------------------------------------

    if (
        "/test/" in
        ("/" + relative_string)
        or filename.endswith("_test.dart")
    ):

        score += 30

    # --------------------------------------------------------
    # Platform generated/build files get lower priority.
    # --------------------------------------------------------

    platform_dirs = [
        "/android/",
        "/ios/",
        "/linux/",
        "/macos/",
        "/windows/",
    ]

    for platform in platform_dirs:

        if platform in (
            "/" + relative_string
        ):

            score -= 20

    if (
        "generated_plugin" in
        relative_string
    ):

        score -= 50

    if (
        "cmakelists.txt"
        in filename
    ):

        score -= 30

    return score


def sort_priority_files(
    files,
    project_root
):

    return sorted(
        files,
        key=lambda path:
        (
            -priority_score(
                path,
                project_root
            ),
            str(path).lower()
        )
    )


# ============================================================
# READ PROJECT FILES
# ============================================================

def read_project_files(
    project_directory: Path,
    source_files
):

    print()
    print("=" * 60)
    print("READING PROJECT SOURCE FILES")
    print("=" * 60)

    file_data = []

    for index, file_path in enumerate(
        source_files,
        start=1
    ):

        try:

            relative_path = (
                file_path.relative_to(
                    project_directory
                )
            )

            source_code = (
                file_path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                )
            )

            if not source_code.strip():

                continue

            # ------------------------------------------------
            # Avoid enormous files.
            # ------------------------------------------------

            if len(source_code) > 100000:

                print(
                    f"[{index:03}] "
                    f"{relative_path} "
                    "(skipped: very large)"
                )

                continue

            file_data.append(
                {
                    "path": str(
                        relative_path
                    ),
                    "name": file_path.name,
                    "content": source_code,
                }
            )

            print(
                f"[{index:03}] "
                f"{relative_path} "
                f"({len(source_code)} chars)"
            )

        except Exception as exc:

            print(
                f"WARNING: Could not read "
                f"{file_path}: {exc}"
            )

    print()

    print(
        f"Readable source files: "
        f"{len(file_data)}"
    )

    return file_data


# ============================================================
# BUILD COMBINED PROJECT SOURCE
# ============================================================

def build_combined_source(
    file_data
):

    sections = []

    for item in file_data:

        sections.append(
            "\n"
            + "=" * 70
            + "\n"
            + f"FILE: {item['path']}\n"
            + "=" * 70
            + "\n"
            + item["content"]
            + "\n"
        )

    return "\n".join(
        sections
    )


# ============================================================
# BUILD PRIORITY EVIDENCE
# ============================================================

def build_priority_evidence(
    file_data
):

    priority_names = {
        "readme.md",
        "pubspec.yaml",
        "pubspec.yml",
        "main.dart",
        "requirements.txt",
        "package.json",
    }

    priority_items = []

    for item in file_data:

        name = (
            item["name"]
            .lower()
        )

        path = (
            item["path"]
            .replace("\\", "/")
            .lower()
        )

        is_priority = (
            name in priority_names
            or path == "lib/main.dart"
            or path == "pubspec.yaml"
            or path == "readme.md"
        )

        if is_priority:

            priority_items.append(
                item
            )

    sections = []

    for item in priority_items:

        content = item[
            "content"
        ]

        # Keep individual evidence manageable.
        if len(content) > 7000:

            content = (
                content[:7000]
                + "\n[priority file truncated]"
            )

        sections.append(
            "\n"
            + "=" * 70
            + "\n"
            + "PRIORITY PROJECT FILE: "
            + item["path"]
            + "\n"
            + "=" * 70
            + "\n"
            + content
        )

    return "\n".join(
        sections
    )


# ============================================================
# SCANNER
# ============================================================

def run_scanner(
    combined_source,
    project_name
):

    print()
    print("=" * 60)
    print("[1/5] RUNNING SCANNER AGENT")
    print("=" * 60)

    scanner = ScannerAgent()

    scanner_data = (
        scanner.scan_content(
            combined_source,
            project_name
        )
    )

    print(
        "Scanner completed."
    )

    print(
        f"Characters       : "
        f"{scanner_data.get('character_count', 0)}"
    )

    print(
        f"Lines            : "
        f"{scanner_data.get('line_count', 0)}"
    )

    print(
        f"Non-empty lines  : "
        f"{scanner_data.get('non_empty_lines', 0)}"
    )

    return scanner_data


# ============================================================
# SEMANTIC CHUNKING
# ============================================================

def create_chunks(
    combined_source
):

    print()
    print("=" * 60)
    print("[2/5] CREATING SEMANTIC CHUNKS")
    print("=" * 60)

    chunker = SemanticChunker(
        chunk_size=6000,
        chunk_overlap=800
    )

    chunks = chunker.split(
        combined_source
    )

    chunks = chunker.add_context(
        chunks
    )

    print(
        f"Created {len(chunks)} "
        "semantic chunks."
    )

    if not chunks:

        raise RuntimeError(
            "No semantic chunks were created."
        )

    return chunks


# ============================================================
# RAG
# ============================================================

def build_rag(
    chunks
):

    print()
    print("=" * 60)
    print("[3/5] BUILDING RAG VECTOR STORE")
    print("=" * 60)

    vector_store = CodeVectorStore(
        persist_directory=str(
            BASE_DIR
            / "output"
            / "chroma_db"
        ),
        model_name="all-MiniLM-L6-v2"
    )

    vector_store.add_chunks(
        chunks
    )

    # --------------------------------------------------------
    # Project-aware retrieval queries.
    # --------------------------------------------------------

    queries = [

        # Project identity
        "project purpose application overview README",
        "main application functionality user workflow",

        # Flutter
        "Flutter application main.dart UI screens widgets",
        "Dart application user interface state management",

        # Audio
        "audio processing audio splitting source separation",
        "music vocals instruments stems audio playback",
        "audio file upload processing output",

        # Dependencies
        "pubspec.yaml dependencies Flutter packages",
        "audio packages just_audio audio_session",

        # Architecture
        "application architecture components modules",
        "data flow processing workflow",

        # Platform
        "Android iOS Windows Linux macOS Web platform support",

        # Testing
        "Flutter widget tests application testing",

        # Configuration
        "configuration assets permissions platform integration",
    ]

    relevant_chunks = []

    for query in queries:

        try:

            results = (
                vector_store.search(
                    query,
                    top_k=1
                )
            )

            for result in results:

                if result not in (
                    relevant_chunks
                ):

                    relevant_chunks.append(
                        result
                    )

        except Exception as exc:

            print(
                f"WARNING: RAG query failed: "
                f"{query}"
            )

            print(exc)

    # --------------------------------------------------------
    # Limit Architect context.
    # --------------------------------------------------------

    MAX_RAG_CHUNKS = 6

    relevant_chunks = (
        relevant_chunks[
            :MAX_RAG_CHUNKS
        ]
    )

    print(
        f"Retrieved "
        f"{len(relevant_chunks)} "
        "relevant chunks using RAG."
    )

    print(
        f"Architect context limited "
        f"to {MAX_RAG_CHUNKS} chunks."
    )

    return relevant_chunks


# ============================================================
# ARCHITECT
# ============================================================

def run_architect(
    project_name,
    scanner_data,
    relevant_chunks,
    priority_evidence
):

    print()
    print("=" * 60)
    print("[4/5] RUNNING ARCHITECT AGENT")
    print("=" * 60)

    architect = ArchitectAgent()

    # --------------------------------------------------------
    # Put priority project files before RAG results.
    # --------------------------------------------------------

    combined_evidence = []

    if priority_evidence.strip():

        combined_evidence.append(
            priority_evidence
        )

    combined_evidence.extend(
        relevant_chunks
    )

    # --------------------------------------------------------
    # Deduplicate.
    # --------------------------------------------------------

    final_evidence = []

    for item in combined_evidence:

        if item not in final_evidence:

            final_evidence.append(
                item
            )

    # --------------------------------------------------------
    # Final context limit.
    # architect.py performs another safety limit.
    # --------------------------------------------------------

    final_evidence = (
        final_evidence[:8]
    )

    blueprint = (
        architect.build_design_blueprint(
            filename=project_name,
            scanner_data=scanner_data,
            relevant_chunks=final_evidence
        )
    )

    print(
        "Architect analysis completed."
    )

    architecture = blueprint.get(
        "architecture",
        {}
    )

    components = architecture.get(
        "components",
        []
    )

    requirements = blueprint.get(
        "functional_requirements",
        []
    )

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
# MANAGER
# ============================================================

def generate_tdd(
    blueprint
):

    print()
    print("=" * 60)
    print("[5/5] RUNNING MANAGER AGENT")
    print("=" * 60)

    manager = ManagerAgent(
        output_directory=str(
            BASE_DIR
            / "output"
        )
    )

    tdd_document = (
        manager.generate_tdd(
            blueprint
        )
    )

    output_file = (
        manager.save_tdd(
            tdd_document,
            filename=(
                "technical_design_document.md"
            )
        )
    )

    print(
        "Manager completed."
    )

    return output_file


# ============================================================
# PROJECT PIPELINE
# ============================================================

def run_project_pipeline(
    project_directory: Path
):

    project_directory = (
        Path(project_directory)
        .resolve()
    )

    print()
    print("=" * 60)
    print("STARTING TDD PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Detect project root.
    # --------------------------------------------------------

    project_directory = (
        find_project_root(
            project_directory
        )
    )

    print()

    print(
        f"Project root: "
        f"{project_directory}"
    )

    project_name = (
        project_directory.name
    )

    # --------------------------------------------------------
    # Find files.
    # --------------------------------------------------------

    source_files = (
        find_source_files(
            project_directory
        )
    )

    print(
        f"Supported source files: "
        f"{len(source_files)}"
    )

    if not source_files:

        raise RuntimeError(
            "No supported source files "
            "found inside the project."
        )

    # --------------------------------------------------------
    # Sort files by importance.
    # --------------------------------------------------------

    source_files = (
        sort_priority_files(
            source_files,
            project_directory
        )
    )

    # --------------------------------------------------------
    # Read files.
    # --------------------------------------------------------

    file_data = (
        read_project_files(
            project_directory,
            source_files
        )
    )

    if not file_data:

        raise RuntimeError(
            "No readable project files found."
        )

    # --------------------------------------------------------
    # Combined source.
    # --------------------------------------------------------

    combined_source = (
        build_combined_source(
            file_data
        )
    )

    # --------------------------------------------------------
    # Priority evidence.
    # --------------------------------------------------------

    priority_evidence = (
        build_priority_evidence(
            file_data
        )
    )

    # --------------------------------------------------------
    # Scanner.
    # --------------------------------------------------------

    scanner_data = run_scanner(
        combined_source,
        project_name
    )

    # --------------------------------------------------------
    # Semantic chunks.
    # --------------------------------------------------------

    chunks = create_chunks(
        combined_source
    )

    # --------------------------------------------------------
    # RAG.
    # --------------------------------------------------------

    relevant_chunks = build_rag(
        chunks
    )

    # --------------------------------------------------------
    # Architect.
    # --------------------------------------------------------

    blueprint = run_architect(
        project_name,
        scanner_data,
        relevant_chunks,
        priority_evidence
    )

    # --------------------------------------------------------
    # Add project metadata.
    # --------------------------------------------------------

    blueprint[
        "project_name"
    ] = project_name

    blueprint[
        "project_files"
    ] = [
        item["path"]
        for item in file_data
    ]

    blueprint[
        "project_file_count"
    ] = len(file_data)

    # Keep the complete readable source corpus outside the LLM context.
    # The Architect still receives bounded RAG/priority evidence, while the
    # Manager can build a complete source appendix from the actual files.
    blueprint[
        "complete_source_files"
    ] = [
        {
            "path": item["path"],
            "content": item["content"],
        }
        for item in file_data
    ]

    # --------------------------------------------------------
    # Manager.
    # --------------------------------------------------------

    output_file = (
        generate_tdd(
            blueprint
        )
    )

    return {
        "output_file": output_file,

        "file_count": len(
            file_data
        ),

        "chunk_count": len(
            chunks
        ),

        "rag_count": len(
            relevant_chunks
        ),

        "scanner_data": scanner_data,
    }


# ============================================================
# ZIP PIPELINE
# ============================================================

def run_zip_pipeline(
    zip_path
):

    zip_path = (
        Path(zip_path)
        .resolve()
    )

    if not zip_path.exists():

        raise FileNotFoundError(
            f"ZIP file not found: "
            f"{zip_path}"
        )

    if (
        zip_path.suffix.lower()
        != ".zip"
    ):

        raise ValueError(
            "Input file must be a ZIP file."
        )

    extract_directory = (
        BASE_DIR
        / "uploaded_project"
    )

    extracted = extract_zip(
        zip_path,
        extract_directory
    )

    project_directory = (
        find_project_root(
            extracted
        )
    )

    print()
    print(
        "Actual project directory:"
    )

    print(
        project_directory
    )

    return run_project_pipeline(
        project_directory
    )


# ============================================================
# WEB COMPATIBILITY FUNCTION
# ============================================================

def run_pipeline(
    source_file=None
):

    if source_file is None:

        source_file = (
            BASE_DIR
            / "sample.py"
        )

    source_path = (
        Path(source_file)
        .resolve()
    )

    if not source_path.exists():

        raise FileNotFoundError(
            f"Input not found: "
            f"{source_path}"
        )

    # --------------------------------------------------------
    # ZIP
    # --------------------------------------------------------

    if (
        source_path.is_file()
        and source_path.suffix.lower()
        == ".zip"
    ):

        result = run_zip_pipeline(
            source_path
        )

        return result[
            "output_file"
        ]

    # --------------------------------------------------------
    # PROJECT DIRECTORY
    # --------------------------------------------------------

    if source_path.is_dir():

        result = (
            run_project_pipeline(
                source_path
            )
        )

        return result[
            "output_file"
        ]

    # --------------------------------------------------------
    # SINGLE SOURCE FILE
    # --------------------------------------------------------

    if source_path.is_file():

        result = (
            run_project_pipeline(
                source_path.parent
            )
        )

        return result[
            "output_file"
        ]

    raise ValueError(
        "Unsupported input type."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "       AGENTIC TDD GENERATION PIPELINE"
    )
    print("=" * 60)

    if len(sys.argv) < 2:

        print()
        print("Usage:")
        print()

        print(
            "ZIP project:"
        )

        print(
            "python pipeline.py project.zip"
        )

        print()

        print(
            "Project folder:"
        )

        print(
            "python pipeline.py project_folder"
        )

        print()

        print(
            "Single source file:"
        )

        print(
            "python pipeline.py sample.py"
        )

        print()

        return

    input_path = (
        Path(sys.argv[1])
        .resolve()
    )

    if not input_path.exists():

        print(
            f"ERROR: Input not found: "
            f"{input_path}"
        )

        return

    try:

        # ----------------------------------------------------
        # ZIP
        # ----------------------------------------------------

        if (
            input_path.is_file()
            and input_path.suffix.lower()
            == ".zip"
        ):

            result = run_zip_pipeline(
                input_path
            )

        # ----------------------------------------------------
        # DIRECTORY
        # ----------------------------------------------------

        elif input_path.is_dir():

            result = (
                run_project_pipeline(
                    input_path
                )
            )

        # ----------------------------------------------------
        # SINGLE FILE
        # ----------------------------------------------------

        elif input_path.is_file():

            output_file = run_pipeline(
                input_path
            )

            result = {
                "output_file": output_file,
                "file_count": 1,
                "chunk_count": 0,
                "rag_count": 0,
            }

        else:

            print(
                "ERROR: Unsupported input."
            )

            return

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print(
            "       TDD GENERATION COMPLETED"
        )
        print("=" * 60)

        print()

        print(
            f"Project files analyzed : "
            f"{result['file_count']}"
        )

        print(
            f"Semantic chunks        : "
            f"{result['chunk_count']}"
        )

        print(
            f"RAG chunks used        : "
            f"{result['rag_count']}"
        )

        print()

        print(
            "Final TDD saved to:"
        )

        print(
            result["output_file"]
        )

        print()

        print(
            "Pipeline completed successfully!"
        )

        print("=" * 60)

    except Exception as exc:

        print()
        print("=" * 60)
        print(
            "PIPELINE ERROR"
        )
        print("=" * 60)

        print()

        print(
            type(exc).__name__
        )

        print(
            exc
        )

        print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()