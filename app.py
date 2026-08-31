from flask import Flask, render_template, request, send_file
from pathlib import Path
import zipfile
import shutil

from pipeline import run_pipeline


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder="web/templates",
    static_folder="web/static"
)


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
PROJECT_DIR = BASE_DIR / "uploaded_project"
OUTPUT_DIR = BASE_DIR / "output"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SUPPORTED SOURCE FILES
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".cpp",
    ".c",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".kt",
    ".kts",
    ".swift",
    ".dart",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sql",
    ".yaml",
    ".yml",
    ".json",
    ".xml",
}


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():
    return render_template("index.html")


# ============================================================
# FIND PROJECT ROOT
# ============================================================

def find_project_root(extracted_dir: Path) -> Path:
    """
    Detect the actual project root inside the extracted ZIP.

    Example:

        uploaded_project/
            SonicSplit-AI-main/
                lib/
                android/
                ios/
                pubspec.yaml

    In this case SonicSplit-AI-main becomes the project root.
    """

    # First check whether the extracted directory itself
    # contains source files.
    direct_files = [
        p for p in extracted_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if direct_files:
        return extracted_dir

    # Look for a single top-level directory.
    directories = [
        p for p in extracted_dir.iterdir()
        if p.is_dir()
        and p.name not in {
            "__MACOSX",
            ".git",
            "__pycache__"
        }
    ]

    if len(directories) == 1:
        return directories[0]

    # Otherwise use the extracted directory.
    return extracted_dir


# ============================================================
# COLLECT PROJECT FILES
# ============================================================

def collect_source_files(project_root: Path):
    """
    Collect all supported source/configuration files
    from the uploaded project.
    """

    files = []

    for path in project_root.rglob("*"):

        if not path.is_file():
            continue

        # Ignore generated/cache folders.
        parts_lower = {
            part.lower()
            for part in path.parts
        }

        ignored_directories = {
            ".git",
            "__pycache__",
            ".dart_tool",
            "build",
            ".gradle",
            "node_modules",
            ".idea",
            ".vscode",
        }

        if parts_lower.intersection(ignored_directories):
            continue

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)


# ============================================================
# GENERATE TDD
# ============================================================

@app.route("/generate", methods=["POST"])
def generate():

    print()
    print("=" * 60)
    print("                 PROJECT UPLOAD")
    print("=" * 60)

    # --------------------------------------------------------
    # Check uploaded file
    # --------------------------------------------------------

    if "project" not in request.files:
        return "No project file uploaded.", 400

    uploaded_file = request.files["project"]

    if uploaded_file.filename == "":
        return "No file selected.", 400

    filename = uploaded_file.filename

    if not filename.lower().endswith(".zip"):
        return "Please upload a ZIP file.", 400

    # --------------------------------------------------------
    # Clean previous project
    # --------------------------------------------------------

    if PROJECT_DIR.exists():
        try:
            shutil.rmtree(PROJECT_DIR)
        except Exception as exc:
            print("Could not remove old project:", exc)
            return "Could not clean previous uploaded project.", 500

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Clean old ZIP files
    # --------------------------------------------------------

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    for old_file in UPLOAD_DIR.iterdir():
        if old_file.is_file():
            try:
                old_file.unlink()
            except Exception:
                pass

    # --------------------------------------------------------
    # Save uploaded ZIP
    # --------------------------------------------------------

    zip_path = UPLOAD_DIR / Path(filename).name

    try:
        uploaded_file.save(zip_path)
    except Exception as exc:
        print("Upload save error:", exc)
        return "Could not save uploaded ZIP file.", 500

    print("ZIP file:", zip_path)

    # --------------------------------------------------------
    # Extract ZIP
    # --------------------------------------------------------

    try:

        with zipfile.ZipFile(zip_path, "r") as zip_ref:

            # Basic ZIP path safety check.
            for member in zip_ref.infolist():

                member_path = PROJECT_DIR / member.filename

                try:
                    member_path.resolve().relative_to(
                        PROJECT_DIR.resolve()
                    )
                except ValueError:
                    return "Unsafe ZIP file detected.", 400

            zip_ref.extractall(PROJECT_DIR)

    except zipfile.BadZipFile:
        return "Invalid ZIP file.", 400

    except Exception as exc:
        print("ZIP extraction error:", exc)
        return f"Could not extract ZIP: {exc}", 500

    # --------------------------------------------------------
    # Detect actual project root
    # --------------------------------------------------------

    project_root = find_project_root(PROJECT_DIR)

    print("Project root:", project_root)

    # --------------------------------------------------------
    # Collect all supported files
    # --------------------------------------------------------

    source_files = collect_source_files(project_root)

    print("Supported source files:", len(source_files))

    if not source_files:
        return (
            "No supported source-code files found inside the project."
        ), 400

    # --------------------------------------------------------
    # Display discovered files
    # --------------------------------------------------------

    print()
    print("PROJECT FILES")
    print("-" * 60)

    for index, source_file in enumerate(source_files, start=1):

        try:
            relative_path = source_file.relative_to(project_root)
        except ValueError:
            relative_path = source_file

        print(
            f"[{index:03d}] {relative_path}"
        )

    print("-" * 60)
    print(
        f"Total supported files: {len(source_files)}"
    )
    print()

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # Pass the PROJECT ROOT to the pipeline.
    #
    # Do NOT pass source_files[0].
    #
    # The pipeline is designed to analyze the project context.
    # --------------------------------------------------------

    try:

        print("=" * 60)
        print("              STARTING TDD PIPELINE")
        print("=" * 60)
        print()

        output_file = run_pipeline(project_root)

    except Exception as exc:

        print()
        print("=" * 60)
        print("                 PIPELINE ERROR")
        print("=" * 60)
        print(exc)
        print()

        return f"Pipeline failed: {exc}", 500

    # --------------------------------------------------------
    # Locate generated TDD
    # --------------------------------------------------------

    if output_file is None:
        return (
            "Pipeline completed but did not return an output file."
        ), 500

    output_path = Path(output_file)

    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    # --------------------------------------------------------
    # Fallback to standard output location
    # --------------------------------------------------------

    if not output_path.exists():

        fallback_file = (
            OUTPUT_DIR /
            "technical_design_document.md"
        )

        if fallback_file.exists():
            output_path = fallback_file

    # --------------------------------------------------------
    # Final check
    # --------------------------------------------------------

    if not output_path.exists():

        return (
            "TDD generation completed, "
            "but the generated document was not found."
        ), 500

    # --------------------------------------------------------
    # Send TDD to browser
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("             TDD GENERATION COMPLETED")
    print("=" * 60)

    print("TDD file:", output_path)
    print()

    return send_file(
        output_path,
        as_attachment=True,
        download_name="technical_design_document.md",
        mimetype="text/markdown"
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )