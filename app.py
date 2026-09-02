from flask import Flask, jsonify, render_template, request, send_file
from pathlib import Path
from werkzeug.utils import secure_filename
import json
import shutil
import tempfile
import time
import zipfile

from pipeline import run_pipeline, safe_extract_zip

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
HISTORY_DIR = OUTPUT_DIR / "history"
HISTORY_FILE = OUTPUT_DIR / "history.json"

for folder in (UPLOAD_DIR, OUTPUT_DIR, HISTORY_DIR):
    folder.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {
    ".py", ".dart", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".kts",
    ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".cs", ".go", ".rs", ".php",
    ".rb", ".swift", ".html", ".css", ".scss", ".sql", ".json", ".yaml",
    ".yml", ".xml", ".md", ".txt",
}


def load_history():
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(items):
    temp = HISTORY_FILE.with_suffix(".tmp")
    temp.write_text(
        json.dumps(items[:50], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(HISTORY_FILE)


def add_history_entry(name, mode, output_file):
    safe_base = secure_filename(Path(name or "tdd").stem) or "tdd"
    filename = f"{int(time.time() * 1000)}_{safe_base}_technical_design_document.md"
    destination = HISTORY_DIR / filename
    shutil.copy2(output_file, destination)

    entry = {
        "id": str(int(time.time() * 1000)),
        "name": name or "Unknown project",
        "mode": mode,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "download_url": f"/history/download/{filename}",
    }
    items = load_history()
    items.insert(0, entry)
    save_history(items)
    return entry


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "Agentic TDD Generator",
        "version": "2.0-evidence-gated",
    })


@app.route("/api/history")
def api_history():
    return jsonify(load_history())


@app.route("/history/download/<path:filename>")
def history_download(filename):
    safe_name = Path(filename).name
    path = HISTORY_DIR / safe_name
    if not path.is_file():
        return jsonify({"error": "History document not found."}), 404
    return send_file(path, as_attachment=True, download_name="technical_design_document.md")


@app.route("/download/latest")
def download_latest():
    path = OUTPUT_DIR / "technical_design_document.md"
    if not path.is_file():
        return jsonify({"error": "No generated TDD available."}), 404
    return send_file(path, as_attachment=True, download_name="technical_design_document.md")


@app.route("/generate", methods=["POST"])
def generate():
    mode = request.form.get("mode", "zip")
    temp_root = None

    try:
        if mode == "zip":
            file = request.files.get("project")
            if not file or not file.filename:
                return jsonify({"error": "No project ZIP uploaded."}), 400
            if Path(file.filename).suffix.lower() != ".zip":
                return jsonify({"error": "Please upload a ZIP file."}), 400

            temp_root = Path(tempfile.mkdtemp(prefix="agentic_tdd_web_"))
            zip_path = temp_root / (secure_filename(file.filename) or "project.zip")
            file.save(zip_path)

            project_dir = temp_root / "project"
            safe_extract_zip(zip_path, project_dir)
            result = run_pipeline(project_dir)
            history = add_history_entry(project_dir.name, "Project ZIP", result["output_file"])

        elif mode in {"file", "code"}:
            if mode == "file":
                file = request.files.get("source")
                if not file or not file.filename:
                    return jsonify({"error": "No source file uploaded."}), 400
                filename = secure_filename(file.filename) or "source.py"
                extension = Path(filename).suffix.lower()
                if extension not in ALLOWED_EXTENSIONS:
                    return jsonify({"error": f"Unsupported source type: {extension or 'unknown'}"}), 400
                content = file.read()
                if len(content) > 10 * 1024 * 1024:
                    return jsonify({"error": "Source file is too large."}), 413
            else:
                filename = secure_filename(request.form.get("filename", "pasted_code.py")) or "pasted_code.py"
                if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
                    filename = "pasted_code.py"
                content = request.form.get("code", "").encode("utf-8")
                if not content.strip():
                    return jsonify({"error": "Please paste source code first."}), 400
                if len(content) > 5 * 1024 * 1024:
                    return jsonify({"error": "Pasted source is too large."}), 413

            temp_root = Path(tempfile.mkdtemp(prefix="agentic_tdd_source_"))
            source_path = temp_root / filename
            source_path.write_bytes(content)
            result = run_pipeline(source_path)
            history = add_history_entry(filename, "Single File" if mode == "file" else "Paste Code", result["output_file"])

        else:
            return jsonify({"error": "Unsupported input mode."}), 400

        return jsonify({
            "success": True,
            "message": "TDD generated successfully.",
            "download_url": "/download/latest",
            "quality": result.get("quality", {}),
            "stats": {
                "files": result["file_count"],
                "chunks": result["chunk_count"],
                "evidence": result["rag_count"],
            },
            "history": history,
        })

    except zipfile.BadZipFile:
        return jsonify({"error": "Invalid or corrupted ZIP file."}), 400
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        app.logger.exception("Pipeline request failed")
        return jsonify({"error": f"Pipeline failed: {type(exc).__name__}: {exc}"}), 500
    finally:
        if temp_root:
            shutil.rmtree(temp_root, ignore_errors=True)


@app.route("/api/clear-history", methods=["POST"])
def clear_history():
    save_history([])
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
