from pathlib import Path
import tempfile
import zipfile

from agents.scanner import ScannerAgent
from parser.chunker import SemanticChunker
from rag.vector_store import CodeVectorStore
from pipeline import safe_extract_zip


def test_scanner_extracts_python_symbols(tmp_path):
    src = tmp_path / "main.py"
    src.write_text(
        "import os\nclass Demo:\n    def run(self):\n        return 1\n",
        encoding="utf-8",
    )
    data = ScannerAgent().scan_files(tmp_path)
    assert data["file_count"] == 1
    record = data["files"][0]
    assert "Demo" in record["classes"]
    assert "run" in record["functions"]
    assert "os" in record["imports"]


def test_chunker_preserves_file_and_line_metadata():
    chunker = SemanticChunker(chunk_size=40, chunk_overlap=5)
    chunks = chunker.split_files([{
        "path": "src/main.py",
        "language": "Python",
        "content": "\n".join(f"line {i}" for i in range(1, 20)),
    }])
    assert chunks
    assert all(x["path"] == "src/main.py" for x in chunks)
    assert all(x["start_line"] <= x["end_line"] for x in chunks)
    assert all(x["id"].startswith("EV-") for x in chunks)


def test_retrieval_returns_metadata():
    store = CodeVectorStore(mode="lexical")
    store.add_chunks([
        {"id": "EV-API", "text": "[FILE: api.py]\nPOST /generate request handler"},
        {"id": "EV-DB", "text": "[FILE: db.py]\nSQLite repository"},
    ])
    result = store.search_with_metadata("API generate request", top_k=1)
    assert result
    assert result[0]["id"] == "EV-API"


def test_zip_slip_is_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../escape.txt", "bad")

    destination = tmp_path / "extract"
    try:
        safe_extract_zip(archive, destination)
    except ValueError:
        pass
    else:
        raise AssertionError("ZIP traversal should be rejected")
