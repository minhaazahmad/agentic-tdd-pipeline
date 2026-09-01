"""Deterministic source scanner used before any LLM call.

The scanner extracts factual project evidence: file inventory, language,
imports, classes/functions, endpoints, environment variables, tests and
source statistics. It never asks an LLM to discover these facts.
"""
from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any


class ScannerAgent:
    """Build a deterministic evidence inventory from project source."""

    def __init__(self, max_file_chars: int = 200_000) -> None:
        self.max_file_chars = max_file_chars

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    @staticmethod
    def _language(path: str) -> str:
        ext = Path(path).suffix.lower()
        mapping = {
            ".py": "Python", ".dart": "Dart", ".js": "JavaScript",
            ".jsx": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
            ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin",
            ".cpp": "C++", ".cc": "C++", ".cxx": "C++", ".c": "C",
            ".h": "C/C++ Header", ".hpp": "C/C++ Header", ".cs": "C#",
            ".go": "Go", ".rs": "Rust", ".php": "PHP", ".rb": "Ruby",
            ".swift": "Swift", ".html": "HTML", ".css": "CSS",
            ".scss": "SCSS", ".sql": "SQL", ".json": "JSON",
            ".yaml": "YAML", ".yml": "YAML", ".xml": "XML",
            ".md": "Markdown", ".txt": "Text",
        }
        return mapping.get(ext, "Unknown")

    @staticmethod
    def _python_symbols(source: str) -> dict[str, list[str]]:
        result = {"classes": [], "functions": [], "imports": []}
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return result
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                result["classes"].append(node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result["functions"].append(node.name)
            elif isinstance(node, ast.Import):
                result["imports"].extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                result["imports"].append(module)
        for key in result:
            result[key] = sorted(set(result[key]))
        return result

    @staticmethod
    def _generic_symbols(source: str) -> dict[str, list[str]]:
        classes = set(re.findall(
            r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)", source
        ))
        functions = set(re.findall(
            r"\b(?:def|function|func)\s+([A-Za-z_][A-Za-z0-9_]*)", source
        ))
        imports = set(re.findall(
            r"(?m)^\s*(?:import|from|require|using|include)\s+([^\s;]+)", source
        ))
        return {
            "classes": sorted(classes),
            "functions": sorted(functions),
            "imports": sorted(imports),
        }

    def scan_files(self, project_root: Path) -> dict[str, Any]:
        project_root = Path(project_root).resolve()
        records: list[dict[str, Any]] = []
        language_counts = Counter()
        skipped: list[dict[str, str]] = []

        for path in sorted(project_root.rglob("*")):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(project_root)
            except ValueError:
                continue

            if any(part in {
                ".git", ".venv", "venv", "env", "__pycache__", "node_modules",
                ".idea", ".vscode", "build", "dist", "target", ".dart_tool",
                "Pods", "bin", "obj", ".gradle", ".cache", "coverage",
                "output", "uploads", "uploaded_project",
            } for part in rel.parts[:-1]):
                continue

            language = self._language(str(rel))
            if language == "Unknown":
                continue

            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                skipped.append({"path": str(rel).replace("\\", "/"), "reason": str(exc)})
                continue

            if not source.strip():
                continue
            if len(source) > self.max_file_chars:
                skipped.append({
                    "path": str(rel).replace("\\", "/"),
                    "reason": f"file exceeds {self.max_file_chars} characters",
                })
                continue

            rel_path = str(rel).replace("\\", "/")
            symbols = (
                self._python_symbols(source)
                if path.suffix.lower() == ".py"
                else self._generic_symbols(source)
            )
            line_count = len(source.splitlines())
            language_counts[language] += 1

            records.append({
                "path": rel_path,
                "language": language,
                "size_chars": len(source),
                "line_count": line_count,
                "sha256": self._sha256(source),
                "classes": symbols["classes"],
                "functions": symbols["functions"],
                "imports": symbols["imports"],
                "is_test": (
                    "/test/" in f"/{rel_path.lower()}"
                    or Path(rel_path).name.lower().startswith("test_")
                    or Path(rel_path).name.lower().endswith(("_test.py", "_test.dart"))
                ),
                "content": source,
            })

        combined_chars = sum(x["size_chars"] for x in records)
        combined_lines = sum(x["line_count"] for x in records)

        return {
            "project_root": str(project_root),
            "project_name": project_root.name,
            "file_count": len(records),
            "character_count": combined_chars,
            "line_count": combined_lines,
            "non_empty_lines": sum(
                sum(1 for line in x["content"].splitlines() if line.strip())
                for x in records
            ),
            "language_counts": dict(language_counts),
            "test_file_count": sum(1 for x in records if x["is_test"]),
            "files": records,
            "skipped_files": skipped,
        }

    def scan_content(self, combined_source: str, project_name: str = "project") -> dict[str, Any]:
        """Compatibility API for the original pipeline."""
        lines = combined_source.splitlines()
        return {
            "project_name": project_name,
            "character_count": len(combined_source),
            "line_count": len(lines),
            "non_empty_lines": sum(1 for line in lines if line.strip()),
        }
