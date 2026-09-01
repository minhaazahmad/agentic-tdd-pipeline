"""Source-aware chunking with stable evidence IDs and line ranges."""
from __future__ import annotations

import re
from typing import Any


class SemanticChunker:
    """Chunk each source file independently; never mix unrelated files."""

    def __init__(self, chunk_size: int = 4500, chunk_overlap: int = 500) -> None:
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_size must be > chunk_overlap >= 0")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _header(path: str, start: int, end: int, chunk_id: str) -> str:
        return (
            f"[EVIDENCE_ID: {chunk_id}]\n"
            f"[FILE: {path}]\n"
            f"[LINES: {start}-{end}]\n"
        )

    def split_files(self, file_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for record in file_records:
            path = str(record["path"])
            lines = str(record["content"]).splitlines()
            if not lines:
                continue

            start_idx = 0
            chunk_index = 0
            while start_idx < len(lines):
                char_count = 0
                end_idx = start_idx
                while end_idx < len(lines):
                    next_len = len(lines[end_idx]) + 1
                    if end_idx > start_idx and char_count + next_len > self.chunk_size:
                        break
                    char_count += next_len
                    end_idx += 1

                content = "\n".join(lines[start_idx:end_idx]).strip()
                if content:
                    start_line = start_idx + 1
                    end_line = end_idx
                    stable = f"{path}:{start_line}:{end_line}:{chunk_index}"
                    chunk_id = "EV-" + re.sub(
                        r"[^A-Za-z0-9]+", "-", stable
                    ).strip("-").upper()

                    text = (
                        self._header(path, start_line, end_line, chunk_id)
                        + content
                    )
                    chunks.append({
                        "id": chunk_id,
                        "text": text,
                        "content": content,
                        "path": path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "chunk_index": chunk_index,
                        "language": record.get("language", "Unknown"),
                    })

                if end_idx >= len(lines):
                    break

                next_start = end_idx
                target = max(start_idx + 1, end_idx - self.chunk_overlap // 80)
                # Ensure forward progress while retaining a small line overlap.
                start_idx = min(max(target, next_start - 8), next_start - 1)
                chunk_index += 1

        return chunks

    def split(self, combined_source: str) -> list[str]:
        """Legacy string API; prefer split_files for accurate metadata."""
        lines = combined_source.splitlines()
        chunks = []
        start = 0
        while start < len(lines):
            size = 0
            end = start
            while end < len(lines):
                size += len(lines[end]) + 1
                if end > start and size > self.chunk_size:
                    break
                end += 1
            chunks.append("\n".join(lines[start:end]))
            if end >= len(lines):
                break
            start = max(start + 1, end - 8)
        return chunks

    def add_context(self, chunks: list[Any]) -> list[Any]:
        """Compatibility method; source-aware chunks already carry context."""
        return chunks
