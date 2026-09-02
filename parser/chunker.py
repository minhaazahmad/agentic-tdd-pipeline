"""
Semantic Map-Reduce chunking with deterministic evidence IDs.

Design:
    Source File
        |
        v
    MAP: semantic units
        |
        v
    Context Injection
        |
        v
    REDUCE: related units -> evidence groups
        |
        v
    Evidence chunks

Goals:
- Preserve file and line boundaries.
- Prefer classes/functions/configuration blocks over arbitrary character cuts.
- Keep deterministic IDs and hashes.
- Inject stable source context into every chunk.
- Keep split_files() compatible with the existing pipeline.
- Avoid mixing unrelated source files.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class SemanticUnit:
    """Smallest semantic unit produced during the MAP phase."""

    path: str
    language: str
    start_line: int
    end_line: int
    content: str
    kind: str
    symbol: str = ""

    @property
    def char_count(self) -> int:
        return len(self.content)


class SemanticChunker:
    """
    Source-aware Map-Reduce chunker.

    MAP:
        Convert each source file into semantic units.

    CONTEXT INJECTION:
        Add deterministic file, language, symbol and neighbouring-unit
        context to each evidence chunk.

    REDUCE:
        Combine compatible adjacent semantic units until the configured
        character budget is reached.
    """

    def __init__(
        self,
        chunk_size: int = 4500,
        chunk_overlap: int = 500,
        context_lines: int = 3,
        max_context_chars: int = 900,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        if context_lines < 0:
            raise ValueError("context_lines cannot be negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_lines = context_lines
        self.max_context_chars = max_context_chars

    # ------------------------------------------------------------------
    # Deterministic helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(
            text.encode("utf-8", errors="ignore")
        ).hexdigest()

    @staticmethod
    def _normalize_path(path: str) -> str:
        return str(path).replace("\\", "/")

    @staticmethod
    def _safe_id(value: str) -> str:
        value = re.sub(r"[^A-Za-z0-9]+", "-", value)
        value = value.strip("-")
        return value.upper() or "UNKNOWN"

    @classmethod
    def _evidence_id(
        cls,
        path: str,
        start_line: int,
        end_line: int,
        index: int,
    ) -> str:
        stable = (
            f"{cls._normalize_path(path)}:"
            f"{start_line}:"
            f"{end_line}:"
            f"{index}"
        )

        digest = hashlib.sha1(
            stable.encode("utf-8")
        ).hexdigest()[:10].upper()

        readable = cls._safe_id(
            f"{path}-{start_line}-{end_line}-{index}"
        )

        return f"EV-{readable}-{digest}"

    # ------------------------------------------------------------------
    # Semantic boundary detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_unit_start(
        line: str,
        language: str,
    ) -> tuple[str, str] | None:
        """
        Detect common semantic boundaries.

        This is deliberately deterministic. No LLM is involved.
        """

        stripped = line.strip()

        if not stripped:
            return None

        # Python
        if language == "Python":
            match = re.match(
                r"^(async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)",
                stripped,
            )
            if match:
                return "function", match.group(2)

            match = re.match(
                r"^class\s+([A-Za-z_][A-Za-z0-9_]*)",
                stripped,
            )
            if match:
                return "class", match.group(1)

        # Dart
        if language == "Dart":
            match = re.match(
                r"^(?:abstract\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)",
                stripped,
            )
            if match:
                return "class", match.group(1)

            match = re.match(
                r"^(?:Future<[^>]+>|[A-Za-z_][A-Za-z0-9_<>?]*)\s+"
                r"([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                stripped,
            )
            if match:
                return "function", match.group(1)

        # Java / Kotlin / C# / C++
        if language in {
            "Java",
            "Kotlin",
            "C#",
            "C++",
            "C",
        }:
            match = re.match(
                r"^(?:public\s+|private\s+|protected\s+|"
                r"internal\s+|static\s+|final\s+|abstract\s+)*"
                r"(?:class|interface|struct|enum)\s+"
                r"([A-Za-z_][A-Za-z0-9_]*)",
                stripped,
            )
            if match:
                return "class", match.group(1)

            match = re.match(
                r"^(?:public\s+|private\s+|protected\s+|"
                r"static\s+|inline\s+|virtual\s+|async\s+)*"
                r"[A-Za-z_][A-Za-z0-9_<>,\[\]? ]*\s+"
                r"([A-Za-z_][A-Za-z0-9_]*)\s*\(",
                stripped,
            )
            if match:
                return "function", match.group(1)

        # JavaScript / TypeScript
        if language in {"JavaScript", "TypeScript"}:
            match = re.match(
                r"^(?:export\s+)?(?:default\s+)?"
                r"(?:async\s+)?function\s+"
                r"([A-Za-z_][A-Za-z0-9_]*)",
                stripped,
            )
            if match:
                return "function", match.group(1)

            match = re.match(
                r"^(?:export\s+)?(?:default\s+)?class\s+"
                r"([A-Za-z_][A-Za-z0-9_]*)",
                stripped,
            )
            if match:
                return "class", match.group(1)

        # Go
        if language == "Go":
            match = re.match(
                r"^func\s+(?:\([^)]*\)\s*)?"
                r"([A-Za-z_][A-Za-z0-9_]*)",
                stripped,
            )
            if match:
                return "function", match.group(1)

            match = re.match(
                r"^type\s+([A-Za-z_][A-Za-z0-9_]*)\s+struct",
                stripped,
            )
            if match:
                return "class", match.group(1)

        # Rust
        if language == "Rust":
            match = re.match(
                r"^(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)",
                stripped,
            )
            if match:
                return "function", match.group(1)

            match = re.match(
                r"^(?:pub\s+)?struct\s+([A-Za-z_][A-Za-z0-9_]*)",
                stripped,
            )
            if match:
                return "class", match.group(1)

        # Configuration / data blocks
        if re.match(
            r"^(?:\[.+\]|---\s*$|"
            r"(?:services|service|database|databases|"
            r"server|servers|deployment|deploy|"
            r"security|auth|logging|monitoring|"
            r"environment|environments|resources)"
            r"\s*:?\s*$)",
            stripped,
            re.IGNORECASE,
        ):
            return "configuration", stripped[:80]

        # SQL
        if language == "SQL":
            match = re.match(
                r"^(CREATE|ALTER)\s+"
                r"(TABLE|VIEW|FUNCTION|PROCEDURE|TRIGGER)\s+"
                r"([A-Za-z_][A-Za-z0-9_]*)",
                stripped,
                re.IGNORECASE,
            )
            if match:
                return (
                    "sql_object",
                    match.group(3),
                )

        return None

    # ------------------------------------------------------------------
    # MAP phase
    # ------------------------------------------------------------------

    def _map_file(
        self,
        record: dict[str, Any],
    ) -> list[SemanticUnit]:
        """
        Convert one source file into semantic units.

        The algorithm first detects semantic boundaries. Very large semantic
        regions are subsequently split by character budget.
        """

        path = self._normalize_path(str(record["path"]))
        language = str(record.get("language", "Unknown"))
        lines = str(record.get("content", "")).splitlines()

        if not lines:
            return []

        boundaries: list[tuple[int, str, str]] = []

        for index, line in enumerate(lines):
            detected = self._detect_unit_start(
                line,
                language,
            )

            if detected:
                kind, symbol = detected
                boundaries.append(
                    (index, kind, symbol)
                )

        # If no semantic boundary is detectable, keep the entire file as
        # one logical region and let the reducer split it safely.
        if not boundaries:
            return [
                SemanticUnit(
                    path=path,
                    language=language,
                    start_line=1,
                    end_line=len(lines),
                    content="\n".join(lines).strip(),
                    kind="source_region",
                )
            ]

        units: list[SemanticUnit] = []

        # Preserve any imports/comments/header before the first symbol.
        first_boundary = boundaries[0][0]

        if first_boundary > 0:
            header = "\n".join(
                lines[:first_boundary]
            ).strip()

            if header:
                units.append(
                    SemanticUnit(
                        path=path,
                        language=language,
                        start_line=1,
                        end_line=first_boundary,
                        content=header,
                        kind="file_header",
                    )
                )

        for position, boundary in enumerate(boundaries):
            start_idx, kind, symbol = boundary

            if position + 1 < len(boundaries):
                end_idx = boundaries[position + 1][0]
            else:
                end_idx = len(lines)

            content = "\n".join(
                lines[start_idx:end_idx]
            ).strip()

            if not content:
                continue

            units.append(
                SemanticUnit(
                    path=path,
                    language=language,
                    start_line=start_idx + 1,
                    end_line=end_idx,
                    content=content,
                    kind=kind,
                    symbol=symbol,
                )
            )

        return units

    def map_files(
        self,
        file_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Public MAP operation.

        Returns serializable dictionaries so the intermediate representation
        can be persisted or inspected independently from the reducer.
        """

        mapped: list[dict[str, Any]] = []

        for record in file_records:
            units = self._map_file(record)

            for unit_index, unit in enumerate(units):
                mapped.append(
                    {
                        "path": unit.path,
                        "language": unit.language,
                        "start_line": unit.start_line,
                        "end_line": unit.end_line,
                        "content": unit.content,
                        "kind": unit.kind,
                        "symbol": unit.symbol,
                        "unit_index": unit_index,
                        "char_count": unit.char_count,
                    }
                )

        return mapped

    # ------------------------------------------------------------------
    # Context injection
    # ------------------------------------------------------------------

    @staticmethod
    def _trim_context(
        value: str,
        maximum: int,
    ) -> str:
        if len(value) <= maximum:
            return value

        return value[:maximum].rstrip() + "\n[CONTEXT_TRUNCATED]"

    def _inject_context(
        self,
        unit: dict[str, Any],
        previous_unit: dict[str, Any] | None,
        next_unit: dict[str, Any] | None,
    ) -> str:
        """
        Inject deterministic metadata around the source.

        This is intentionally explicit so the model receives provenance
        together with the source rather than relying on hidden assumptions.
        """

        previous_text = ""

        if previous_unit:
            previous_text = (
                f"Previous semantic unit: "
                f"{previous_unit.get('kind', 'unknown')}"
            )

            if previous_unit.get("symbol"):
                previous_text += (
                    f" `{previous_unit['symbol']}`"
                )

        next_text = ""

        if next_unit:
            next_text = (
                f"Next semantic unit: "
                f"{next_unit.get('kind', 'unknown')}"
            )

            if next_unit.get("symbol"):
                next_text += (
                    f" `{next_unit['symbol']}`"
                )

        context_parts = [
            f"File: {unit['path']}",
            f"Language: {unit['language']}",
            f"Semantic type: {unit['kind']}",
        ]

        if unit.get("symbol"):
            context_parts.append(
                f"Symbol: {unit['symbol']}"
            )

        if previous_text:
            context_parts.append(previous_text)

        if next_text:
            context_parts.append(next_text)

        context = "\n".join(
            f"[{part}]"
            for part in context_parts
        )

        return self._trim_context(
            context,
            self.max_context_chars,
        )

    # ------------------------------------------------------------------
    # REDUCE phase
    # ------------------------------------------------------------------

    def _split_large_unit(
        self,
        unit: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Split a semantic unit only when it exceeds the character budget.

        Splitting remains line-aware and deterministic.
        """

        content = str(unit["content"])
        lines = content.splitlines()

        if len(content) <= self.chunk_size:
            return [unit]

        result: list[dict[str, Any]] = []

        start = 0
        part_index = 0

        while start < len(lines):
            current_size = 0
            end = start

            while end < len(lines):
                line_size = len(lines[end]) + 1

                if (
                    end > start
                    and current_size + line_size
                    > self.chunk_size
                ):
                    break

                current_size += line_size
                end += 1

            part_content = "\n".join(
                lines[start:end]
            ).strip()

            if part_content:
                result.append(
                    {
                        **unit,
                        "content": part_content,
                        "start_line": (
                            int(unit["start_line"]) + start
                        ),
                        "end_line": (
                            int(unit["start_line"]) + end - 1
                        ),
                        "part_index": part_index,
                    }
                )

            if end >= len(lines):
                break

            # Small deterministic line overlap.
            overlap_lines = max(
                1,
                min(
                    8,
                    self.chunk_overlap // 100,
                ),
            )

            start = max(
                start + 1,
                end - overlap_lines,
            )

            part_index += 1

        return result

    def reduce_units(
        self,
        mapped_units: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        REDUCE phase.

        Adjacent compatible units are combined while staying inside the
        configured character budget.
        """

        expanded: list[dict[str, Any]] = []

        for unit in mapped_units:
            expanded.extend(
                self._split_large_unit(unit)
            )

        if not expanded:
            return []

        reduced: list[dict[str, Any]] = []

        current: dict[str, Any] | None = None

        for unit in expanded:
            if current is None:
                current = dict(unit)
                continue

            same_file = (
                current["path"] == unit["path"]
            )

            combined_size = (
                len(current["content"])
                + len(unit["content"])
                + 2
            )

            # Keep semantic classes/functions together with adjacent source
            # from the same file when possible.
            can_merge = (
                same_file
                and combined_size <= self.chunk_size
            )

            if can_merge:
                current["content"] = (
                    current["content"].rstrip()
                    + "\n\n"
                    + unit["content"].lstrip()
                )

                current["end_line"] = unit["end_line"]

                if current.get("kind") != unit.get("kind"):
                    current["kind"] = "semantic_group"

                symbols = []

                if current.get("symbol"):
                    symbols.append(
                        str(current["symbol"])
                    )

                if unit.get("symbol"):
                    symbols.append(
                        str(unit["symbol"])
                    )

                current["symbol"] = ", ".join(
                    dict.fromkeys(symbols)
                )

                continue

            reduced.append(current)
            current = dict(unit)

        if current is not None:
            reduced.append(current)

        return reduced

    # ------------------------------------------------------------------
    # Final evidence generation
    # ------------------------------------------------------------------

    def build_evidence_chunks(
        self,
        file_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Complete Map -> Context Injection -> Reduce operation.
        """

        mapped = self.map_files(file_records)
        reduced = self.reduce_units(mapped)

        chunks: list[dict[str, Any]] = []

        for index, unit in enumerate(reduced):
            previous_unit = (
                reduced[index - 1]
                if index > 0
                and reduced[index - 1]["path"] == unit["path"]
                else None
            )

            next_unit = (
                reduced[index + 1]
                if index + 1 < len(reduced)
                and reduced[index + 1]["path"] == unit["path"]
                else None
            )

            context = self._inject_context(
                unit,
                previous_unit,
                next_unit,
            )

            evidence_id = self._evidence_id(
                unit["path"],
                int(unit["start_line"]),
                int(unit["end_line"]),
                index,
            )

            source_hash = self._sha256(
                str(unit["content"])
            )

            text = (
                f"[EVIDENCE_ID: {evidence_id}]\n"
                f"[FILE: {unit['path']}]\n"
                f"[LANGUAGE: {unit['language']}]\n"
                f"[LINES: {unit['start_line']}-"
                f"{unit['end_line']}]\n"
                f"[SEMANTIC_TYPE: {unit['kind']}]\n"
                f"[SOURCE_SHA256: {source_hash}]\n"
                f"[CONTEXT]\n"
                f"{context}\n"
                f"[/CONTEXT]\n"
                f"[SOURCE]\n"
                f"{unit['content']}\n"
                f"[/SOURCE]"
            )

            chunks.append(
                {
                    "id": evidence_id,
                    "text": text,
                    "content": unit["content"],
                    "path": unit["path"],
                    "start_line": unit["start_line"],
                    "end_line": unit["end_line"],
                    "chunk_index": index,
                    "language": unit["language"],
                    "kind": unit["kind"],
                    "symbol": unit.get("symbol", ""),
                    "source_sha256": source_hash,
                    "context": context,
                    "char_count": len(unit["content"]),
                }
            )

        return chunks

    # ------------------------------------------------------------------
    # Existing pipeline compatibility
    # ------------------------------------------------------------------

    def split_files(
        self,
        file_records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Backward-compatible public API used by pipeline.py.

        Existing callers continue to receive the same important fields:
            id, text, content, path, start_line, end_line,
            chunk_index, language

        Additional semantic metadata is included without breaking callers.
        """

        return self.build_evidence_chunks(
            file_records
        )

    # ------------------------------------------------------------------
    # Legacy string API
    # ------------------------------------------------------------------

    def split(
        self,
        combined_source: str,
    ) -> list[str]:
        """
        Legacy API.

        This method cannot recover file/symbol metadata because the caller
        supplied only one combined string. It therefore performs deterministic
        line-aware chunking.
        """

        lines = combined_source.splitlines()

        if not lines:
            return []

        chunks: list[str] = []

        start = 0

        while start < len(lines):
            current_size = 0
            end = start

            while end < len(lines):
                line_size = len(lines[end]) + 1

                if (
                    end > start
                    and current_size + line_size
                    > self.chunk_size
                ):
                    break

                current_size += line_size
                end += 1

            value = "\n".join(
                lines[start:end]
            ).strip()

            if value:
                chunks.append(value)

            if end >= len(lines):
                break

            overlap_lines = max(
                1,
                min(
                    8,
                    self.chunk_overlap // 100,
                ),
            )

            start = max(
                start + 1,
                end - overlap_lines,
            )

        return chunks

    def add_context(
        self,
        chunks: list[Any],
    ) -> list[Any]:
        """
        Compatibility API.

        New source-aware callers should use split_files(), which performs
        context injection as part of the normal Map-Reduce flow.
        """

        return chunks