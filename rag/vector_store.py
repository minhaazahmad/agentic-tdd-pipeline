"""
Lightweight evidence retrieval store.

Design goals:
- Dependency-free lexical retrieval by default.
- Optional sentence-transformers hybrid reranking.
- Application source gets higher priority than generated platform
  boilerplate.
- README/configuration remains useful but is not treated as proof
  of implementation by itself.
- Stable metadata and evidence IDs are preserved.
- Existing CodeVectorStore API remains compatible with the pipeline.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(
    r"[A-Za-z0-9_./:@-]+"
)


class CodeVectorStore:

    # ============================================================
    # FILE CATEGORIES
    # ============================================================

    APPLICATION_EXTENSIONS = {
        ".py",
        ".dart",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".kt",
        ".kts",
        ".cpp",
        ".cc",
        ".cxx",
        ".c",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".swift",
        ".sql",
        ".html",
        ".css",
        ".scss",
    }

    DOCUMENTATION_EXTENSIONS = {
        ".md",
        ".txt",
    }

    CONFIGURATION_EXTENSIONS = {
        ".yaml",
        ".yml",
        ".json",
        ".xml",
        ".toml",
        ".ini",
        ".cfg",
    }

    PLATFORM_ROOTS = {
        "android",
        "ios",
        "linux",
        "windows",
        "macos",
        "web",
    }

    LOW_VALUE_NAMES = {
        "cmakelists.txt",
        "gradle.properties",
        "gradlew",
        "gradlew.bat",
        "launch_background.xml",
        "styles.xml",
        "night_styles.xml",
        "flutter_window.h",
        "flutter_window.cpp",
        "my_application.cc",
        "win32_window.cpp",
        "win32_window.h",
    }

    HIGH_VALUE_NAMES = {
        "main.py",
        "app.py",
        "pipeline.py",
        "main.dart",
        "pubspec.yaml",
        "requirements.txt",
        "package.json",
        "pyproject.toml",
        "readme.md",
        "readme.txt",
    }

    # ============================================================
    # INIT
    # ============================================================

    def __init__(
        self,
        persist_directory: str = "output/rag_store",
        model_name: str = "all-MiniLM-L6-v2",
        mode: str | None = None,
    ) -> None:

        self.persist_directory = (
            persist_directory
        )

        self.model_name = model_name

        self.mode = (
            mode
            or os.getenv(
                "RAG_MODE",
                "lexical",
            )
        ).lower()

        self.documents: list[
            dict[str, Any]
        ] = []

        self.embedding_model = None

        # Optional hybrid mode.
        if self.mode == "hybrid":

            try:

                from sentence_transformers import (
                    SentenceTransformer,
                )

                self.embedding_model = (
                    SentenceTransformer(
                        model_name
                    )
                )

                self.mode = "hybrid"

            except Exception as exc:

                print(
                    "Embedding backend unavailable; "
                    "using lexical retrieval: "
                    f"{exc}"
                )

                self.mode = "lexical"

    # ============================================================
    # TOKENIZATION
    # ============================================================

    @staticmethod
    def _tokens(
        text: str,
    ) -> list[str]:

        return [
            token.lower()
            for token in TOKEN_RE.findall(
                text
            )
        ]

    # ============================================================
    # PATH CLASSIFICATION
    # ============================================================

    @classmethod
    def _path_info(
        cls,
        path: str,
    ) -> dict[str, Any]:

        normalized = (
            str(path)
            .replace("\\", "/")
            .strip("/")
        )

        path_obj = Path(
            normalized
        )

        suffix = (
            path_obj.suffix.lower()
        )

        name = (
            path_obj.name.lower()
        )

        parts = [
            part.lower()
            for part in path_obj.parts
        ]

        platform = bool(
            set(parts)
            & cls.PLATFORM_ROOTS
        )

        application = (
            suffix
            in cls.APPLICATION_EXTENSIONS
        )

        documentation = (
            suffix
            in cls.DOCUMENTATION_EXTENSIONS
            or name.startswith(
                "readme"
            )
        )

        configuration = (
            suffix
            in cls.CONFIGURATION_EXTENSIONS
        )

        low_value = (
            name
            in cls.LOW_VALUE_NAMES
        )

        high_value = (
            name
            in cls.HIGH_VALUE_NAMES
        )

        # A file such as android/app/src/main/
        # is platform boilerplate unless its filename
        # clearly represents application logic.
        platform_boilerplate = (
            platform
            and not high_value
            and (
                low_value
                or not application
            )
        )

        return {
            "path": normalized,
            "suffix": suffix,
            "name": name,
            "platform": platform,
            "application": application,
            "documentation": documentation,
            "configuration": configuration,
            "low_value": low_value,
            "high_value": high_value,
            "platform_boilerplate": (
                platform_boilerplate
            ),
        }

    # ============================================================
    # DOCUMENT SCORE
    # ============================================================

    @classmethod
    def _document_prior(
        cls,
        metadata: dict[str, Any],
    ) -> float:

        info = cls._path_info(
            str(
                metadata.get(
                    "path",
                    "",
                )
            )
        )

        score = 1.0

        # Application implementation is strongest.
        if info["application"]:
            score += 0.30

        # Main/application entry files are highly useful.
        if info["high_value"]:
            score += 0.20

        # Documentation is useful, but should not dominate.
        if info["documentation"]:
            score += 0.08

        # Configuration can reveal dependencies/endpoints/settings.
        if info["configuration"]:
            score += 0.10

        # Generated platform files are down-ranked.
        if info["platform_boilerplate"]:
            score -= 0.45

        if info["low_value"]:
            score -= 0.30

        return max(
            0.20,
            score,
        )

    # ============================================================
    # LEXICAL SCORE
    # ============================================================

    @staticmethod
    def _score(
        query_tokens: list[str],
        doc_tokens: list[str],
    ) -> float:

        if (
            not query_tokens
            or not doc_tokens
        ):
            return 0.0

        q = Counter(
            query_tokens
        )

        d = Counter(
            doc_tokens
        )

        overlap = sum(
            min(
                q[token],
                d[token],
            )
            for token in q
        )

        unique_query_tokens = (
            len(
                set(
                    query_tokens
                )
            )
        )

        coverage = (
            overlap
            / max(
                1,
                unique_query_tokens,
            )
        )

        # Slight preference for compact evidence.
        length_penalty = (
            1.0
            / math.sqrt(
                max(
                    1.0,
                    len(doc_tokens)
                    / 100.0,
                )
            )
        )

        # Longer meaningful identifiers receive a small bonus.
        exact_bonus = sum(
            1.0
            for token in set(
                query_tokens
            )
            if (
                token in d
                and len(token) > 5
            )
        )

        return (
            coverage
            + 0.04
            * exact_bonus
            + 0.02
            * length_penalty
        )

    # ============================================================
    # QUERY INTENT
    # ============================================================

    @staticmethod
    def _query_tokens(
        query: str,
    ) -> list[str]:

        return [
            token
            for token in CodeVectorStore._tokens(
                query
            )
            if len(token) > 1
        ]

    @classmethod
    def _intent_bonus(
        cls,
        query: str,
        metadata: dict[str, Any],
        text: str,
    ) -> float:

        query_lower = (
            query.lower()
        )

        info = cls._path_info(
            str(
                metadata.get(
                    "path",
                    "",
                )
            )
        )

        text_lower = text.lower()

        bonus = 0.0

        # --------------------------------------------------------
        # UI / application workflow
        # --------------------------------------------------------

        if any(
            term in query_lower
            for term in (
                "ui",
                "screen",
                "workflow",
                "application",
                "entry point",
                "main",
            )
        ):

            if info["application"]:
                bonus += 0.12

            if any(
                term in text_lower
                for term in (
                    "widget",
                    "screen",
                    "build(",
                    "materialapp",
                    "runapp",
                    "main(",
                )
            ):
                bonus += 0.10

        # --------------------------------------------------------
        # API / network
        # --------------------------------------------------------

        if any(
            term in query_lower
            for term in (
                "api",
                "http",
                "network",
                "endpoint",
                "request",
                "response",
            )
        ):

            if any(
                term in text_lower
                for term in (
                    "http",
                    "https",
                    "request",
                    "response",
                    "endpoint",
                    "post(",
                    "get(",
                    "put(",
                    "delete(",
                )
            ):
                bonus += 0.16

        # --------------------------------------------------------
        # File / storage
        # --------------------------------------------------------

        if any(
            term in query_lower
            for term in (
                "file",
                "upload",
                "download",
                "storage",
                "archive",
                "zip",
            )
        ):

            if any(
                term in text_lower
                for term in (
                    "file",
                    "upload",
                    "download",
                    "archive",
                    "zip",
                    "path",
                    "directory",
                )
            ):
                bonus += 0.14

        # --------------------------------------------------------
        # Testing
        # --------------------------------------------------------

        if any(
            term in query_lower
            for term in (
                "test",
                "tests",
                "validation",
            )
        ):

            path = str(
                metadata.get(
                    "path",
                    "",
                )
            ).lower()

            if (
                "test"
                in path
            ):
                bonus += 0.22

            if any(
                term in text_lower
                for term in (
                    "test(",
                    "expect(",
                    "assert",
                    "pytest",
                    "testcase",
                )
            ):
                bonus += 0.16

        # --------------------------------------------------------
        # Dependencies / configuration
        # --------------------------------------------------------

        if any(
            term in query_lower
            for term in (
                "dependency",
                "dependencies",
                "package",
                "configuration",
                "environment",
            )
        ):

            if (
                info["configuration"]
                or info["high_value"]
            ):
                bonus += 0.16

        # --------------------------------------------------------
        # README / setup
        # --------------------------------------------------------

        if any(
            term in query_lower
            for term in (
                "readme",
                "setup",
                "installation",
                "usage",
                "architecture",
            )
        ):

            if info[
                "documentation"
            ]:
                bonus += 0.16

        # --------------------------------------------------------
        # Penalize generated platform boilerplate.
        # --------------------------------------------------------

        if info[
            "platform_boilerplate"
        ]:

            bonus -= 0.18

        return bonus

    # ============================================================
    # ADD CHUNKS
    # ============================================================

    def add_chunks(
        self,
        chunks: list[Any],
    ) -> None:

        self.documents.clear()

        for index, chunk in enumerate(
            chunks
        ):

            if isinstance(
                chunk,
                str,
            ):

                text = chunk.strip()

                metadata = {
                    "chunk_index": index,
                }

                chunk_id = (
                    f"chunk_{index}"
                )

            else:

                text = str(
                    chunk.get(
                        "text",
                        chunk.get(
                            "content",
                            "",
                        ),
                    )
                ).strip()

                metadata = {
                    "chunk_index": index,
                    "id": str(
                        chunk.get(
                            "id",
                            f"chunk_{index}",
                        )
                    ),
                    "path": str(
                        chunk.get(
                            "path",
                            "",
                        )
                    ),
                    "start_line": chunk.get(
                        "start_line"
                    ),
                    "end_line": chunk.get(
                        "end_line"
                    ),
                    "language": str(
                        chunk.get(
                            "language",
                            "",
                        )
                    ),
                }

                chunk_id = metadata[
                    "id"
                ]

            if not text:
                continue

            path_info = (
                self._path_info(
                    str(
                        metadata.get(
                            "path",
                            "",
                        )
                    )
                )
            )

            metadata[
                "source_type"
            ] = (
                "platform"
                if path_info[
                    "platform_boilerplate"
                ]
                else (
                    "documentation"
                    if path_info[
                        "documentation"
                    ]
                    else (
                        "configuration"
                        if path_info[
                            "configuration"
                        ]
                        else "application"
                    )
                )
            )

            metadata[
                "retrieval_prior"
            ] = round(
                self._document_prior(
                    metadata
                ),
                4,
            )

            self.documents.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "_tokens": self._tokens(
                        text
                    ),
                }
            )

        # Optional embedding vectors.
        if (
            self.embedding_model is not None
            and self.documents
        ):

            vectors = (
                self.embedding_model.encode(
                    [
                        doc["text"]
                        for doc in self.documents
                    ],
                    show_progress_bar=False,
                )
            )

            for doc, vector in zip(
                self.documents,
                vectors,
            ):

                doc[
                    "_embedding"
                ] = vector

        print(
            "Stored "
            + str(
                len(
                    self.documents
                )
            )
            + " evidence chunks ("
            + self.mode
            + " retrieval)."
        )

    # ============================================================
    # HYBRID SCORE
    # ============================================================

    def _hybrid_score(
        self,
        query: str,
        doc: dict[str, Any],
        lexical: float,
    ) -> float:

        if (
            self.embedding_model is None
            or "_embedding" not in doc
        ):

            return lexical

        try:

            q = (
                self.embedding_model.encode(
                    [query],
                    show_progress_bar=False,
                )[0]
            )

            v = doc[
                "_embedding"
            ]

            denom = (
                sum(
                    float(x) ** 2
                    for x in q
                )
                ** 0.5
            ) * (
                sum(
                    float(x) ** 2
                    for x in v
                )
                ** 0.5
            )

            cosine = (
                0.0
                if denom == 0
                else sum(
                    float(a)
                    * float(b)
                    for a, b in zip(
                        q,
                        v,
                    )
                )
                / denom
            )

            return (
                0.55
                * lexical
                + 0.45
                * cosine
            )

        except Exception:

            return lexical

    # ============================================================
    # SEARCH
    # ============================================================

    def search_with_metadata(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[
        dict[str, Any]
    ]:

        query = str(
            query
        ).strip()

        if not query:
            return []

        q_tokens = (
            self._query_tokens(
                query
            )
        )

        ranked = []

        for doc in self.documents:

            lexical = self._score(
                q_tokens,
                doc["_tokens"],
            )

            semantic = (
                self._hybrid_score(
                    query,
                    doc,
                    lexical,
                )
            )

            prior = float(
                doc[
                    "metadata"
                ].get(
                    "retrieval_prior",
                    1.0,
                )
            )

            intent = (
                self._intent_bonus(
                    query,
                    doc[
                        "metadata"
                    ],
                    doc[
                        "text"
                    ],
                )
            )

            # Combine relevance and source quality.
            score = (
                semantic
                * prior
                + intent
            )

            ranked.append(
                {
                    "id": doc[
                        "id"
                    ],
                    "text": doc[
                        "text"
                    ],
                    "metadata": doc[
                        "metadata"
                    ],
                    "score": round(
                        float(
                            score
                        ),
                        6,
                    ),
                }
            )

        ranked.sort(
            key=lambda item: (
                item["score"],
                item["id"],
            ),
            reverse=True,
        )

        return ranked[
            : max(
                1,
                top_k,
            )
        ]

    # ============================================================
    # SIMPLE SEARCH API
    # ============================================================

    def search(
        self,
        query: str,
        top_k: int = 3,
    ) -> list[str]:

        return [
            item["text"]
            for item in self.search_with_metadata(
                query,
                top_k,
            )
        ]

    # ============================================================
    # CLEAR
    # ============================================================

    def clear(
        self,
    ) -> None:

        self.documents.clear()


# ================================================================
# SELF TEST
# ================================================================


if __name__ == "__main__":

    store = CodeVectorStore()

    store.add_chunks(
        [
            {
                "id": "EV-MAIN",
                "path": "lib/main.dart",
                "text": (
                    "[FILE: lib/main.dart]\n"
                    "Flutter application with "
                    "audio file selection and "
                    "playback controls."
                ),
                "language": "Dart",
            },
            {
                "id": "EV-README",
                "path": "README.md",
                "text": (
                    "[FILE: README.md]\n"
                    "The application supports "
                    "audio stem separation."
                ),
                "language": "Markdown",
            },
            {
                "id": "EV-ANDROID",
                "path": (
                    "android/app/src/main/"
                    "AndroidManifest.xml"
                ),
                "text": (
                    "[FILE: AndroidManifest.xml]\n"
                    "Android application manifest."
                ),
                "language": "XML",
            },
            {
                "id": "EV-TEST",
                "path": "test/widget_test.dart",
                "text": (
                    "[FILE: test/widget_test.dart]\n"
                    "Widget test using testWidgets "
                    "and expect."
                ),
                "language": "Dart",
            },
        ]
    )

    print(
        "\nAPI search:"
    )

    print(
        store.search_with_metadata(
            "application API network",
            top_k=3,
        )
    )

    print(
        "\nTesting search:"
    )

    print(
        store.search_with_metadata(
            "tests validation",
            top_k=3,
        )
    )

    print(
        "\nUI search:"
    )

    print(
        store.search_with_metadata(
            "main UI application workflow",
            top_k=3,
        )
    )