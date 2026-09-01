"""
Evidence-Grounded Architect Agent.

Responsibilities:
    Scanner -> Chunker -> Retriever -> Architect

The Architect converts retrieved source evidence into a structured
technical-design blueprint.

Important guarantees:
- Never invent evidence IDs.
- Never use platform boilerplate as application architecture.
- Prefer real application source over generated runner files.
- Distinguish README/documentation claims from implementation evidence.
- Unsupported details are explicitly marked.
- LLM failures automatically fall back to deterministic generation.
- Returned data always has the schema expected by the Manager/Critic.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from langchain_groq import ChatGroq
except Exception:
    ChatGroq = None


load_dotenv()


NOT_DETERMINED = (
    "Not determined from available project evidence."
)


class ArchitectAgent:
    """
    Evidence-grounded technical architecture generator.
    """

    # ============================================================
    # INIT
    # ============================================================

    def __init__(self) -> None:
        self.llm = None

        api_key = os.getenv("GROQ_API_KEY")

        if api_key and ChatGroq is not None:
            try:
                self.llm = ChatGroq(
                    api_key=api_key,
                    model=os.getenv(
                        "GROQ_MODEL",
                        "llama-3.1-8b-instant",
                    ),
                    temperature=0,
                    max_tokens=3000,
                )
            except Exception as exc:
                print(
                    "Architect LLM initialization failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                self.llm = None

    # ============================================================
    # BASIC HELPERS
    # ============================================================

    @staticmethod
    def _text(item: dict[str, Any]) -> str:
        return str(
            item.get(
                "text",
                item.get(
                    "content",
                    "",
                ),
            )
        )

    @staticmethod
    def _path(item: dict[str, Any]) -> str:
        metadata = item.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {}

        return str(
            metadata.get(
                "path",
                item.get(
                    "path",
                    "",
                ),
            )
        ).replace("\\", "/")

    @staticmethod
    def _id(item: dict[str, Any]) -> str:
        metadata = item.get(
            "metadata",
            {},
        )

        if not isinstance(metadata, dict):
            metadata = {}

        return str(
            item.get(
                "id",
                metadata.get(
                    "id",
                    "",
                ),
            )
        ).strip().upper()

    # ============================================================
    # EVIDENCE CLASSIFICATION
    # ============================================================

    @classmethod
    def _is_documentation(
        cls,
        item: dict[str, Any],
    ) -> bool:
        path = cls._path(item).lower()
        name = Path(path).name.lower()
        suffix = Path(path).suffix.lower()

        return (
            suffix in {".md", ".txt"}
            or name.startswith("readme")
        )

    @classmethod
    def _is_configuration(
        cls,
        item: dict[str, Any],
    ) -> bool:
        path = cls._path(item).lower()
        suffix = Path(path).suffix.lower()
        name = Path(path).name.lower()

        return (
            suffix
            in {
                ".yaml",
                ".yml",
                ".json",
                ".toml",
                ".ini",
                ".cfg",
            }
            or name in {
                "requirements.txt",
                "pubspec.yaml",
                "package.json",
                "pyproject.toml",
            }
        )

    @classmethod
    def _is_application_source(
        cls,
        item: dict[str, Any],
    ) -> bool:
        path = cls._path(item).lower()
        suffix = Path(path).suffix.lower()

        return suffix in {
            ".py",
            ".dart",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".java",
            ".kt",
            ".kts",
            ".swift",
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
            ".sql",
            ".html",
            ".css",
            ".scss",
        }

    @classmethod
    def _is_platform_boilerplate(
        cls,
        item: dict[str, Any],
    ) -> bool:
        path = cls._path(item).lower()

        parts = set(
            Path(path).parts
        )

        filename = Path(
            path
        ).name.lower()

        platform_dirs = {
            "android",
            "ios",
            "linux",
            "windows",
            "macos",
        }

        generated_names = {
            "cmakelists.txt",
            "launch_background.xml",
            "styles.xml",
            "night_styles.xml",
            "flutter_window.h",
            "flutter_window.cpp",
            "my_application.cc",
            "my_application.h",
            "win32_window.cpp",
            "win32_window.h",
            "resource.h",
            "runner.rc",
            "generated_plugin_registrant.dart",
            "generated_plugin_registrant.cc",
            "generated_plugin_registrant.h",
        }

        if filename in generated_names:
            return True

        if (
            parts & platform_dirs
        ):
            # Application-specific source inside a platform
            # directory can still be meaningful.
            if filename.endswith(
                (
                    ".dart",
                    ".py",
                    ".js",
                    ".jsx",
                    ".ts",
                    ".tsx",
                    ".kt",
                    ".java",
                    ".swift",
                )
            ):
                return False

            return True

        if (
            path.startswith("web/")
            and filename in {
                "index.html",
                "manifest.json",
            }
        ):
            return True

        return False

    # ============================================================
    # EVIDENCE RANKING
    # ============================================================

    @classmethod
    def _evidence_priority(
        cls,
        item: dict[str, Any],
    ) -> float:
        score = 0.0

        path = cls._path(item)
        name = Path(path).name.lower()
        text = cls._text(item).lower()

        if cls._is_application_source(item):
            score += 5.0

        if cls._is_documentation(item):
            score += 2.0

        if cls._is_configuration(item):
            score += 2.0

        if cls._is_platform_boilerplate(item):
            score -= 6.0

        if name in {
            "main.py",
            "app.py",
            "pipeline.py",
            "main.dart",
        }:
            score += 3.0

        implementation_keywords = [
            "class ",
            "def ",
            "async ",
            "http",
            "https",
            "upload",
            "download",
            "audio",
            "player",
            "volume",
            "widget",
            "request",
            "response",
            "endpoint",
            "archive",
            "extract",
            "permission",
        ]

        for keyword in implementation_keywords:
            if keyword in text:
                score += 0.12

        return score

    @classmethod
    def _rank_evidence(
        cls,
        evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        ranked = []

        for item in evidence:
            ranked.append(
                (
                    cls._evidence_priority(item),
                    item,
                )
            )

        ranked.sort(
            key=lambda pair: (
                pair[0],
                cls._path(pair[1]),
                cls._id(pair[1]),
            ),
            reverse=True,
        )

        return [
            item
            for _, item in ranked
        ]

    # ============================================================
    # SELECT USEFUL EVIDENCE
    # ============================================================

    @classmethod
    def _select_evidence(
        cls,
        evidence: list[dict[str, Any]],
        max_items: int = 24,
    ) -> list[dict[str, Any]]:
        ranked = cls._rank_evidence(
            evidence
        )

        selected = []
        seen_ids = set()
        seen_paths = set()

        # First pass: application implementation.
        for item in ranked:
            if cls._is_platform_boilerplate(item):
                continue

            evidence_id = cls._id(item)
            path = cls._path(item)

            if not evidence_id:
                continue

            if evidence_id in seen_ids:
                continue

            if path and path in seen_paths:
                continue

            selected.append(item)
            seen_ids.add(evidence_id)

            if path:
                seen_paths.add(path)

            if len(selected) >= max_items:
                break

        return selected

    # ============================================================
    # CONTEXT BUILDER
    # ============================================================

    @classmethod
    def _build_context(
        cls,
        evidence: list[dict[str, Any]],
        max_chars: int = 8500,
    ) -> str:
        selected = cls._select_evidence(
            evidence,
            max_items=24,
        )

        blocks = []
        used_chars = 0

        for item in selected:
            evidence_id = cls._id(item)
            path = cls._path(item)
            text = cls._text(item).strip()

            if not text:
                continue

            remaining = (
                max_chars
                - used_chars
            )

            if remaining <= 0:
                break

            # Keep individual evidence compact so Groq does not
            # reject the request because of TPM/token limits.
            snippet = text[
                : min(
                    1100,
                    remaining,
                )
            ]

            if cls._is_documentation(item):
                evidence_type = "DOCUMENTATION"
            elif cls._is_configuration(item):
                evidence_type = "CONFIGURATION"
            else:
                evidence_type = "IMPLEMENTATION"

            block = (
                f"EVIDENCE_ID: {evidence_id}\n"
                f"FILE: {path}\n"
                f"TYPE: {evidence_type}\n"
                f"SOURCE:\n{snippet}"
            )

            blocks.append(block)
            used_chars += len(block) + 2

        return "\n\n".join(blocks)

    # ============================================================
    # VALID IDS
    # ============================================================

    @classmethod
    def _valid_evidence_ids(
        cls,
        evidence: list[dict[str, Any]],
    ) -> set[str]:
        return {
            cls._id(item)
            for item in evidence
            if cls._id(item)
        }

    # ============================================================
    # JSON EXTRACTION
    # ============================================================

    @staticmethod
    def _strip_markdown(
        content: str,
    ) -> str:
        text = str(
            content
        ).strip()

        text = re.sub(
            r"^```json\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"^```\s*",
            "",
            text,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        return text.strip()

    @classmethod
    def _extract_json_object(
        cls,
        content: str,
    ) -> str:
        content = cls._strip_markdown(
            content
        )

        start = content.find("{")

        if start < 0:
            raise ValueError(
                "Architect response contained no JSON object."
            )

        depth = 0
        in_string = False
        escaped = False

        for index in range(
            start,
            len(content),
        ):
            char = content[index]

            if escaped:
                escaped = False
                continue

            if (
                char == "\\"
                and in_string
            ):
                escaped = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == "{":
                depth += 1

            elif char == "}":
                depth -= 1

                if depth == 0:
                    return content[
                        start:index + 1
                    ]

        raise ValueError(
            "Architect returned incomplete JSON."
        )

    @classmethod
    def _parse_json(
        cls,
        content: str,
    ) -> dict[str, Any]:
        content = cls._strip_markdown(
            content
        )

        try:
            value = json.loads(
                content
            )

            if isinstance(
                value,
                dict,
            ):
                return value

        except json.JSONDecodeError:
            pass

        candidate = (
            cls._extract_json_object(
                content
            )
        )

        value = json.loads(
            candidate
        )

        if not isinstance(
            value,
            dict,
        ):
            raise ValueError(
                "Architect JSON must be an object."
            )

        return value

    # ============================================================
    # CITATION CLEANING
    # ============================================================

    @classmethod
    def _sanitize_citations(
        cls,
        value: Any,
        valid_ids: set[str],
    ) -> Any:
        if isinstance(
            value,
            dict,
        ):
            return {
                key: cls._sanitize_citations(
                    item,
                    valid_ids,
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            list,
        ):
            return [
                cls._sanitize_citations(
                    item,
                    valid_ids,
                )
                for item in value
            ]

        if not isinstance(
            value,
            str,
        ):
            return value

        pattern = re.compile(
            r"\[?(EV-[A-Za-z0-9_-]+)\]?",
            flags=re.IGNORECASE,
        )

        def replacement(
            match: re.Match[str],
        ) -> str:
            evidence_id = (
                match.group(1)
                .upper()
            )

            if evidence_id in valid_ids:
                return (
                    "["
                    + evidence_id
                    + "]"
                )

            return ""

        return pattern.sub(
            replacement,
            value,
        ).strip()

    # ============================================================
    # LIST NORMALIZATION
    # ============================================================

    @staticmethod
    def _list(
        value: Any,
    ) -> list[Any]:
        if isinstance(
            value,
            list,
        ) and value:
            return value

        if isinstance(
            value,
            str,
        ) and value.strip():
            return [value.strip()]

        return [
            NOT_DETERMINED
        ]

    # ============================================================
    # BLUEPRINT NORMALIZATION
    # ============================================================

    @classmethod
    def _normalize(
        cls,
        data: dict[str, Any],
        filename: str,
        scanner_data: dict[str, Any],
    ) -> dict[str, Any]:

        architecture = data.get(
            "architecture",
            {},
        )

        if not isinstance(
            architecture,
            dict,
        ):
            architecture = {}

        result = {
            "document_title": data.get(
                "document_title",
                "Technical Design Document - "
                + filename,
            ),

            "source_file": data.get(
                "source_file",
                filename,
            ),

            "system_overview": data.get(
                "system_overview",
                NOT_DETERMINED,
            ),

            "project_scope": cls._list(
                data.get(
                    "project_scope"
                )
            ),

            "architecture": {
                "components": cls._list(
                    architecture.get(
                        "components"
                    )
                ),
                "component_responsibilities": cls._list(
                    architecture.get(
                        "component_responsibilities"
                    )
                ),
                "data_flow": cls._list(
                    architecture.get(
                        "data_flow"
                    )
                ),
                "dependencies": cls._list(
                    architecture.get(
                        "dependencies"
                    )
                ),
            },

            "functional_requirements": cls._list(
                data.get(
                    "functional_requirements"
                )
            ),

            "non_functional_requirements": cls._list(
                data.get(
                    "non_functional_requirements"
                )
            ),

            "technical_details": cls._list(
                data.get(
                    "technical_details"
                )
            ),

            "api_design": cls._list(
                data.get(
                    "api_design"
                )
            ),

            "data_storage": cls._list(
                data.get(
                    "data_storage"
                )
            ),

            "error_handling": cls._list(
                data.get(
                    "error_handling"
                )
            ),

            "security": cls._list(
                data.get(
                    "security"
                )
            ),

            "configuration": cls._list(
                data.get(
                    "configuration"
                )
            ),

            "testing_strategy": cls._list(
                data.get(
                    "testing_strategy"
                )
            ),

            "deployment": cls._list(
                data.get(
                    "deployment"
                )
            ),

            "performance": cls._list(
                data.get(
                    "performance"
                )
            ),

            "limitations": cls._list(
                data.get(
                    "limitations"
                )
            ),

            "source_files_of_interest": (
                data.get(
                    "source_files_of_interest",
                    [],
                )
                if isinstance(
                    data.get(
                        "source_files_of_interest",
                        [],
                    ),
                    list,
                )
                else []
            ),

            "evidence_notes": (
                data.get(
                    "evidence_notes",
                    [],
                )
                if isinstance(
                    data.get(
                        "evidence_notes",
                        [],
                    ),
                    list,
                )
                else []
            ),
        }

        result[
            "source_statistics"
        ] = {
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

        return result

    # ============================================================
    # FALLBACK CITATION
    # ============================================================

    @staticmethod
    def _citation(
        evidence_id: str,
    ) -> str:
        if not evidence_id:
            return ""

        return (
            " ["
            + evidence_id
            + "]"
        )

    # ============================================================
    # DETERMINISTIC FALLBACK
    # ============================================================

    @classmethod
    def _fallback(
        cls,
        filename: str,
        scanner_data: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:

        selected = cls._select_evidence(
            evidence,
            max_items=12,
        )

        valid_ids = [
            cls._id(item)
            for item in selected
            if cls._id(item)
        ]

        all_text = "\n".join(
            cls._text(item)
            for item in selected
        )

        lower = all_text.lower()

        def ev(
            index: int = 0,
        ) -> str:
            if not valid_ids:
                return ""

            index = max(
                0,
                min(
                    index,
                    len(valid_ids) - 1,
                ),
            )

            return cls._citation(
                valid_ids[index]
            )

        # ========================================================
        # COMPONENTS
        # ========================================================

        components = []

        if any(
            term in lower
            for term in (
                "flutter",
                "dart",
                "widget",
                "materialapp",
                "scaffold",
            )
        ):
            components.append(
                "Flutter/Dart application layer"
                + ev(0)
            )

        if any(
            term in lower
            for term in (
                "audio",
                "just_audio",
                "player",
                "volume",
            )
        ):
            components.append(
                "Audio playback and control layer"
                + ev(1)
            )

        if any(
            term in lower
            for term in (
                "file_picker",
                "path_provider",
                "archive",
                "upload",
                "download",
                "zip",
            )
        ):
            components.append(
                "File selection and local file-handling layer"
                + ev(2)
            )

        if any(
            term in lower
            for term in (
                "http",
                "multipart",
                "request",
                "response",
            )
        ):
            components.append(
                "Network communication layer"
                + ev(3)
            )

        if any(
            "test" in cls._path(item).lower()
            for item in selected
        ):
            components.append(
                "Automated testing layer"
                + ev(
                    min(
                        4,
                        len(valid_ids) - 1,
                    )
                    if valid_ids
                    else 0
                )
            )

        if not components:
            components.append(
                "Application implementation layer"
                + ev(0)
            )

        # ========================================================
        # DATA FLOW
        # ========================================================

        data_flow = []

        if any(
            term in lower
            for term in (
                "file_picker",
                "pickfiles",
                "pickfile",
            )
        ):
            data_flow.append(
                "The application provides a file-selection flow."
                + ev(0)
            )

        if any(
            term in lower
            for term in (
                "http",
                "multipart",
                "request",
            )
        ):
            data_flow.append(
                "Selected data is processed through "
                "network request logic evidenced in the source."
                + ev(2)
            )

        if any(
            term in lower
            for term in (
                "archive",
                "zip",
                "extract",
            )
        ):
            data_flow.append(
                "Archive/file output is handled by "
                "the local file-processing logic."
                + ev(3)
            )

        if any(
            term in lower
            for term in (
                "just_audio",
                "player",
                "volume",
            )
        ):
            data_flow.append(
                "Audio data is passed to the application's "
                "playback/control functionality."
                + ev(1)
            )

        if not data_flow:
            data_flow.append(
                NOT_DETERMINED
            )

        # ========================================================
        # DEPENDENCIES
        # ========================================================

        dependency_names = [
            "http",
            "file_picker",
            "just_audio",
            "archive",
            "path_provider",
            "permission_handler",
        ]

        dependencies = []

        for name in dependency_names:
            if name in lower:
                dependencies.append(
                    name
                    + ev(0)
                )

        if not dependencies:
            dependencies.append(
                NOT_DETERMINED
            )

        # ========================================================
        # FUNCTIONAL REQUIREMENTS
        # ========================================================

        functional = []

        if any(
            term in lower
            for term in (
                "file_picker",
                "pickfiles",
                "pickfile",
            )
        ):
            functional.append(
                "Allow the user to select an input file."
                + ev(0)
            )

        if any(
            term in lower
            for term in (
                "audio",
                "just_audio",
                "player",
            )
        ):
            functional.append(
                "Provide audio playback functionality "
                "supported by the retrieved implementation."
                + ev(1)
            )

        if any(
            term in lower
            for term in (
                "volume",
                "slider",
            )
        ):
            functional.append(
                "Provide audio volume controls where "
                "implemented by the application."
                + ev(1)
            )

        if not functional:
            functional.append(
                NOT_DETERMINED
            )

        # ========================================================
        # API
        # ========================================================

        if any(
            term in lower
            for term in (
                "http",
                "multipart",
                "endpoint",
            )
        ):
            api_design = [
                "Network behavior is documented only to "
                "the extent directly exposed by the "
                "retrieved source evidence."
                + ev(2)
            ]
        else:
            api_design = [
                NOT_DETERMINED
            ]

        # ========================================================
        # TESTING
        # ========================================================

        has_tests = any(
            "test" in cls._path(item).lower()
            for item in selected
        )

        if has_tests:
            testing = [
                "Testing behavior is derived from the "
                "available project test source."
                + ev(
                    min(
                        4,
                        len(valid_ids) - 1,
                    )
                    if valid_ids
                    else 0
                )
            ]
        else:
            testing = [
                NOT_DETERMINED
            ]

        # ========================================================
        # CONFIGURATION
        # ========================================================

        has_config = any(
            cls._is_configuration(item)
            for item in selected
        )

        if has_config:
            configuration = [
                "Project configuration and dependency "
                "information is derived from the available "
                "configuration evidence."
                + ev(0)
            ]
        else:
            configuration = [
                NOT_DETERMINED
            ]

        # ========================================================
        # SOURCE FILES
        # ========================================================

        source_files = []

        for item in selected:
            path = cls._path(item)

            if (
                path
                and not cls._is_platform_boilerplate(item)
            ):
                if path not in source_files:
                    source_files.append(path)

        # ========================================================
        # BLUEPRINT
        # ========================================================

        return cls._normalize(
            {
                "document_title": (
                    "Technical Design Document - "
                    + filename
                ),

                "source_file": filename,

                "system_overview": (
                    "The project architecture is derived "
                    "from deterministic source analysis and "
                    "retrieved implementation evidence."
                    + ev(0)
                ),

                "project_scope": [
                    (
                        "Document implemented functionality "
                        "supported by the supplied project evidence."
                        + ev(0)
                    ),
                    (
                        "Do not infer undocumented external "
                        "services or infrastructure."
                    ),
                ],

                "architecture": {
                    "components": components,

                    "component_responsibilities": [
                        (
                            "Application responsibilities are "
                            "derived from implementation evidence."
                            + ev(0)
                        ),
                        (
                            "Generated platform scaffolding is "
                            "not treated as application architecture."
                        ),
                    ],

                    "data_flow": data_flow,

                    "dependencies": dependencies,
                },

                "functional_requirements": functional,

                "non_functional_requirements": [
                    NOT_DETERMINED
                ],

                "technical_details": [
                    (
                        "Technical details are limited to "
                        "behavior directly supported by source "
                        "and configuration evidence."
                        + ev(0)
                    )
                ],

                "api_design": api_design,

                "data_storage": [
                    NOT_DETERMINED
                ],

                "error_handling": [
                    NOT_DETERMINED
                ],

                "security": [
                    (
                        "Security characteristics are reported "
                        "only when directly observable in the "
                        "retrieved source evidence."
                    )
                ],

                "configuration": configuration,

                "testing_strategy": testing,

                "deployment": [
                    NOT_DETERMINED
                ],

                "performance": [
                    NOT_DETERMINED
                ],

                "limitations": [
                    (
                        "External backend infrastructure, "
                        "model execution, GPU/CUDA behavior, "
                        "database architecture and authentication "
                        "are not inferred without direct evidence."
                    ),
                    (
                        "Documentation claims are distinguished "
                        "from implementation evidence."
                    ),
                ],

                "source_files_of_interest": source_files[:15],

                "evidence_notes": [
                    (
                        "Selected evidence chunks: "
                        + str(len(selected))
                    ),
                    (
                        "Evidence IDs used: "
                        + (
                            ", ".join(valid_ids)
                            if valid_ids
                            else "none"
                        )
                    ),
                ],
            },
            filename,
            scanner_data,
        )

    # ============================================================
    # LLM PROMPT
    # ============================================================

    @classmethod
    def _build_prompt(
        cls,
        filename: str,
        context: str,
        valid_ids: set[str],
    ) -> str:

        allowed_ids = "\n".join(
            sorted(valid_ids)
        )

        return f"""
You are the Architect Agent of an evidence-grounded
Technical Design Document generation system.

PROJECT:
{filename}

Your job is to create a technically accurate blueprint
using ONLY the supplied source evidence.

STRICT RULES:

1. Never invent implementation details.
2. Never invent evidence IDs.
3. Only cite IDs from ALLOWED EVIDENCE IDS.
4. Every concrete implementation claim should have [EV-ID].
5. Unsupported information MUST be:
   "{NOT_DETERMINED}"
6. Android/iOS/Linux/Windows/macOS generated runner files
   are platform scaffolding unless they clearly contain
   application-specific logic.
7. Never use platform scaffolding as proof of application
   business architecture.
8. Prefer real application source files.
9. README/documentation represents documented or intended
   behavior. It does not automatically prove implementation.
10. If README claims a backend/model/API exists but source
    evidence does not show it, do not state that it is
    implemented.
11. Do not infer GPU, CUDA, FastAPI, Demucs, native ML,
    database, authentication, cloud infrastructure or
    external services without direct evidence.
12. Do not invent API endpoints.
13. Do not invent data stores.
14. Do not invent performance metrics.
15. Do not convert dependency names into architecture claims
    unless the source shows their actual use.
16. Keep the output concise.
17. Return ONLY valid JSON.
18. Do not use Markdown fences.
19. Complete the entire JSON object.
20. Do not put commentary outside JSON.

ALLOWED EVIDENCE IDS:
{allowed_ids}

SOURCE EVIDENCE:
{context}

Return EXACTLY this JSON structure:

{{
  "document_title": "string",
  "source_file": "string",
  "system_overview": "string",
  "project_scope": ["string"],
  "architecture": {{
    "components": ["string"],
    "component_responsibilities": ["string"],
    "data_flow": ["string"],
    "dependencies": ["string"]
  }},
  "functional_requirements": ["string"],
  "non_functional_requirements": ["string"],
  "technical_details": ["string"],
  "api_design": ["string"],
  "data_storage": ["string"],
  "error_handling": ["string"],
  "security": ["string"],
  "configuration": ["string"],
  "testing_strategy": ["string"],
  "deployment": ["string"],
  "performance": ["string"],
  "limitations": ["string"],
  "source_files_of_interest": ["string"],
  "evidence_notes": ["string"]
}}
"""

    # ============================================================
    # PUBLIC METHOD
    # ============================================================

    def build_design_blueprint(
        self,
        filename: str,
        scanner_data: dict[str, Any],
        relevant_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:

        valid_ids = (
            self._valid_evidence_ids(
                relevant_chunks
            )
        )

        context = self._build_context(
            relevant_chunks,
            max_chars=8500,
        )

        # ========================================================
        # FALLBACK IF LLM IS UNAVAILABLE
        # ========================================================

        if self.llm is None:

            result = self._fallback(
                filename,
                scanner_data,
                relevant_chunks,
            )

            result[
                "_allowed_evidence_ids"
            ] = sorted(valid_ids)

            result[
                "_architect_llm_used"
            ] = False

            return result

        # ========================================================
        # LLM CALL
        # ========================================================

        prompt = self._build_prompt(
            filename,
            context,
            valid_ids,
        )

        try:

            response = self.llm.invoke(
                prompt
            )

            content = str(
                getattr(
                    response,
                    "content",
                    response,
                )
            )

            parsed = self._parse_json(
                content
            )

            parsed = self._normalize(
                parsed,
                filename,
                scanner_data,
            )

            parsed = self._sanitize_citations(
                parsed,
                valid_ids,
            )

            parsed[
                "_allowed_evidence_ids"
            ] = sorted(valid_ids)

            parsed[
                "_architect_llm_used"
            ] = True

            return parsed

        except Exception as exc:

            print(
                "Architect LLM failed; "
                "deterministic fallback used: "
                f"{type(exc).__name__}: {exc}"
            )

            result = self._fallback(
                filename,
                scanner_data,
                relevant_chunks,
            )

            result[
                "_architect_llm_used"
            ] = False

            result[
                "_architect_llm_error"
            ] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            result[
                "_allowed_evidence_ids"
            ] = sorted(valid_ids)

            return result