"""
Deterministic + optional LLM quality critic for the Agentic TDD pipeline.

The critic validates:
- blueprint schema
- evidence citation validity
- evidence coverage
- source diversity
- implementation/documentation distinction
- unsupported technical claims

Platform-generated Flutter runner/scaffolding files are explicitly
excluded from architecture-quality warnings.
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


class CriticAgent:

    REQUIRED_KEYS = {
        "document_title",
        "source_file",
        "system_overview",
        "project_scope",
        "architecture",
        "functional_requirements",
        "non_functional_requirements",
        "technical_details",
        "api_design",
        "data_storage",
        "error_handling",
        "security",
        "configuration",
        "testing_strategy",
        "deployment",
        "performance",
        "limitations",
    }

    CONCRETE_SECTIONS = {
        "system_overview",
        "project_scope",
        "architecture",
        "functional_requirements",
        "non_functional_requirements",
        "technical_details",
        "api_design",
        "data_storage",
        "error_handling",
        "security",
        "configuration",
        "testing_strategy",
        "deployment",
        "performance",
    }

    DOCUMENTATION_EXTENSIONS = {
        ".md",
        ".txt",
        ".rst",
    }

    IMPLEMENTATION_EXTENSIONS = {
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

    PLATFORM_PATH_TERMS = {
        "android/",
        "ios/",
        "linux/",
        "windows/",
        "macos/",
    }

    PLATFORM_FILENAMES = {
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
        "utils.h",
    }

    RISKY_TERMS = {
        "cuda",
        "gpu",
        "fastapi",
        "demucs",
        "deep learning",
        "machine learning",
        "native inference",
        "model inference",
        "database",
        "postgres",
        "mysql",
        "mongodb",
        "authentication",
        "oauth",
        "jwt",
        "encryption",
        "cloud",
        "docker",
        "kubernetes",
    }

    def __init__(self) -> None:
        self.llm = None

        api_key = os.getenv("GROQ_API_KEY")

        if api_key and ChatGroq is not None:
            try:
                self.llm = ChatGroq(
                    api_key=api_key,
                    model=os.getenv(
                        "GROQ_MODEL",
                        "openai/gpt-oss-20b",
                    ),
                    temperature=0,
                    max_tokens=1200,
                    reasoning_effort="low",
                    reasoning_format="hidden",
                    model_kwargs={
                        "response_format": {
                            "type": "json_object"
                        }
                    },
                )
            except Exception as exc:
                print(
                    "Critic LLM initialization failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                self.llm = None

    # ============================================================
    # BASIC HELPERS
    # ============================================================

    @staticmethod
    def _flatten(value: Any) -> str:
        if isinstance(value, dict):
            return " ".join(
                CriticAgent._flatten(v)
                for v in value.values()
            )

        if isinstance(value, list):
            return " ".join(
                CriticAgent._flatten(v)
                for v in value
            )

        return str(value)

    @staticmethod
    def _evidence_text(
        item: dict[str, Any],
    ) -> str:
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
    def _evidence_path(
        item: dict[str, Any],
    ) -> str:
        metadata = item.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        return str(
            metadata.get(
                "path",
                item.get(
                    "path",
                    "",
                ),
            )
        ).replace(
            "\\",
            "/",
        )

    @staticmethod
    def _extract_evidence_ids(
        text: str,
    ) -> set[str]:
        pattern = re.compile(
            r"\bEV-[A-Z0-9][A-Z0-9_-]*\b",
            flags=re.IGNORECASE,
        )

        return {
            match.group(0).upper()
            for match in pattern.finditer(
                str(text)
            )
        }

    # ============================================================
    # SOURCE CLASSIFICATION
    # ============================================================

    @classmethod
    def _is_documentation(
        cls,
        path: str,
    ) -> bool:
        suffix = os.path.splitext(
            path.lower()
        )[1]

        name = os.path.basename(
            path.lower()
        )

        return (
            suffix
            in cls.DOCUMENTATION_EXTENSIONS
            or name
            in {
                "readme",
                "readme.md",
                "readme.txt",
            }
        )

    @classmethod
    def _is_implementation(
        cls,
        path: str,
    ) -> bool:
        suffix = os.path.splitext(
            path.lower()
        )[1]

        return (
            suffix
            in cls.IMPLEMENTATION_EXTENSIONS
        )

    @classmethod
    def _is_platform_boilerplate(
        cls,
        path: str,
    ) -> bool:
        normalized = (
            path.lower()
            .replace(
                "\\",
                "/",
            )
            .lstrip("./")
        )

        name = os.path.basename(
            normalized
        )

        # Exact generated runner/scaffolding files.
        if name in cls.PLATFORM_FILENAMES:
            return True

        # Platform directories are generally scaffolding.
        # Application-specific Dart/Java/Kotlin/Swift source
        # is allowed through.
        platform_prefixes = tuple(
            cls.PLATFORM_PATH_TERMS
        )

        if normalized.startswith(
            platform_prefixes
        ):
            application_extensions = {
                ".dart",
                ".java",
                ".kt",
                ".kts",
                ".swift",
                ".py",
                ".js",
                ".ts",
            }

            suffix = Path(
                normalized
            ).suffix.lower()

            if suffix not in application_extensions:
                return True

            # Known runner files remain boilerplate.
            if name in {
                "mainactivity.kt",
                "appdelegate.swift",
                "main.dart",
            }:
                return name != "main.dart"

        # Flutter web generated shell.
        if normalized.startswith(
            "web/"
        ) and name in {
            "index.html",
            "manifest.json",
        }:
            return True

        return False

    # ============================================================
    # EVIDENCE INDEX
    # ============================================================

    @classmethod
    def _build_evidence_index(
        cls,
        evidence: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:

        index: dict[
            str,
            dict[str, Any],
        ] = {}

        for item in evidence:

            evidence_id = str(
                item.get(
                    "id",
                    item.get(
                        "metadata",
                        {},
                    ).get(
                        "id",
                        "",
                    ),
                )
            ).strip().upper()

            if not evidence_id:
                continue

            path = cls._evidence_path(
                item
            )

            index[
                evidence_id
            ] = {
                "id": evidence_id,
                "path": path,
                "text": cls._evidence_text(
                    item
                ),
                "documentation": (
                    cls._is_documentation(
                        path
                    )
                ),
                "implementation": (
                    cls._is_implementation(
                        path
                    )
                ),
                "platform": (
                    cls._is_platform_boilerplate(
                        path
                    )
                ),
            }

        return index

    # ============================================================
    # SECTION HELPERS
    # ============================================================

    @classmethod
    def _section_citations(
        cls,
        value: Any,
    ) -> set[str]:
        return cls._extract_evidence_ids(
            cls._flatten(
                value
            )
        )

    @classmethod
    def _section_is_undetermined(
        cls,
        value: Any,
    ) -> bool:
        text = cls._flatten(
            value
        ).lower()

        return (
            NOT_DETERMINED.lower()
            in text
        )

    # ============================================================
    # DETERMINISTIC VALIDATION
    # ============================================================

    def validate(
        self,
        blueprint: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:

        evidence_index = (
            self._build_evidence_index(
                evidence
            )
        )

        evidence_ids = set(
            evidence_index
        )

        issues: list[
            dict[str, Any]
        ] = []

        warnings: list[
            dict[str, Any]
        ] = []

        # --------------------------------------------------------
        # Schema
        # --------------------------------------------------------

        missing = sorted(
            self.REQUIRED_KEYS
            - set(
                blueprint
            )
        )

        if missing:
            issues.append(
                {
                    "description": (
                        "Missing blueprint keys: "
                        + ", ".join(
                            missing
                        )
                    ),
                    "evidence_ids": [],
                }
            )

        architecture = blueprint.get(
            "architecture",
            {},
        )

        if not isinstance(
            architecture,
            dict,
        ):

            issues.append(
                {
                    "description": (
                        "Architecture must be an object."
                    ),
                    "evidence_ids": [],
                }
            )

        else:

            if not architecture.get(
                "components"
            ):
                issues.append(
                    {
                        "description": (
                            "Architecture has no components."
                        ),
                        "evidence_ids": [],
                    }
                )

            if not architecture.get(
                "data_flow"
            ):
                warnings.append(
                    {
                        "description": (
                            "Architecture has no explicit "
                            "data flow."
                        ),
                        "evidence_ids": [],
                    }
                )

        requirements = blueprint.get(
            "functional_requirements",
            [],
        )

        if not requirements:
            warnings.append(
                {
                    "description": (
                        "No functional requirements "
                        "were extracted."
                    ),
                    "evidence_ids": [],
                }
            )

        # --------------------------------------------------------
        # Citation validity
        # --------------------------------------------------------

        blueprint_text = self._flatten(
            blueprint
        )

        cited_ids = (
            self._extract_evidence_ids(
                blueprint_text
            )
        )

        unknown_citations = sorted(
            cited_ids
            - evidence_ids
        )

        if unknown_citations:
            issues.append(
                {
                    "description": (
                        "Blueprint references "
                        "evidence IDs that were not "
                        "retrieved: "
                        + ", ".join(
                            unknown_citations
                        )
                    ),
                    "evidence_ids": (
                        unknown_citations
                    ),
                }
            )

        declared_ids = blueprint.get(
            "_allowed_evidence_ids",
            [],
        )

        if isinstance(
            declared_ids,
            list,
        ):

            invalid_declared = sorted(
                {
                    str(x).upper()
                    for x in declared_ids
                }
                - evidence_ids
            )

            if invalid_declared:
                issues.append(
                    {
                        "description": (
                            "Architect declared "
                            "unavailable evidence IDs: "
                            + ", ".join(
                                invalid_declared
                            )
                        ),
                        "evidence_ids": (
                            invalid_declared
                        ),
                    }
                )

        # --------------------------------------------------------
        # Evidence coverage
        # --------------------------------------------------------

        sections_with_support = 0
        meaningful_sections = 0

        for section in self.CONCRETE_SECTIONS:

            value = blueprint.get(
                section,
                "",
            )

            if not value:
                continue

            meaningful_sections += 1

            citations = (
                self._section_citations(
                    value
                )
            )

            if citations:
                sections_with_support += 1
                continue

            if self._section_is_undetermined(
                value
            ):
                sections_with_support += 1

        coverage = (
            sections_with_support
            / meaningful_sections
            if meaningful_sections
            else 1.0
        )

        if coverage < 0.50:
            issues.append(
                {
                    "description": (
                        "Low evidence coverage "
                        "across technical sections."
                    ),
                    "evidence_ids": [],
                }
            )

        elif coverage < 0.75:
            warnings.append(
                {
                    "description": (
                        "Some technical sections "
                        "have limited direct "
                        "evidence coverage."
                    ),
                    "evidence_ids": [],
                }
            )

        # --------------------------------------------------------
        # Source diversity
        # --------------------------------------------------------

        cited_source_paths: set[str] = set()

        cited_implementation = 0
        cited_documentation = 0

        for evidence_id in cited_ids:

            record = evidence_index.get(
                evidence_id
            )

            if not record:
                continue

            # IMPORTANT:
            # Platform boilerplate is ignored completely
            # for architecture-quality source diversity.
            if record["platform"]:
                continue

            path = record[
                "path"
            ]

            if path:
                cited_source_paths.add(
                    path
                )

            if record[
                "implementation"
            ]:
                cited_implementation += 1

            if record[
                "documentation"
            ]:
                cited_documentation += 1

        if (
            cited_ids
            and len(
                cited_source_paths
            ) == 1
            and cited_implementation > 0
        ):
            warnings.append(
                {
                    "description": (
                        "Blueprint relies on a "
                        "single application evidence source. "
                        "Use multiple independent source files "
                        "where available."
                    ),
                    "evidence_ids": sorted(
                        cited_ids
                    ),
                }
            )

        # --------------------------------------------------------
        # Documentation-only architecture claims
        # --------------------------------------------------------

        architecture_claims = self._flatten(
            blueprint.get(
                "architecture",
                {},
            )
        ).lower()

        documentation_only = (
            cited_documentation > 0
            and cited_implementation == 0
            and bool(
                architecture_claims
            )
        )

        if documentation_only:
            warnings.append(
                {
                    "description": (
                        "High-level architecture claims "
                        "are supported primarily by "
                        "documentation rather than "
                        "implementation evidence."
                    ),
                    "evidence_ids": sorted(
                        cited_ids
                    ),
                }
            )

        # --------------------------------------------------------
        # Risky technical claims
        # --------------------------------------------------------

        lower_text = blueprint_text.lower()

        for term in self.RISKY_TERMS:

            if term not in lower_text:
                continue

            # If the document explicitly says that the
            # information is unknown, this is safe.
            if (
                NOT_DETERMINED.lower()
                in lower_text
            ):
                continue

            term_support = []

            for evidence_id in cited_ids:

                record = evidence_index.get(
                    evidence_id
                )

                if not record:
                    continue

                if record["platform"]:
                    continue

                if term in record[
                    "text"
                ].lower():
                    term_support.append(
                        evidence_id
                    )

            if not term_support:
                warnings.append(
                    {
                        "description": (
                            f"Claim containing '{term}' "
                            "has no directly matching "
                            "retrieved evidence."
                        ),
                        "evidence_ids": [],
                    }
                )

        # --------------------------------------------------------
        # Platform boilerplate misuse
        # --------------------------------------------------------
        #
        # IMPORTANT CHANGE:
        # We DO NOT create a warning merely because a platform
        # citation exists.
        #
        # The Architect is allowed to retain platform evidence
        # for traceability. The critic only warns if the blueprint
        # explicitly turns platform scaffolding into an application
        # architecture claim.
        # --------------------------------------------------------

        platform_citations = []

        for evidence_id in cited_ids:

            record = evidence_index.get(
                evidence_id
            )

            if (
                record
                and record["platform"]
            ):
                platform_citations.append(
                    evidence_id
                )

        platform_architecture_terms = (
            "business logic",
            "application architecture",
            "core architecture",
            "application service",
            "domain service",
            "data layer",
            "backend implementation",
        )

        platform_misuse = []

        if platform_citations:

            architecture_text = (
                self._flatten(
                    blueprint.get(
                        "architecture",
                        {},
                    )
                ).lower()
            )

            if any(
                term in architecture_text
                for term in platform_architecture_terms
            ):
                # Only report if there are no implementation
                # citations supporting the same architecture.
                if cited_implementation == 0:
                    platform_misuse = (
                        platform_citations
                    )

        if platform_misuse:
            warnings.append(
                {
                    "description": (
                        "Platform scaffolding appears to be "
                        "used as application architecture "
                        "without implementation support."
                    ),
                    "evidence_ids": sorted(
                        set(
                            platform_misuse
                        )
                    ),
                }
            )

        # --------------------------------------------------------
        # Generic fallback detection
        # --------------------------------------------------------

        generic_phrases = [
            "application components identified",
            "responsibilities are derived",
            "implementation details are derived",
            "document the user-visible functionality",
        ]

        generic_count = sum(
            1
            for phrase in generic_phrases
            if phrase in lower_text
        )

        if generic_count >= 3:
            warnings.append(
                {
                    "description": (
                        "Blueprint contains several "
                        "generic fallback statements "
                        "instead of concrete project-specific "
                        "technical findings."
                    ),
                    "evidence_ids": sorted(
                        cited_ids
                    ),
                }
            )

        # --------------------------------------------------------
        # Score
        # --------------------------------------------------------

        score = 10.0

        score -= min(
            4.0,
            1.5 * len(
                issues
            ),
        )

        score -= min(
            2.0,
            0.20 * len(
                warnings
            ),
        )

        if len(
            cited_source_paths
        ) >= 4:
            score += 0.2

        elif len(
            cited_source_paths
        ) >= 2:
            score += 0.1

        score = max(
            0.0,
            min(
                10.0,
                round(
                    score,
                    1,
                ),
            ),
        )

        return {
            "passed": not issues,
            "score": score,
            "issues": issues,
            "warnings": warnings,
            "evidence_ids": sorted(
                evidence_ids
            ),
            "evidence_coverage": round(
                coverage,
                2,
            ),
            "cited_source_count": len(
                cited_source_paths
            ),
            "cited_implementation_count": (
                cited_implementation
            ),
            "cited_documentation_count": (
                cited_documentation
            ),
        }

    # ============================================================
    # JSON PARSING
    # ============================================================

    @staticmethod
    def _parse_json_response(
        content: str,
    ) -> dict[str, Any]:

        text = str(
            content
        ).strip()

        if text.startswith(
            "```"
        ):
            text = re.sub(
                r"^```(?:json)?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )

            text = re.sub(
                r"\s*```$",
                "",
                text,
            )

        try:
            parsed = json.loads(
                text
            )

            if isinstance(
                parsed,
                dict,
            ):
                return parsed

        except json.JSONDecodeError:
            pass

        start = text.find("{")

        if start < 0:
            raise ValueError(
                "LLM critic returned no JSON object."
            )

        depth = 0
        in_string = False
        escaped = False

        for index in range(
            start,
            len(text),
        ):

            char = text[index]

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
                    candidate = text[
                        start:index + 1
                    ]

                    parsed = json.loads(
                        candidate
                    )

                    if isinstance(
                        parsed,
                        dict,
                    ):
                        return parsed

                    break

        raise ValueError(
            "LLM critic returned incomplete JSON."
        )

    # ============================================================
    # OPTIONAL LLM REVIEW
    # ============================================================

    def review_with_llm(
        self,
        blueprint: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Optional semantic review.

        LLM failure never creates a false quality failure.
        """

        if self.llm is None:
            return {
                "enabled": False,
                "available": False,
                "issues": [],
                "warnings": [
                    "LLM critic unavailable; "
                    "deterministic quality checks were used."
                ],
                "suggested_repairs": [],
            }

        # Keep prompt intentionally small.
        # This prevents TPM errors on free Groq tiers.
        compact_evidence = []

        for item in evidence[:8]:

            evidence_id = str(
                item.get(
                    "id",
                    "",
                )
            )

            path = self._evidence_path(
                item
            )

            if self._is_platform_boilerplate(
                path
            ):
                continue

            compact_evidence.append(
                (
                    evidence_id
                    + "\n"
                    + path
                    + "\n"
                    + self._evidence_text(
                        item
                    )[:300]
                )
            )

        compact_evidence_text = (
            "\n\n".join(
                compact_evidence
            )
        )

        compact_blueprint = json.dumps(
            blueprint,
            ensure_ascii=False,
        )[:3500]

        prompt = f"""
You are a strict software architecture reviewer.

Review the blueprint ONLY against supplied evidence.

Rules:
- Never invent facts.
- Never invent evidence IDs.
- Distinguish README claims from implementation.
- Ignore Flutter/Android/iOS/Linux/Windows/macOS boilerplate.
- Do not treat generated runner files as application architecture.
- Identify unsupported claims.
- Identify contradictions.
- Identify important missing evidence.
- "Not determined from available project evidence." is acceptable.
- Return ONLY valid JSON.
- Do not use Markdown fences.

Return:

{{
  "issues": [],
  "warnings": [],
  "suggested_repairs": []
}}

BLUEPRINT:
{compact_blueprint}

EVIDENCE:
{compact_evidence_text}
"""

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
            ).strip()

            parsed = (
                self._parse_json_response(
                    content
                )
            )

            issues = parsed.get(
                "issues",
                [],
            )

            warnings = parsed.get(
                "warnings",
                [],
            )

            repairs = parsed.get(
                "suggested_repairs",
                [],
            )

            if not isinstance(
                issues,
                list,
            ):
                issues = []

            if not isinstance(
                warnings,
                list,
            ):
                warnings = []

            if not isinstance(
                repairs,
                list,
            ):
                repairs = []

            return {
                "enabled": True,
                "available": True,
                "issues": [
                    str(x)
                    for x in issues
                ],
                "warnings": [
                    str(x)
                    for x in warnings
                ],
                "suggested_repairs": [
                    str(x)
                    for x in repairs
                ],
            }

        except Exception as exc:
            
            print(
               "CRITIC LLM ERROR:",
               type(exc).__name__,
               str(exc),
            )

            return {
                "enabled": True,
                "available": False,
                "issues": [],
                "warnings": [
                    "LLM critic unavailable; "
                    "deterministic quality checks were used."
                ],
                "suggested_repairs": [],
                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            }

    # ============================================================
    # COMBINED REVIEW
    # ============================================================

    def review(
        self,
        blueprint: dict[str, Any],
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:

        deterministic = self.validate(
            blueprint,
            evidence,
        )

        llm = self.review_with_llm(
            blueprint,
            evidence,
        )

        issues = (
            list(
                deterministic.get(
                    "issues",
                    [],
                )
            )
            + list(
                llm.get(
                    "issues",
                    [],
                )
            )
        )

        warnings = (
            list(
                deterministic.get(
                    "warnings",
                    [],
                )
            )
            + list(
                llm.get(
                    "warnings",
                    [],
                )
            )
        )

        # --------------------------------------------------------
        # Deduplicate findings
        # --------------------------------------------------------

        unique_issues = []
        seen_issues = set()

        for item in issues:

            key = str(
                item
            )

            if key in seen_issues:
                continue

            seen_issues.add(
                key
            )

            unique_issues.append(
                item
            )

        unique_warnings = []
        seen_warnings = set()

        for item in warnings:

            key = str(
                item
            )

            if key in seen_warnings:
                continue

            seen_warnings.add(
                key
            )

            unique_warnings.append(
                item
            )

        # --------------------------------------------------------
        # Final score
        # --------------------------------------------------------

        score = float(
            deterministic.get(
                "score",
                0.0,
            )
        )

        llm_issue_count = len(
            llm.get(
                "issues",
                [],
            )
        )

        llm_warning_count = len(
            llm.get(
                "warnings",
                [],
            )
        )

        score -= min(
            2.0,
            1.0 * llm_issue_count,
        )

        score -= min(
            1.0,
            0.15 * llm_warning_count,
        )

        score = max(
            0.0,
            min(
                10.0,
                round(
                    score,
                    1,
                ),
            ),
        )

        return {
            "passed": not unique_issues,
            "score": score,
            "issues": unique_issues,
            "warnings": unique_warnings,
            "suggested_repairs": llm.get(
                "suggested_repairs",
                [],
            ),
            "evidence_ids": deterministic.get(
                "evidence_ids",
                [],
            ),
            "evidence_coverage": deterministic.get(
                "evidence_coverage",
                0.0,
            ),
            "cited_source_count": deterministic.get(
                "cited_source_count",
                0,
            ),
            "cited_implementation_count": deterministic.get(
                "cited_implementation_count",
                0,
            ),
            "cited_documentation_count": deterministic.get(
                "cited_documentation_count",
                0,
            ),
            "llm_enabled": llm.get(
                "enabled",
                False,
            ),
            "llm_available": llm.get(
                "available",
                False,
            ),
        }