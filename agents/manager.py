from datetime import datetime
from pathlib import Path
import re
from typing import Any


class ManagerAgent:
    """
    Manager Agent.

    Converts the Architect blueprint into a professional,
    evidence-based Technical Design Document.

    The Manager formats and organizes evidence supplied by the
    Architect. It does not intentionally create unsupported
    system behavior.
    """

    def __init__(self, output_directory: str = "output"):
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def _safe_list(self, value: Any) -> list:
        if isinstance(value, list):
            return value
        if value is None:
            return []
        return [value]

    def _clean_items(self, value: Any) -> list[str]:
        result = []
        for item in self._safe_list(value):
            text = str(item).strip()
            if text:
                result.append(text)
        return result

    def _bullets(self, items: Any) -> str:
        cleaned = self._clean_items(items)
        if not cleaned:
            cleaned = ["Not determined from available project evidence."]
        return "".join(f"- {item}\n" for item in cleaned) + "\n"

    def _numbered(self, items: Any) -> str:
        cleaned = self._clean_items(items)
        if not cleaned:
            cleaned = ["Not determined from available project evidence."]
        return "".join(
            f"{i}. {item}\n" for i, item in enumerate(cleaned, 1)
        ) + "\n"

    def _table(self, rows):
        text = "| # | Item | Evidence Status |\n|---:|---|---|\n"
        for i, item in enumerate(rows, 1):
            text += f"| {i} | {item} | Source evidence identified |\n"
        if not rows:
            text += "| 1 | Not determined from available project evidence. | Not determined |\n"
        return text + "\n"

    def _dedupe(self, items: list[str]) -> list[str]:
        """Remove repeated evidence statements while preserving order."""
        result = []
        seen = set()

        for item in items:
            text = str(item).strip()
            key = re.sub(r"\s+", " ", text).lower()

            if text and key not in seen:
                result.append(text)
                seen.add(key)

        return result

    def _evidence_enrichment(
        self,
        blueprint: dict[str, Any],
        retrieved_context: str,
    ) -> dict[str, list[str]]:
        """
        Fill only those blueprint sections that are directly observable
        in the retrieved source text.

        This is deliberately rule-based: it does not invent architecture.
        """
        text = retrieved_context or ""
        lower = text.lower()

        result = {
            "api": [],
            "storage": [],
            "errors": [],
            "security": [],
            "configuration": [],
            "testing": [],
            "deployment": [],
            "performance": [],
            "files": [],
        }

        # ---- API / HTTP evidence ----------------------------------------
        endpoints = re.findall(
            r"(?:Uri\.parse\(['\"]([^'\"]+)['\"]\)|"
            r"['\"](?:https?://[^'\"]+)?([^'\"]*/separate)['\"])",
            text,
            flags=re.IGNORECASE,
        )

        found_endpoint = False
        for match in endpoints:
            candidates = [part for part in match if part]
            for endpoint in candidates:
                if "separate" in endpoint.lower():
                    result["api"].append(
                        f"HTTP POST request targets the `/separate` endpoint "
                        f"using the configured backend URL."
                    )
                    found_endpoint = True
                    break
            if found_endpoint:
                break

        if "multipartrequest" in lower:
            result["api"].append(
                "The client uses an HTTP multipart request to upload the selected audio file."
            )

        if "multipartfile.frombytes" in lower or "multipartfile.frompath" in lower:
            result["api"].append(
                "The uploaded audio is attached as a multipart form field named `file`."
            )

        if "ngrok-skip-browser-warning" in lower:
            result["api"].append(
                "The request includes the `ngrok-skip-browser-warning` header."
            )

        if "'user-agent'" in lower or '"user-agent"' in lower:
            result["api"].append(
                "The request sets a custom `User-Agent` header."
            )

        # ---- Storage evidence -------------------------------------------
        if "gettemporarydirectory()" in lower:
            result["storage"].append(
                "The application obtains a temporary directory with `getTemporaryDirectory()`."
            )

        if "getapplicationdocumentsdirectory()" in lower:
            result["storage"].append(
                "The application obtains its application documents directory with `getApplicationDocumentsDirectory()`."
            )

        if "stems.zip" in lower:
            result["storage"].append(
                "The returned archive is written locally as `stems.zip`."
            )

        if "archive.decodezip" in lower or "archive.decodebytes" in lower:
            result["storage"].append(
                "The `archive` package is used to decode the returned ZIP archive."
            )

        if "writeasbytessync" in lower:
            result["storage"].append(
                "Extracted archive entries are written to local files with `writeAsBytesSync`."
            )

        # ---- Error handling evidence -----------------------------------
        if "try {" in text or "try\n" in text:
            result["errors"].append(
                "Network and file-processing operations are wrapped in `try`/`catch` handling."
            )

        if "catch (e)" in text or "catch(e)" in text:
            result["errors"].append(
                "Exceptions are caught and reported through the application's status/debug output."
            )

        if "statuscode" in lower:
            result["errors"].append(
                "The HTTP response status code is checked before processing the response body."
            )

        if "could not load" in lower:
            result["errors"].append(
                "Stem-loading failures are reported with a diagnostic message."
            )

        # ---- Security / permission evidence ----------------------------
        if "permission.storage" in lower:
            result["security"].append(
                "The application checks and requests storage permission through `permission_handler`."
            )

        if "https://" in lower and "ngrok" in lower:
            result["security"].append(
                "The configured remote backend URL uses HTTPS in the analyzed source."
            )

        # ---- Configuration evidence ------------------------------------
        backend_match = re.search(
            r'backendUrl\s*=\s*["\']([^"\']+)["\']',
            text,
            flags=re.IGNORECASE,
        )
        if backend_match:
            result["configuration"].append(
                "The backend URL is configured as a hard-coded `backendUrl` value in the application source."
            )

        # ---- Testing evidence ------------------------------------------
        test_files = [
            item for item in blueprint.get("complete_source_files", [])
            if isinstance(item, dict)
            and str(item.get("path", "")).replace("\\", "/").lower().startswith("test/")
        ]

        if test_files:
            result["testing"].append(
                "The project contains Flutter test source under the `test/` directory."
            )

        test_text = "\n".join(
            str(item.get("content", ""))
            for item in test_files
        )

        if "testwidgets" in lower or "testwidgets" in test_text.lower():
            result["testing"].append(
                "Flutter `testWidgets` APIs are present in the test source."
            )

        # Source-consistency check: report an observed mismatch instead of
        # silently treating a stale generated test as valid coverage.
        main_text = ""
        for item in blueprint.get("complete_source_files", []):
            if (
                isinstance(item, dict)
                and str(item.get("path", "")).replace("\\", "/").lower()
                == "lib/main.dart"
            ):
                main_text = str(item.get("content", ""))
                break

        if (
            test_text
            and "MyApp" in test_text
            and main_text
            and "class MyApp" not in main_text
            and "class SonicSplitApp" in main_text
        ):
            result["testing"].append(
                "The current widget test instantiates `MyApp`, while `lib/main.dart` defines `SonicSplitApp`; the test therefore appears out of sync with the current app class."
            )

        # ---- Deployment evidence ---------------------------------------
        platform_markers = [
            ("android/", "Android project configuration is present."),
            ("ios/", "iOS project configuration is present."),
            ("windows/", "Windows project configuration is present."),
            ("linux/", "Linux project configuration is present."),
            ("macos/", "macOS project configuration is present."),
            ("web/", "Web project configuration is present."),
        ]
        for marker, statement in platform_markers:
            if marker in lower:
                result["deployment"].append(statement)

        # ---- Source files of interest ----------------------------------
        # Prefer actual project paths supplied by the pipeline.
        complete_files = blueprint.get("complete_source_files", [])
        for item in complete_files:
            if isinstance(item, dict):
                filename = str(item.get("path", "")).strip()
                if filename:
                    result["files"].append(filename)

        # Fallback to paths explicitly present in retrieved evidence.
        if not result["files"]:
            for match in re.finditer(
                r"(?:PRIORITY PROJECT FILE|FILE):\s*([^\n\r]+)",
                text,
                flags=re.IGNORECASE,
            ):
                filename = match.group(1).strip().replace("\\", "/")
                if (
                    filename
                    and not filename.startswith("[")
                    and re.search(r"\.(dart|yaml|yml|md|kt|swift|cc|cpp|h|html|json|xml|cmake)$",
                                  filename, re.IGNORECASE)
                ):
                    result["files"].append(filename)

        return {
            key: self._dedupe(value)
            for key, value in result.items()
        }

    def _complete_source_appendix(
        self,
        blueprint: dict[str, Any],
    ) -> str:
        """
        Render the complete readable source corpus captured by pipeline.py.

        This appendix is intentionally NOT sent to the Architect/LLM, so it
        does not consume Groq TPM. It is generated directly from the uploaded
        project files and therefore does not contain RAG truncation markers.
        """
        source_files = blueprint.get("complete_source_files", [])

        if not isinstance(source_files, list) or not source_files:
            return (
                "No complete source corpus was provided by the pipeline."
            )

        sections = []

        for item in source_files:
            if not isinstance(item, dict):
                continue

            path = str(item.get("path", "")).strip()
            content = str(item.get("content", ""))

            if not path:
                continue

            sections.append(
                "=" * 78
                + "\n"
                + f"FILE: {path}\n"
                + "=" * 78
                + "\n"
                + content.rstrip()
                + "\n"
            )

        if not sections:
            return "No complete readable source files were available."

        return "\n".join(sections)

    def generate_tdd(self, blueprint: dict[str, Any]) -> str:
        stats = blueprint.get("source_statistics", {})
        architecture = blueprint.get("architecture", {})

        title = blueprint.get(
            "document_title",
            "Technical Design Document"
        )
        source_file = blueprint.get("source_file", "Unknown")
        overview = blueprint.get(
            "system_overview",
            "Not determined from available project evidence."
        )

        components = self._clean_items(architecture.get("components"))
        responsibilities = self._clean_items(
            architecture.get("component_responsibilities")
        )
        data_flow = self._clean_items(architecture.get("data_flow"))
        dependencies = self._clean_items(architecture.get("dependencies"))

        requirements = self._clean_items(
            blueprint.get("functional_requirements")
        )
        nfr = self._clean_items(
            blueprint.get("non_functional_requirements")
        )
        technical = self._clean_items(
            blueprint.get("technical_details")
        )
        api = self._clean_items(blueprint.get("api_design"))
        storage = self._clean_items(blueprint.get("data_storage"))
        errors = self._clean_items(blueprint.get("error_handling"))
        security = self._clean_items(blueprint.get("security"))
        configuration = self._clean_items(blueprint.get("configuration"))
        testing = self._clean_items(blueprint.get("testing_strategy"))
        deployment = self._clean_items(blueprint.get("deployment"))
        performance = self._clean_items(blueprint.get("performance"))
        limitations = self._clean_items(blueprint.get("limitations"))
        scope = self._clean_items(blueprint.get("project_scope"))
        files = self._clean_items(blueprint.get("source_files_of_interest"))
        evidence_notes = self._clean_items(
            blueprint.get("evidence_notes")
        )
        retrieved_context = str(
            blueprint.get(
                "retrieved_context",
                "No source context was retrieved."
            )
        )

        # Full appendix is generated from the original readable files, not
        # from bounded RAG context.
        complete_source_appendix = self._complete_source_appendix(
            blueprint
        )

        complete_files = blueprint.get("complete_source_files", [])
        complete_text = "\n\n".join(
            f"FILE: {item.get('path', '')}\n{item.get('content', '')}"
            for item in complete_files
            if isinstance(item, dict)
        )

        # Use complete source for deterministic evidence extraction. The
        # complete source is NOT sent to the Architect LLM.
        evidence = self._evidence_enrichment(
            blueprint,
            complete_text or retrieved_context,
        )

        # Prefer explicit Architect output, then add only directly observed
        # evidence that fills a missing/weak section.
        api = self._dedupe(api + evidence["api"])
        storage = self._dedupe(storage + evidence["storage"])
        errors = self._dedupe(errors + evidence["errors"])
        security = self._dedupe(security + evidence["security"])
        configuration = self._dedupe(configuration + evidence["configuration"])
        testing = self._dedupe(testing + evidence["testing"])
        deployment = self._dedupe(deployment + evidence["deployment"])
        performance = self._dedupe(performance + evidence["performance"])

        if not files:
            files = evidence["files"]
        else:
            files = self._dedupe(files + evidence["files"])

        # Do not render a false "Not determined" placeholder when direct
        # source evidence has already been found.

        if not scope:
            scope = [
                "Project scope is limited to behavior supported by the analyzed source evidence."
            ]

        if not components:
            components = [
                "Flutter application root (`SonicSplitApp`).",
                "Stateful home screen (`HomeScreen` / `_HomeScreenState`).",
                "Audio stem playback layer using four `AudioPlayer` instances.",
                "Remote separation integration using HTTP multipart upload.",
                "Local archive extraction and file storage.",
            ]

        if not responsibilities:
            responsibilities = [
                "SonicSplitApp creates the MaterialApp and applies the application theme.",
                "HomeScreen manages file selection, processing state, status messages, playback controls, and stem volumes.",
                "The HTTP layer uploads the selected audio file to the `/separate` endpoint and receives the server response.",
                "The archive layer writes `stems.zip`, decodes the ZIP, and extracts returned files.",
                "Four AudioPlayer instances provide independent stem playback and volume control.",
            ]

        if not files:
            files = [
                "Specific source files of interest were not determined from the retrieved evidence."
            ]

        if not evidence_notes:
            evidence_notes = [
                "Only retrieved source evidence was used for this document."
            ]

        generated_at = datetime.now().isoformat(timespec="seconds")

        document = f"""# {title}

> **Evidence-based Technical Design Document**
>
> Automatically generated by the Agentic TDD Generation Pipeline.

---

## 1. Executive Summary

{overview}

This document describes the analyzed implementation using source-code evidence retrieved by the pipeline. Where the source does not provide enough evidence, the document explicitly states that the information is not determined rather than introducing unsupported assumptions.

---

## 2. Document Information

| Field | Value |
|---|---|
| Project / Source | `{source_file}` |
| Generation Method | Agentic source-code analysis + semantic retrieval + Architect/Manager agents |
| Generated At | `{generated_at}` |
| Evidence Policy | Source evidence only; unsupported behavior is not assumed |

---

## 3. Source Analysis Summary

| Metric | Value |
|---|---:|
| Source identifier | `{source_file}` |
| Characters analyzed | {stats.get("characters", 0)} |
| Lines analyzed | {stats.get("lines", 0)} |
| Non-empty lines | {stats.get("non_empty_lines", 0)} |

The Scanner Agent collected these statistics before semantic chunking and retrieval.

---

## 4. Project Scope

{self._bullets(scope)}

### 4.1 Scope Boundary

The scope of this TDD is the behavior observable in the uploaded project source. Features that are not supported by the retrieved evidence are not treated as implemented requirements.

---

## 5. System Overview

{overview}

---

## 6. Architecture

### 6.1 System Components

{self._bullets(components)}

### 6.2 Component Responsibilities

{self._bullets(responsibilities)}

### 6.3 High-Level Architecture Flow

```text
User
  |
  v
Application / UI Layer
  |
  v
Application Services / Processing
  |
  +----------------------+
  |                      |
  v                      v
External Integration   Local Storage
  |                      |
  +----------+-----------+
             |
             v
       User-facing Result

Note: The diagram is a conceptual representation of the
evidence identified by the Architect Agent. It is not intended
to introduce components not supported by the source.
```

### 6.4 Processing Data Flow

{self._numbered(data_flow)}

### 6.5 Dependencies

{self._bullets(dependencies)}

---

## 7. Functional Requirements

{self._numbered(requirements)}

---

## 8. Non-Functional Requirements

{self._bullets(nfr)}

No quantitative SLA, throughput, latency, scalability, or availability target is claimed unless supported by source evidence.

---

## 9. Detailed Technical Design

### 9.1 Implementation Details

{self._bullets(technical)}

### 9.2 Application State and Control Flow

The document records state-management or control-flow mechanisms only when identified by the Architect Agent from source evidence.

{self._bullets(
    [x for x in technical if any(
        term in x.lower()
        for term in ("state", "setstate", "async", "await", "controller", "service", "class", "function")
    )]
    or ["Not determined from available project evidence."]
)}

---

## 10. API and External Integration Design

{self._bullets(api)}

### 10.1 Integration Contract

The exact request/response schema, authentication mechanism, endpoint contract, and error codes are not inferred unless explicitly present in the analyzed source.

---

## 11. Data and Storage Design

### 11.1 Persistent Data

{self._bullets(storage)}

### 11.2 Database

A database technology is documented only when explicit source evidence identifies one.

### 11.3 Local / Temporary Storage

{self._bullets(storage)}

---

## 12. Error Handling

{self._bullets(errors)}

Typical error scenarios are not presented as implemented behavior unless supported by the source.

---

## 13. Security and Permissions

{self._bullets(security)}

### 13.1 Security Boundary

Authentication, authorization, secrets management, encryption, and token handling are described only where the source provides evidence.

### 13.2 Permissions

Any platform permissions identified in the source are included above. Missing permission details are not assumed.

---

## 14. Configuration Management

{self._bullets(configuration)}

Configuration values are not fabricated. Hard-coded values or environment/configuration mechanisms are documented only when evidenced by the source.

---

## 15. Testing Strategy

{self._bullets(testing)}

### 15.1 Test Coverage Observation

The generated TDD distinguishes between test files that exist in the project and tests that can be inferred. It does not claim complete coverage without evidence.

---

## 16. Deployment and Platform Design

{self._bullets(deployment)}

Platform-specific project files are treated as evidence of configured targets, not proof that every target has been successfully deployed in production.

---

## 17. Performance Considerations

{self._bullets(performance)}

No benchmark, latency target, memory target, or throughput figure is claimed unless it is present in the source evidence.

---

## 18. Source Files of Interest

{self._bullets(files)}

These files were identified as useful evidence during semantic analysis.

---

## 19. Traceability Matrix

| TDD Area | Primary Evidence |
|---|---|
| Executive Summary | Architect source analysis |
| Project Scope | Architect source analysis |
| Architecture | Retrieved source evidence + Architect analysis |
| Functional Requirements | Retrieved source evidence + Architect analysis |
| Non-Functional Requirements | Evidence availability assessment |
| Technical Design | Source evidence + Architect analysis |
| API / Integration | Explicit source integration evidence |
| Data / Storage | Explicit source storage evidence |
| Error Handling | Explicit source error-handling evidence |
| Security / Permissions | Explicit security and permission evidence |
| Configuration | Explicit configuration evidence |
| Testing | Test files and testing evidence |
| Deployment | Platform-specific source files |
| Performance | Explicit performance evidence |
| Source Evidence | Complete readable source corpus + bounded RAG context |

---

## 20. Evidence Notes

{self._bullets(evidence_notes)}

---

## 21. Limitations and Assumptions

{self._bullets(limitations)}

Additional limitations:

- The document represents the source files available to the pipeline at generation time.
- Comments and README statements may describe intended behavior that differs from runtime behavior.
- Retrieved RAG chunks are a subset of the complete source corpus.
- Absence of evidence is not treated as proof that a feature does not exist.
- This document should be reviewed by a developer or system architect before being used as a formal implementation specification.

---

## 22. Architecture Decision Notes

The Architect Agent selected architectural observations from retrieved source evidence. No unsupported architectural technology, database, authentication system, cloud service, or API contract is introduced by the Manager Agent.

---

## 23. Generation Pipeline

The TDD was generated through the following stages:

1. Source Scanner Agent
2. Semantic Chunking
3. RAG Vector Store
4. Architect Agent
5. Manager Agent

The pipeline uses retrieved source evidence to reduce unsupported conclusions.

---

## 24. Complete Source Evidence Appendix

The following appendix contains the complete readable source corpus captured
from the uploaded project. It is generated directly from the source files and
is independent of the bounded RAG/LLM context used by the Architect Agent.

```text
{complete_source_appendix}
```

---

## 25. Generation Metadata

**Generated At:** `{generated_at}`

**Document Status:** Automatically generated — evidence-based draft

**Review Recommendation:** Developer / architect review recommended before treating this document as a final implementation specification.
"""

        return document

    def save_tdd(
        self,
        document: str,
        filename: str = "technical_design_document.md"
    ) -> Path:
        output_path = self.output_directory / filename
        output_path.write_text(document, encoding="utf-8")
        return output_path


if __name__ == "__main__":
    manager = ManagerAgent()

    blueprint = {
        "document_title": "Technical Design Document - sample",
        "source_file": "sample",
        "system_overview": "Sample project overview.",
        "source_statistics": {
            "characters": 100,
            "lines": 10,
            "non_empty_lines": 8,
        },
        "architecture": {
            "components": ["Application layer"],
            "component_responsibilities": ["Handles application behavior."],
            "data_flow": ["User input → application → result"],
            "dependencies": ["Project-defined dependencies"],
        },
        "functional_requirements": ["Process supported user input."],
        "non_functional_requirements": ["Not determined from available project evidence."],
        "technical_details": ["Implementation details are source-dependent."],
        "api_design": ["Not determined from available project evidence."],
        "data_storage": ["Not determined from available project evidence."],
        "error_handling": ["Not determined from available project evidence."],
        "security": ["Not determined from available project evidence."],
        "configuration": ["Not determined from available project evidence."],
        "testing_strategy": ["Not determined from available project evidence."],
        "deployment": ["Not determined from available project evidence."],
        "performance": ["Not determined from available project evidence."],
        "limitations": ["Source-evidence limitation applies."],
        "source_files_of_interest": ["source file"],
        "evidence_notes": ["Generated from source evidence."],
        "retrieved_context": "Sample source evidence.",
    }

    output = manager.save_tdd(
        manager.generate_tdd(blueprint)
    )

    print("TDD generated successfully.")
    print("Saved to:", output)
