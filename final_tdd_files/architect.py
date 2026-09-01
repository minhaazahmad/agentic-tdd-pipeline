import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# ============================================================
# ENVIRONMENT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(
    ENV_FILE,
    override=True
)


# ============================================================
# ARCHITECT AGENT
# ============================================================

class ArchitectAgent:
    """
    Architect Agent

    Converts retrieved project source evidence into a
    structured TDD blueprint.

    Important:
    The agent must describe the uploaded project itself.
    It must not use sample.py, previous TDD files, or
    pipeline implementation as evidence for the target
    project's functionality.
    """

    def __init__(self):

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        self.llm = None

        if (
            api_key
            and api_key.startswith("gsk_")
        ):

            self.llm = ChatGroq(
                api_key=api_key,
                model=os.getenv(
                    "GROQ_MODEL",
                    "llama-3.1-8b-instant"
                ),
                temperature=0,
                max_tokens=1800,
            )


    # ========================================================
    # CLEAN PROJECT EVIDENCE
    # ========================================================

    def _clean_context(
        self,
        relevant_chunks
    ):

        MAX_CHUNK_CHARS = 4200
        MAX_CONTEXT_CHARS = 9500

        cleaned_chunks = []

        blocked_file_markers = [
            "sample.py",
            "technical_design_document.md",
            "agents\\architect.py",
            "agents/architect.py",
            "agents\\manager.py",
            "agents/manager.py",
            "pipeline.py",
            "scanner.py",
            "vector_store.py",
            "chunker.py",
            "app.py",
        ]

        for chunk in relevant_chunks:

            text = str(
                chunk
            ).strip()

            if not text:
                continue

            # ------------------------------------------------
            # Remove evidence belonging to the TDD pipeline
            # itself.
            # ------------------------------------------------

            lower_text = text.lower()

            should_skip = False

            for marker in blocked_file_markers:

                if marker.lower().replace("\\", "/") in lower_text:

                    should_skip = True
                    break

            if should_skip:
                continue

            # ------------------------------------------------
            # Keep chunks short.
            # ------------------------------------------------

            if len(text) > MAX_CHUNK_CHARS:

                text = (
                    text[:MAX_CHUNK_CHARS]
                    + "\n[chunk truncated]"
                )

            text = re.sub(
                r"\s*\[(?:chunk|project evidence|retrieved context) truncated\]",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()

            cleaned_chunks.append(text)

        # ----------------------------------------------------
        # Combine evidence.
        # ----------------------------------------------------

        context = "\n\n".join(
            cleaned_chunks
        )

        if len(context) > MAX_CONTEXT_CHARS:

            context = (
                context[:MAX_CONTEXT_CHARS]
                + "\n[project evidence truncated]"
            )

        if not context.strip():

            context = (
                "No clean project evidence was retrieved."
            )

        return context


    # ========================================================
    # FALLBACK BLUEPRINT
    # ========================================================

    def _fallback_blueprint(
        self,
        filename: str,
        scanner_data: dict[str, Any],
        context: str,
    ):

        context_lower = context.lower()

        components = []
        dependencies = []
        requirements = []
        technical_details = []
        data_flow = []

        # ----------------------------------------------------
        # Flutter
        # ----------------------------------------------------

        if (
            "flutter" in context_lower
            or "dart" in context_lower
            or "pubspec.yaml" in context_lower
        ):

            components.append(
                "Flutter application layer"
            )

            dependencies.append(
                "Flutter / Dart"
            )

            requirements.append(
                "Provide a cross-platform application "
                "interface using Flutter."
            )

        # ----------------------------------------------------
        # Audio
        # ----------------------------------------------------

        if (
            "audio" in context_lower
            or "music" in context_lower
            or "song" in context_lower
            or "vocal" in context_lower
            or "stem" in context_lower
        ):

            components.append(
                "Audio processing functionality"
            )

            requirements.append(
                "Process audio input according to "
                "the functionality implemented by the project."
            )

        # ----------------------------------------------------
        # File handling
        # ----------------------------------------------------

        if (
            "file" in context_lower
            or "upload" in context_lower
            or "download" in context_lower
        ):

            components.append(
                "File handling component"
            )

            requirements.append(
                "Allow the application to handle "
                "supported user files."
            )

        # ----------------------------------------------------
        # Audio separation
        # ----------------------------------------------------

        if (
            "separate" in context_lower
            or "separation" in context_lower
            or "stem separation" in context_lower
        ):

            components.append(
                "Audio separation component"
            )

            requirements.append(
                "Separate audio content into the "
                "supported output components."
            )

        # ----------------------------------------------------
        # FFmpeg
        # ----------------------------------------------------

        if "ffmpeg" in context_lower:

            components.append(
                "FFmpeg audio processing layer"
            )

            dependencies.append(
                "FFmpeg"
            )

            technical_details.append(
                "FFmpeg is used for audio-related "
                "processing where supported by the source."
            )

            data_flow.append(
                "Audio input → FFmpeg processing → audio output"
            )

        # ----------------------------------------------------
        # Demucs
        # ----------------------------------------------------

        if "demucs" in context_lower:

            components.append(
                "Demucs-based source separation"
            )

            dependencies.append(
                "Demucs"
            )

            technical_details.append(
                "Demucs is referenced for audio "
                "source separation."
            )

        # ----------------------------------------------------
        # Desktop
        # ----------------------------------------------------

        if (
            "windows" in context_lower
            or "linux" in context_lower
            or "macos" in context_lower
        ):

            components.append(
                "Cross-platform desktop platform layer"
            )

        # ----------------------------------------------------
        # Web
        # ----------------------------------------------------

        if (
            "web/index.html" in context_lower
            or "web application" in context_lower
        ):

            components.append(
                "Flutter web platform layer"
            )

        # ----------------------------------------------------
        # Main Dart
        # ----------------------------------------------------

        if "main.dart" in context_lower:

            technical_details.append(
                "The primary Flutter application entry "
                "point is implemented in Dart."
            )

            data_flow.append(
                "User interaction → Flutter UI → application processing"
            )

        # ----------------------------------------------------
        # Default values
        # ----------------------------------------------------

        if not components:

            components.append(
                "Application components identified "
                "from available project evidence."
            )

        if not requirements:

            requirements.append(
                "Functional requirements could not be "
                "fully determined from the available evidence."
            )

        if not dependencies:

            dependencies.append(
                "Dependencies are defined by the project "
                "configuration and source evidence."
            )

        if not technical_details:

            technical_details.append(
                "Technical implementation details are "
                "limited to the retrieved source evidence."
            )

        if not data_flow:

            data_flow.append(
                "Application flow could not be fully "
                "determined from the retrieved evidence."
            )

        # ----------------------------------------------------
        # Remove duplicates.
        # ----------------------------------------------------

        components = list(
            dict.fromkeys(
                components
            )
        )

        dependencies = list(
            dict.fromkeys(
                dependencies
            )
        )

        requirements = list(
            dict.fromkeys(
                requirements
            )
        )

        technical_details = list(
            dict.fromkeys(
                technical_details
            )
        )

        data_flow = list(
            dict.fromkeys(
                data_flow
            )
        )

        return {
            "document_title": (
                f"Technical Design Document - {filename}"
            ),

            "source_file": filename,

            "system_overview": (
                "The project was analyzed using source-code "
                "statistics, semantic chunking, RAG retrieval, "
                "and evidence-based architectural analysis."
            ),

            "source_statistics": {
                "characters": scanner_data.get(
                    "character_count",
                    0
                ),

                "lines": scanner_data.get(
                    "line_count",
                    0
                ),

                "non_empty_lines": scanner_data.get(
                    "non_empty_lines",
                    0
                ),
            },

            "architecture": {
                "components": components,
                "data_flow": data_flow,
                "dependencies": dependencies,
            },

            "functional_requirements": requirements,

            "technical_details": technical_details,

            "retrieved_context": context,
        }


    # ========================================================
    # BUILD DESIGN BLUEPRINT
    # ========================================================

    def build_design_blueprint(
        self,
        filename: str,
        scanner_data: dict[str, Any],
        relevant_chunks: list[str],
    ):

        # ----------------------------------------------------
        # CLEAN RAG EVIDENCE
        # ----------------------------------------------------

        context = self._clean_context(
            relevant_chunks
        )

        # ----------------------------------------------------
        # API KEY CHECK
        # ----------------------------------------------------

        api_key = os.getenv(
            "GROQ_API_KEY"
        )

        if not api_key:

            print(
                "GROQ_API_KEY not found."
            )

            print(
                "Using deterministic "
                "evidence-based analysis."
            )

            return self._fallback_blueprint(
                filename,
                scanner_data,
                context
            )

        if not api_key.startswith("gsk_"):

            print(
                "Invalid GROQ_API_KEY."
            )

            print(
                "Using deterministic "
                "evidence-based analysis."
            )

            return self._fallback_blueprint(
                filename,
                scanner_data,
                context
            )

        # ----------------------------------------------------
        # CREATE LLM
        # ----------------------------------------------------

        if self.llm is None:

            self.llm = ChatGroq(
                api_key=api_key,
                model=os.getenv(
                    "GROQ_MODEL",
                    "llama-3.1-8b-instant"
                ),
                temperature=0,
                max_tokens=1800,
            )

        # ----------------------------------------------------
        # ARCHITECT PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are the Architect Agent of an Agentic TDD Generation Pipeline.

Your task is to analyze the uploaded PROJECT itself.

IMPORTANT RULES:

1. Use ONLY the project evidence supplied below.
2. Do NOT use general assumptions.
3. Do NOT invent APIs, databases, authentication,
   backend services, or technologies.
4. Do NOT describe the TDD-generation pipeline.
5. Do NOT describe sample.py.
6. Do NOT describe technical_design_document.md.
7. Do NOT assume that the project has PostgreSQL,
   JWT authentication, REST APIs, or Python backend
   unless the supplied project evidence explicitly
   proves it.
8. Prefer concrete source files such as:
   main.dart, pubspec.yaml, README.md and actual
   application source files.
9. If something is not supported, say:
   "Not determined from available project evidence."
10. Keep the response concise.

PROJECT:
{filename}

SCANNER STATISTICS:

Characters:
{scanner_data.get("character_count", 0)}

Lines:
{scanner_data.get("line_count", 0)}

Non-empty lines:
{scanner_data.get("non_empty_lines", 0)}

PROJECT SOURCE EVIDENCE:

{context}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "system_overview": "string",

    "architecture": {{
        "components": [
            "component"
        ],

        "data_flow": [
            "step"
        ],

        "dependencies": [
            "dependency"
        ]
    }},

    "functional_requirements": [
        "requirement"
    ],

    "technical_details": [
        "technical detail"
    ]
}}

Keep each array concise.

Do not mention information that is not
supported by the supplied project evidence.
"""

        # ----------------------------------------------------
        # CALL GROQ
        # ----------------------------------------------------

        try:

            print(
                "Calling Groq Architect LLM..."
            )

            response = self.llm.invoke(
                prompt
            )

            content = response.content

            if not isinstance(
                content,
                str
            ):

                content = str(
                    content
                )

            content = content.strip()

            # ------------------------------------------------
            # Remove Markdown code fences.
            # ------------------------------------------------

            if content.startswith(
                "```"
            ):

                lines = (
                    content.splitlines()
                )

                if (
                    lines
                    and lines[0].startswith(
                        "```"
                    )
                ):

                    lines = lines[1:]

                if (
                    lines
                    and lines[-1].strip()
                    == "```"
                ):

                    lines = lines[:-1]

                content = "\n".join(
                    lines
                ).strip()

            # ------------------------------------------------
            # Parse JSON.
            # ------------------------------------------------

            result = json.loads(
                content
            )

            architecture = result.get(
                "architecture",
                {}
            )

            if not isinstance(
                architecture,
                dict
            ):

                architecture = {}

            components = architecture.get(
                "components",
                []
            )

            data_flow = architecture.get(
                "data_flow",
                []
            )

            dependencies = architecture.get(
                "dependencies",
                []
            )

            functional_requirements = (
                result.get(
                    "functional_requirements",
                    []
                )
            )

            technical_details = (
                result.get(
                    "technical_details",
                    []
                )
            )

            # ------------------------------------------------
            # Type safety.
            # ------------------------------------------------

            if not isinstance(
                components,
                list
            ):
                components = []

            if not isinstance(
                data_flow,
                list
            ):
                data_flow = []

            if not isinstance(
                dependencies,
                list
            ):
                dependencies = []

            if not isinstance(
                functional_requirements,
                list
            ):
                functional_requirements = []

            if not isinstance(
                technical_details,
                list
            ):
                technical_details = []

            # ------------------------------------------------
            # Final blueprint.
            # ------------------------------------------------

            return {
                "document_title": (
                    f"Technical Design Document - {filename}"
                ),

                "source_file": filename,

                "system_overview": result.get(
                    "system_overview",
                    "Not determined from available project evidence."
                ),

                "source_statistics": {
                    "characters": scanner_data.get(
                        "character_count",
                        0
                    ),

                    "lines": scanner_data.get(
                        "line_count",
                        0
                    ),

                    "non_empty_lines": scanner_data.get(
                        "non_empty_lines",
                        0
                    ),
                },

                "architecture": {
                    "components": components,
                    "data_flow": data_flow,
                    "dependencies": dependencies,
                },

                "functional_requirements": (
                    functional_requirements
                ),

                "technical_details": (
                    technical_details
                ),

                "retrieved_context": context,
            }

        # ----------------------------------------------------
        # AI FAILURE
        # ----------------------------------------------------

        except Exception as error:

            print()
            print(
                "Architect LLM analysis failed:"
            )

            print(
                error
            )

            print()

            print(
                "Using deterministic "
                "evidence-based analysis."
            )

            return self._fallback_blueprint(
                filename,
                scanner_data,
                context
            )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("ARCHITECT AGENT TEST")
    print("=" * 60)
    print()

    api_key = os.getenv(
        "GROQ_API_KEY",
        ""
    )

    print(
        "Environment file:",
        ENV_FILE
    )

    print(
        "Environment file exists:",
        ENV_FILE.exists()
    )

    print(
        "API key loaded:",
        bool(api_key)
    )

    print(
        "Model:",
        os.getenv(
            "GROQ_MODEL",
            "llama-3.1-8b-instant"
        )
    )

    test_chunks = [
        """
        FILE: lib/main.dart

        Flutter application with audio processing
        functionality and user interface.
        """,

        """
        FILE: pubspec.yaml

        Flutter project dependencies and packages.
        """,
    ]

    scanner_data = {
        "character_count": 70000,
        "line_count": 2200,
        "non_empty_lines": 1800,
    }

    architect = ArchitectAgent()

    blueprint = architect.build_design_blueprint(
        filename="SonicSplit-AI-main",
        scanner_data=scanner_data,
        relevant_chunks=test_chunks,
    )

    print()
    print("=" * 60)
    print("ARCHITECT OUTPUT")
    print("=" * 60)
    print()

    print(
        json.dumps(
            blueprint,
            indent=2,
            ensure_ascii=False
        )
    )