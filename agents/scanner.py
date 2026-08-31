from pathlib import Path
from typing import Any


class ScannerAgent:
    """
    Scanner Agent:
    Reads source/configuration files and extracts basic structural
    information before the Architect Agent processes them.
    """

    SUPPORTED_EXTENSIONS = {
        ".py", ".js", ".ts", ".java", ".cpp", ".c",
        ".json", ".yaml", ".yml", ".xml", ".conf",
        ".ini", ".txt", ".sql"
    }

    def scan_file(self, file_path: str) -> dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        content = path.read_text(encoding="utf-8", errors="ignore")

        return self.scan_content(
            content=content,
            filename=path.name
        )

    def scan_content(
        self,
        content: str,
        filename: str = "unknown"
    ) -> dict[str, Any]:

        lines = content.splitlines()

        return {
            "filename": filename,
            "extension": Path(filename).suffix.lower(),
            "character_count": len(content),
            "line_count": len(lines),
            "empty_lines": sum(
                1 for line in lines if not line.strip()
            ),
            "non_empty_lines": sum(
                1 for line in lines if line.strip()
            ),
            "preview": content[:500],
        }


if __name__ == "__main__":
    scanner = ScannerAgent()

    sample_code = """
def calculate_total(price, tax):
    total = price + (price * tax)
    return total


def main():
    result = calculate_total(100, 0.18)
    print(result)
"""

    result = scanner.scan_content(
        sample_code,
        "sample.py"
    )

    print("=== SCANNER AGENT ===")
    print(f"File: {result['filename']}")
    print(f"Extension: {result['extension']}")
    print(f"Characters: {result['character_count']}")
    print(f"Lines: {result['line_count']}")
    print(f"Non-empty lines: {result['non_empty_lines']}")
    print("\nPreview:")
    print(result["preview"])