"""
agent/tools/text_tool.py

Reads a specific .txt, .md, or .csv file by path.
Used when the user explicitly names a text-based file.
"""

import logging
from langchain_core.tools import Tool
from config.settings import settings

logger = logging.getLogger(__name__)

SUPPORTED = {".txt", ".md", ".csv", ".log"}


def read_text_file(file_path: str) -> str:
    """Read and return the contents of a text file."""
    file_path = file_path.strip().strip('"').strip("'")

    # Resolve path relative to project root
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    resolved = project_root / file_path
    if resolved.exists():
        file_path = str(resolved)

    import os
    _, ext = os.path.splitext(file_path.lower())

    if ext not in SUPPORTED:
        return (
            f"Error: Unsupported file type '{ext}'. "
            f"Supported types: {', '.join(SUPPORTED)}"
        )

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if not content.strip():
            return f"File '{file_path}' exists but is empty."

        return content[: settings.MAX_CONTENT_LENGTH * 2]

    except FileNotFoundError:
        return (
            f"Error: File not found at '{file_path}'. "
            "Check that the path is correct and relative to the project root."
        )
    except Exception as exc:
        logger.error("Text file read error for '%s': %s", file_path, exc)
        return f"Error reading file '{file_path}': {str(exc)}"


text_tool = Tool(
    name="TextFileReader",
    func=read_text_file,
    description=(
        "Reads the full contents of a local text file (.txt, .md, .csv, .log). "
        "Use this when the user explicitly names a specific file to read. "
        "Input must be the file path relative to the project root, "
        "e.g. 'sample_docs/notes.txt'. "
        "For general questions, prefer LocalDocumentSearch instead."
    ),
)
