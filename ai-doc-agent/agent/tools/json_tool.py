"""
agent/tools/json_tool.py

Reads a specific JSON file and returns its contents as formatted text.
Handles both flat objects and nested/array structures.
"""

import json
import logging
from langchain_core.tools import Tool
from config.settings import settings

logger = logging.getLogger(__name__)


def read_json_file(file_path: str) -> str:
    """Load a JSON file and return a human-readable representation."""
    file_path = file_path.strip().strip('"').strip("'")

    # Resolve path relative to project root
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    resolved = project_root / file_path
    if resolved.exists():
        file_path = str(resolved)

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        formatted = json.dumps(data, indent=2, ensure_ascii=False)

        if len(formatted) > settings.MAX_CONTENT_LENGTH * 2:
            # Summarise large JSON rather than truncating mid-structure
            if isinstance(data, list):
                summary = (
                    f"JSON Array with {len(data)} items.\n"
                    f"First item sample:\n{json.dumps(data[0], indent=2)}"
                )
                return summary[: settings.MAX_CONTENT_LENGTH]
            elif isinstance(data, dict):
                summary = (
                    f"JSON Object with keys: {list(data.keys())}\n"
                    f"Content (truncated):\n{formatted[: settings.MAX_CONTENT_LENGTH]}"
                )
                return summary

        return formatted

    except FileNotFoundError:
        return (
            f"Error: File not found at '{file_path}'. "
            "Check that the path is correct and relative to the project root."
        )
    except json.JSONDecodeError as exc:
        return f"Error: Could not parse JSON in '{file_path}': {str(exc)}"
    except Exception as exc:
        logger.error("JSON read error for '%s': %s", file_path, exc)
        return f"Error reading JSON file '{file_path}': {str(exc)}"


json_tool = Tool(
    name="JSONFileReader",
    func=read_json_file,
    description=(
        "Reads and parses a local JSON file (.json). "
        "Use this when the user explicitly asks about a specific JSON file. "
        "Input must be the file path relative to the project root, "
        "e.g. 'sample_docs/config.json'. "
        "For general questions, prefer LocalDocumentSearch instead."
    ),
)
