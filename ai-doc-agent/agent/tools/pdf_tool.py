"""
agent/tools/pdf_tool.py

Reads a specific PDF file by path and returns its text content.
Used when the user explicitly names a PDF file.
"""

import logging
from langchain_core.tools import Tool
from config.settings import settings

logger = logging.getLogger(__name__)


def read_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file at the given path.

    Tries pypdf first. Returns extracted text or an error string.
    """
    file_path = file_path.strip().strip('"').strip("'")

    # Resolve path relative to project root
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    resolved = project_root / file_path
    if resolved.exists():
        file_path = str(resolved)

    try:
        from pypdf import PdfReader
    except ImportError:
        return "Error: pypdf is not installed. Run: pip install pypdf"

    try:
        reader = PdfReader(file_path)
        pages = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pages.append(f"[Page {page_num + 1}]\n{text}")

        if not pages:
            return f"PDF '{file_path}' was loaded but no text could be extracted (possibly a scanned image PDF)."

        full_text = "\n\n".join(pages)
        return full_text[: settings.MAX_CONTENT_LENGTH * 2]  # allow more for direct reads

    except FileNotFoundError:
        return (
            f"Error: File not found at '{file_path}'. "
            "Make sure the path is relative to the project root, e.g. sample_docs/report.pdf"
        )
    except Exception as exc:
        logger.error("PDF read error for '%s': %s", file_path, exc)
        return f"Error reading PDF '{file_path}': {str(exc)}"


pdf_tool = Tool(
    name="PDFReader",
    func=read_pdf,
    description=(
        "Reads and extracts full text from a specific PDF file. "
        "Use this when the user explicitly provides a PDF file path or file name. "
        "Input must be the file path relative to the project root, "
        "e.g. 'sample_docs/report.pdf'. "
        "Prefer LocalDocumentSearch for general questions — use this only when "
        "the user asks to read a specific PDF file directly."
    ),
)
