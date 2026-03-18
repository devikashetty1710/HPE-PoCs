"""
vectorstore/document_indexer.py

Scans the LOCAL_DOCS_DIR folder, loads every supported file,
chunks the text, and indexes it into ChromaDB.

Supported formats: .pdf, .txt, .json, .md, .csv
Runs once on agent startup — already-indexed chunks are skipped.
"""

import json
import logging
import os
from pathlib import Path
from typing import List

from config.settings import settings
from vectorstore.chroma_store import vector_store

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".json", ".md", ".csv"}
CHUNK_SIZE = 400       # characters per chunk
CHUNK_OVERLAP = 80    # overlap between adjacent chunks


# ------------------------------------------------------------------
# Text chunking (no LangChain dependency here — keep it simple)
# ------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split a long string into overlapping fixed-size chunks."""
    chunks = []
    start = 0
    text = text.strip()
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ------------------------------------------------------------------
# Per-format loaders
# ------------------------------------------------------------------

def _load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed — cannot index %s", path)
        return ""

    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _load_json(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    # Flatten JSON to a readable string so it can be chunked
    return json.dumps(data, indent=2)


def _load_csv(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_md(path: str) -> str:
    return _load_txt(path)


LOADERS = {
    ".pdf":  _load_pdf,
    ".txt":  _load_txt,
    ".json": _load_json,
    ".csv":  _load_csv,
    ".md":   _load_md,
}


# ------------------------------------------------------------------
# Main indexer
# ------------------------------------------------------------------

def index_local_documents(docs_dir: str = None) -> dict:
    """
    Walk docs_dir, load each supported file, chunk and index into ChromaDB.

    Returns a summary dict: {file_path: chunks_added}
    """
    docs_dir = docs_dir or settings.LOCAL_DOCS_DIR
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        logger.warning("LOCAL_DOCS_DIR '%s' does not exist — creating it.", docs_dir)
        docs_path.mkdir(parents=True, exist_ok=True)
        return {}

    summary = {}
    files = [
        p for p in docs_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.info("No supported documents found in '%s'.", docs_dir)
        return {}

    logger.info("Indexing %d file(s) from '%s'...", len(files), docs_dir)

    for file_path in files:
        ext = file_path.suffix.lower()
        loader = LOADERS.get(ext)
        if not loader:
            continue

        try:
            text = loader(str(file_path))
            if not text.strip():
                logger.warning("Empty content in '%s' — skipping.", file_path)
                continue

            chunks = chunk_text(text)
            added = vector_store.add_documents(
                chunks=chunks,
                source_path=str(file_path),
                file_type=ext.lstrip("."),
            )
            summary[str(file_path)] = added
            logger.info("  %-50s  %d new chunks", file_path.name, added)

        except Exception as exc:
            logger.error("Failed to index '%s': %s", file_path, exc)
            summary[str(file_path)] = 0

    total_new = sum(summary.values())
    logger.info(
        "Indexing complete. %d new chunks added across %d file(s).",
        total_new,
        len(files),
    )
    return summary
