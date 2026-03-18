"""
loader.py
---------
Reads source files and returns a list of document dicts.

ANALOGY: Think of this as the "librarian" — it walks into the library
(your files), picks up each book (document), reads the title and summary,
and hands you a neat stack of index cards. Every other part of the code
only ever sees those index cards, never the messy raw files.

KEY TERMS:
  - Document : a single piece of content (one Jira ticket, one GitHub issue)
  - Parsing  : reading raw text and converting it into a structured object
  - Dict     : Python dictionary — a key-value store like {"name": "Devika"}
"""

import re
from pathlib import Path


def parse_documents(filepath: str) -> list[dict]:
    """
    Read a text file that contains multiple documents separated by '---'
    and return them as a list of dicts with keys: id, title, content.

    Parameters
    ----------
    filepath : str
        Path to the .txt source file.

    Returns
    -------
    list[dict]
        e.g. [{"id": "KAN-1", "title": "Login bug", "content": "..."}, ...]
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {filepath}")

    raw = path.read_text(encoding="utf-8")

    # Split on the '---' separator that divides documents
    # Each chunk looks like:
    #   ID: KAN-1
    #   Title: Login page shows error ...
    #   Description: Users report ...
    chunks = [c.strip() for c in raw.split("---") if c.strip()]

    documents = []
    for chunk in chunks:
        doc = {}

        # Extract ID field  (e.g.  "ID: KAN-1")
        id_match = re.search(r"^ID:\s*(.+)$", chunk, re.MULTILINE)
        doc["id"] = id_match.group(1).strip() if id_match else "UNKNOWN"

        # Extract Title field
        title_match = re.search(r"^Title:\s*(.+)$", chunk, re.MULTILINE)
        doc["title"] = title_match.group(1).strip() if title_match else ""

        # Extract Description field (may span multiple lines)
        desc_match = re.search(r"^Description:\s*(.+)", chunk, re.MULTILINE | re.DOTALL)
        doc["content"] = desc_match.group(1).strip() if desc_match else chunk

        # Full text = title + content (used for embedding)
        doc["full_text"] = f"{doc['title']}. {doc['content']}"

        documents.append(doc)

    print(f"[Loader] Loaded {len(documents)} documents from '{filepath}'")
    return documents
