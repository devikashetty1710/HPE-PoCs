"""
agent/tools/local_search_tool.py

PRIMARY TOOL — always tried first.

Performs a semantic search over all locally indexed documents
(PDFs, TXT, JSON, CSV, Markdown) stored in FAISS.

The agent is instructed to call this tool before any web tool.
If it returns a confident result, no web search is needed.
"""

import os
import logging
from langchain_core.tools import Tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)

FAISS_DIR = "faiss_db"
RELEVANCE_THRESHOLD = 0.10

# Load once at startup — not on every query
_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        if not os.path.exists(FAISS_DIR):
            return None
        _vectorstore = FAISS.load_local(
            FAISS_DIR,
            _embeddings,
            allow_dangerous_deserialization=True
        )
    return _vectorstore


def search_local_documents(query: str) -> str:
    """
    Search all indexed local documents for content relevant to the query.

    Returns the top matching chunks with their source files, or a clear
    NOT_FOUND_LOCALLY signal so the agent knows to fall back to the web.
    """
    if not query.strip():
        return "Error: empty query provided."

    vectorstore = _get_vectorstore()

    if vectorstore is None:
        return (
            "NOT_FOUND_LOCALLY: No local documents have been indexed yet. "
            "Please add files to the sample_docs/ folder and run ingest.py."
        )

    results = vectorstore.similarity_search_with_relevance_scores(query, k=3)

    if not results:
        return "NOT_FOUND_LOCALLY: No results returned from local document search."

    relevant = [(doc, score) for doc, score in results if score >= RELEVANCE_THRESHOLD]

    if not relevant:
        return (
            f"NOT_FOUND_LOCALLY: No sufficiently relevant content found in local "
            f"documents. Best score was {results[0][1]:.2f} — "
            f"below threshold {RELEVANCE_THRESHOLD}."
        )

    # Format results clearly
    lines = [f"Found {len(relevant)} relevant chunk(s) in local documents:\n"]
    seen_sources = set()

    for i, (doc, score) in enumerate(relevant, 1):
        source = doc.metadata.get("source_file", "unknown file")
        seen_sources.add(source)
        lines.append(
            f"--- Result {i} | Source: {source} | Score: {score:.2f} ---\n"
            f"{doc.page_content[:500]}\n"
        )

    lines.append(f"\n[Indexed sources: {', '.join(seen_sources)}]")
    return "\n".join(lines)


local_search_tool = Tool(
    name="LocalDocumentSearch",
    func=search_local_documents,
    description=(
        "ALWAYS USE THIS TOOL FIRST before searching the web. "
        "Searches all locally indexed documents — PDFs, text files, JSON files, "
        "CSV files, and Markdown files — using semantic similarity. "
        "Input: a natural language question or keyword phrase. "
        "If the result starts with NOT_FOUND_LOCALLY, the answer is not in local "
        "documents and you should proceed to web search tools."
    ),
)