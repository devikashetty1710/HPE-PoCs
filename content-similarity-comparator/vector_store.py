"""
vector_store.py
---------------
Manages a persistent ChromaDB vector store for document embeddings.

ANALOGY: Think of ChromaDB as a smart filing cabinet with a photographic
memory. The first time you hand it a document, it reads it, remembers its
"meaning fingerprint" (embedding), and files it away. Next time you bring
the same document, it recognises it immediately and hands back the
fingerprint without re-reading — saving you time and API quota.

KEY TERMS:
  - ChromaDB    : an open-source vector database that runs locally
  - Collection  : a named group of vectors inside ChromaDB (like a table)
  - Persist     : save to disk so data survives between runs
  - Document ID : a unique key for each document (e.g. DEV-7, GH-4)
  - Upsert      : insert if new, update if already exists

HOW IT WORKS:
  1. First run  → embed all docs → store in ChromaDB → done
  2. Second run → check ChromaDB → already stored → skip embedding
  3. New doc    → not in ChromaDB → embed only that one → store it
"""

import os
import numpy as np
from pathlib import Path

# ChromaDB — local vector database
import chromadb
from chromadb.config import Settings


# ── Constants ─────────────────────────────────────────────────────────────────
# Where ChromaDB saves its data on disk
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def get_client() -> chromadb.ClientAPI:
    """
    Create (or reuse) a persistent ChromaDB client.

    PersistentClient saves everything to disk at CHROMA_DIR.
    Next time you run, it loads from there automatically.
    """
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client


def get_collection(name: str) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection by name.

    A collection is like a table in a regular database.
    We create one per source:
      - "jira_docs"   for Jira tickets
      - "github_docs" for GitHub issues

    Parameters
    ----------
    name : str — collection name e.g. "jira_docs"

    Returns
    -------
    chromadb.Collection
    """
    client = get_client()
    # get_or_create_collection → creates if not exists, returns existing if it does
    collection = client.get_or_create_collection(
        name=name,
        # cosine similarity is what we use in comparator.py
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def get_cached_embeddings(
    docs: list[dict],
    collection_name: str,
) -> tuple[list[dict], list[int]]:
    """
    Check which documents already have embeddings stored in ChromaDB.

    Parameters
    ----------
    docs            : list of document dicts from loader/fetcher
    collection_name : e.g. "jira_docs" or "github_docs"

    Returns
    -------
    tuple:
      - missing_docs  : list of docs that are NOT in ChromaDB yet
      - missing_indices: their positions in the original docs list
    """
    collection = get_collection(collection_name)

    # Get all IDs currently stored in this collection
    existing = collection.get(include=[])   # include=[] → only fetch IDs, not vectors
    existing_ids = set(existing["ids"])

    missing_docs    = []
    missing_indices = []

    for i, doc in enumerate(docs):
        if doc["id"] not in existing_ids:
            missing_docs.append(doc)
            missing_indices.append(i)

    cached_count  = len(docs) - len(missing_docs)
    missing_count = len(missing_docs)

    print(f"[VectorStore] Collection '{collection_name}':")
    print(f"  {cached_count} docs already cached  |  {missing_count} new docs need embedding")

    return missing_docs, missing_indices


def store_embeddings(
    docs: list[dict],
    embeddings: np.ndarray,
    collection_name: str,
) -> None:
    """
    Store new embeddings into ChromaDB.

    Parameters
    ----------
    docs            : list of document dicts
    embeddings      : numpy array of shape (len(docs), dim)
    collection_name : collection to store into
    """
    if not docs:
        return

    collection = get_collection(collection_name)

    # ChromaDB expects:
    #   ids        : list of unique string IDs
    #   embeddings : list of lists (not numpy arrays)
    #   documents  : list of text strings (stored for reference)
    #   metadatas  : list of dicts with extra info

    ids        = [doc["id"]         for doc in docs]
    texts      = [doc["full_text"]  for doc in docs]
    vectors    = embeddings.tolist()   # convert numpy → plain Python list
    metadatas  = [{"title": doc["title"], "source": collection_name} for doc in docs]

    # upsert = insert if new, update if already exists
    collection.upsert(
        ids        = ids,
        embeddings = vectors,
        documents  = texts,
        metadatas  = metadatas,
    )

    print(f"[VectorStore] Stored {len(docs)} embeddings into '{collection_name}'")


def load_all_embeddings(
    docs: list[dict],
    collection_name: str,
) -> np.ndarray:
    collection = get_collection(collection_name)
    ids = [doc["id"] for doc in docs]

    result = collection.get(
        ids     = ids,
        include = ["embeddings"],
    )

    # ChromaDB doesn't guarantee order — re-order to match docs list
    id_to_embedding = {
        result["ids"][i]: result["embeddings"][i]
        for i in range(len(result["ids"]))
    }

    # Build embeddings in exact same order as docs
    ordered = [id_to_embedding[doc["id"]] for doc in docs]
    embeddings = np.array(ordered, dtype=np.float32)

    print(f"[VectorStore] Loaded {len(embeddings)} embeddings from '{collection_name}'")
    return embeddings

def clear_collection(collection_name: str) -> None:
    """
    Delete all embeddings in a collection.
    Useful when you want to force a full re-embed (e.g. after changing models).

    Usage:
      from vector_store import clear_collection
      clear_collection("jira_docs")
    """
    client = get_client()
    try:
        client.delete_collection(collection_name)
        print(f"[VectorStore] Cleared collection '{collection_name}'")
    except Exception:
        print(f"[VectorStore] Collection '{collection_name}' not found — nothing to clear")


def get_collection_stats() -> None:
    """Print stats about all collections in the ChromaDB store."""
    client = get_client()
    collections = client.list_collections()

    print(f"\n[VectorStore] ChromaDB at: {CHROMA_DIR}")
    print(f"[VectorStore] Total collections: {len(collections)}")
    for col in collections:
        c     = client.get_collection(col.name)
        count = c.count()
        print(f"  - {col.name}: {count} embeddings stored")
    print()
