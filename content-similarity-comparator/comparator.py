"""
comparator.py
-------------
Compares two sets of embedding vectors and returns pairs that are similar.

ANALOGY: Remember those GPS coordinates from embedder.py? Now imagine
you have two groups of people (Source A and Source B) on a map. This
module measures the angle between every possible pair — one from group A,
one from group B. If they're facing the same direction (small angle →
score near 1.0), they're talking about the same thing.

That angle-based measurement is called COSINE SIMILARITY:
  • Score = 1.0   → identical meaning
  • Score = 0.8+  → very similar topic
  • Score = 0.5   → loosely related
  • Score ≈ 0.0   → unrelated

KEY TERMS:
  - Cosine similarity : dot product of two unit vectors = cos(angle between them)
  - Dot product       : multiply matching numbers and add them up
  - Threshold         : minimum score to call two docs "similar" (e.g. 0.75)
  - Cross-comparison  : every doc in A compared against every doc in B
"""

import numpy as np


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Because we already normalised vectors in embedder.py (length = 1),
    this is simply a dot product — which is fast and numerically stable.

    Parameters
    ----------
    vec_a, vec_b : 1-D numpy arrays of the same length

    Returns
    -------
    float in range [-1, 1]  (for normalised embeddings: always [0, 1])
    """
    return float(np.dot(vec_a, vec_b))


def compare_sources(
    docs_a: list[dict],
    embeddings_a: np.ndarray,
    docs_b: list[dict],
    embeddings_b: np.ndarray,
    threshold: float = 0.75,
) -> list[dict]:
    """
    Compare every document in Source A against every document in Source B.
    Return all pairs whose cosine similarity meets the threshold, sorted
    by score (highest first).

    Parameters
    ----------
    docs_a        : list of doc dicts from Source A  (from loader.py)
    embeddings_a  : numpy array, shape (len(docs_a), dim)
    docs_b        : list of doc dicts from Source B
    embeddings_b  : numpy array, shape (len(docs_b), dim)
    threshold     : minimum similarity score to include a pair

    Returns
    -------
    list[dict] — each item:
      {
        "score"   : float,       # similarity 0–1
        "doc_a"   : dict,        # source A document
        "doc_b"   : dict,        # source B document
      }
    """
    print(f"\n[Comparator] Comparing {len(docs_a)} x {len(docs_b)} document pairs ...")
    print(f"[Comparator] Similarity threshold = {threshold}\n")

    matches = []

    # Outer loop: each document in Source A
    for i, (doc_a, emb_a) in enumerate(zip(docs_a, embeddings_a)):
        # Inner loop: each document in Source B
        for j, (doc_b, emb_b) in enumerate(zip(docs_b, embeddings_b)):
            score = cosine_similarity(emb_a, emb_b)

            # Only keep pairs above the threshold
            if score >= threshold:
                matches.append({
                    "score": round(score, 4),
                    "doc_a": doc_a,
                    "doc_b": doc_b,
                })

    # Sort by score, best match first
    matches.sort(key=lambda x: x["score"], reverse=True)

    print(f"[Comparator] Found {len(matches)} similar pairs above threshold {threshold}")
    return matches
