"""
embedder.py
-----------
Converts text documents into embedding vectors using either:
  • Gemini API   (cloud, needs GEMINI_API_KEY)
  • Ollama       (local, needs ollama running with llama3.2:3b)
  • Dummy        (returns random vectors — no LLM needed, for testing)

ANALOGY: Imagine every document is a person at a party. An "embedding"
is like giving each person GPS coordinates on a map of topics. Documents
that talk about similar things get placed close together on that map.
The LLM is the cartographer — it reads the text and assigns the coordinates.

Those coordinates are just a list of numbers, e.g.:
  [0.12, -0.87, 0.44, 0.03, ...]   ← a "vector" with hundreds of dimensions

KEY TERMS:
  - Embedding  : a fixed-size list of numbers that represents text meaning
  - Vector     : just a list of numbers  [x1, x2, x3, ...]
  - Dimension  : each number in the list is one dimension
  - API        : Application Programming Interface — a way to talk to an
                 external service (like Gemini) over the internet
  - Local model: an LLM running on YOUR machine, no internet needed
"""

import os
import time
import numpy as np
from dotenv import load_dotenv


load_dotenv()  # Read .env file to get GEMINI_API_KEY


# ─────────────────────────────── DUMMY MODE ──────────────────────────────────

def embed_dummy(texts: list[str], dim: int = 128) -> np.ndarray:
    """
    Returns random vectors. Useful to test the pipeline without any LLM.
    Two calls with the same text will give DIFFERENT random vectors —
    that's fine for structural testing, NOT for real similarity scoring.

    Parameters
    ----------
    texts : list[str]  — the documents to embed
    dim   : int        — how many dimensions per vector (default 128)

    Returns
    -------
    np.ndarray of shape (len(texts), dim)
    """
    print(f"[Embedder-Dummy] Generating {len(texts)} random vectors (dim={dim})")
    # np.random.randn → random numbers from a "normal distribution"
    vectors = np.random.randn(len(texts), dim)
    # Normalise so each vector has length 1 (required for cosine similarity)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


# ─────────────────────────────── GEMINI MODE ─────────────────────────────────

def embed_gemini(texts: list[str]) -> np.ndarray:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError("Run: pip install google-genai")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Add it to your .env file:\n"
            "  GEMINI_API_KEY=your_key_here"
        )

    client = genai.Client(api_key=api_key)
    model_name = "models/gemini-embedding-001"

    print(f"[Embedder-Gemini] Embedding {len(texts)} texts via '{model_name}' ...")
    vectors = []
    for i, text in enumerate(texts):
        result = client.models.embed_content(
            model=model_name,
            contents=text,
        )
        vec = np.array(result.embeddings[0].values, dtype=np.float32)
        vectors.append(vec)
        print(f"  [{i+1}/{len(texts)}] embedded (dim={len(vec)})")
        time.sleep(0.3)

    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1, norms)

# ─────────────────────────────── OLLAMA MODE ─────────────────────────────────

def embed_ollama(texts: list[str], model: str = "llama3.2:3b") -> np.ndarray:
    """
    Calls a locally running Ollama server for embeddings.

    Requires:
      1. Install Ollama: https://ollama.com/download
      2. Pull model:  ollama pull llama3.2:3b
      3. Start server: ollama serve   (or it starts automatically on install)

    Ollama runs on http://localhost:11434 by default.
    The /api/embeddings endpoint accepts { model, prompt } and returns { embedding }.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("Run: pip install requests")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    endpoint = f"{base_url}/api/embeddings"

    print(f"[Embedder-Ollama] Embedding {len(texts)} texts via '{model}' at {base_url} ...")

    vectors = []
    for i, text in enumerate(texts):
        payload = {"model": model, "prompt": text}
        try:
            resp = requests.post(endpoint, json=payload, timeout=120)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot reach Ollama at {base_url}.\n"
                "Make sure Ollama is installed and running:\n"
                "  ollama serve"
            )

        vec = np.array(resp.json()["embedding"], dtype=np.float32)
        vectors.append(vec)
        print(f"  [{i+1}/{len(texts)}] embedded (dim={len(vec)})")

    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1, norms)


# ─────────────────────────────── PUBLIC API ───────────────────────────────────

def get_embeddings(texts: list[str], mode: str = "dummy") -> np.ndarray:
    """
    Unified entry point. Call this from main.py.

    mode options:
      "dummy"  → random vectors (no LLM, instant, for testing)
      "gemini" → Google Gemini cloud API
      "ollama" → local Ollama server
    """
    mode = mode.lower()
    if mode == "dummy":
        return embed_dummy(texts)
    elif mode == "gemini":
        return embed_gemini(texts)
    elif mode == "ollama":
        return embed_ollama(texts)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: dummy | gemini | ollama")
