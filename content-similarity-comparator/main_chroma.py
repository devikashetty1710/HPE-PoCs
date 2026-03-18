"""
main.py (ChromaDB version)
--------------------------
Same pipeline as before but now with ChromaDB caching.

NEW FLOW:
  1. Load docs from files or live APIs
  2. Check ChromaDB — which docs are already embedded?
  3. Only embed the NEW docs (skip cached ones)
  4. Store new embeddings into ChromaDB
  5. Load ALL embeddings from ChromaDB
  6. Compare → Report → Generate

Usage
-----
  # Normal run — uses cache automatically
  python main.py --mode gemini --source live --threshold 0.75

  # Force re-embed everything (ignore cache)
  python main.py --mode gemini --source live --threshold 0.75 --refresh

  # See what's stored in ChromaDB
  python main.py --stats
"""

import argparse
import sys

from loader      import parse_documents
from embedder    import get_embeddings
from comparator  import compare_sources
from reporter    import print_results, save_json
from vector_store import (
    get_cached_embeddings,
    store_embeddings,
    load_all_embeddings,
    clear_collection,
    get_collection_stats,
)
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare similar content between two sources using LLM embeddings + ChromaDB cache."
    )
    parser.add_argument(
        "--mode",
        choices=["dummy", "gemini", "ollama"],
        default="dummy",
        help="Embedding backend (default: dummy)",
    )
    parser.add_argument(
        "--source",
        choices=["files", "live"],
        default="files",
        help="Data source: files or live APIs (default: files)",
    )
    parser.add_argument(
        "--source_a",
        default="data/source_a.txt",
        help="Path to Source A (default: data/source_a.txt)",
    )
    parser.add_argument(
        "--source_b",
        default="data/source_b.txt",
        help="Path to Source B (default: data/source_b.txt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Minimum similarity score (default: 0.75)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to JSON",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate RAG summary for each match",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Clear ChromaDB cache and re-embed everything",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show ChromaDB collection stats and exit",
    )
    return parser.parse_args()


def embed_with_cache(
    docs: list[dict],
    collection_name: str,
    mode: str,
    refresh: bool = False,
) -> "np.ndarray":
    """
    Smart embedding function:
      - If refresh=True  → clear cache, re-embed everything
      - If doc is cached → skip embedding, load from ChromaDB
      - If doc is new    → embed only new docs, store in ChromaDB
    """
    import numpy as np

    # ── Optional: clear cache first ───────────────────────────────────────
    if refresh:
        print(f"  [Cache] Refreshing — clearing '{collection_name}' ...")
        clear_collection(collection_name)

    # ── Check which docs need embedding ───────────────────────────────────
    # In dummy mode, skip cache entirely (random vectors aren't worth caching)
    if mode == "dummy":
        print(f"  [Cache] Dummy mode — skipping cache")
        texts = [d["full_text"] for d in docs]
        return get_embeddings(texts, mode="dummy")

    missing_docs, missing_indices = get_cached_embeddings(docs, collection_name)

    # ── Embed only the missing docs ────────────────────────────────────────
    if missing_docs:
        texts_to_embed = [d["full_text"] for d in missing_docs]
        print(f"  [Cache] Embedding {len(missing_docs)} new docs via '{mode}' ...")
        new_embeddings = get_embeddings(texts_to_embed, mode=mode)

        # Store new embeddings in ChromaDB
        store_embeddings(missing_docs, new_embeddings, collection_name)
    else:
        print(f"  [Cache] All docs already cached — skipping embedding entirely! ✓")

    # ── Load ALL embeddings from ChromaDB (cached + new) ──────────────────
    all_embeddings = load_all_embeddings(docs, collection_name)
    return all_embeddings


def main():
    args = parse_args()

    # ── Stats mode — just show ChromaDB contents and exit ─────────────────
    if args.stats:
        get_collection_stats()
        return 0

    print("\n" + "=" * 60)
    print("  Content Similarity Comparator  (ChromaDB enabled)")
    print(f"  Mode      : {args.mode.upper()}")
    print(f"  Source    : {args.source.upper()}")
    print(f"  Threshold : {args.threshold}")
    print(f"  Refresh   : {args.refresh}")
    print("=" * 60 + "\n")

    # ── STEP 1: Load documents ─────────────────────────────────────────────
    print("[Step 1/4] Loading documents ...")
    if args.source == "live":
        from fetchers.jira_fetcher   import fetch_jira_tickets
        from fetchers.github_fetcher import fetch_github_issues
        docs_a = fetch_jira_tickets()
        docs_b = fetch_github_issues()
    else:
        docs_a = parse_documents(args.source_a)
        docs_b = parse_documents(args.source_b)

    print(f"  Source A: {len(docs_a)} documents")
    print(f"  Source B: {len(docs_b)} documents")

    if not docs_a or not docs_b:
        print("[ERROR] One or both sources returned no documents.")
        return 1

    # ── STEP 2: Smart embedding with ChromaDB cache ────────────────────────
    print(f"\n[Step 2/4] Smart embedding (mode={args.mode}) ...")
    print("  Source A:")
    embeddings_a = embed_with_cache(
        docs_a, "jira_docs", args.mode, refresh=args.refresh
    )
    print("  Source B:")
    embeddings_b = embed_with_cache(
        docs_b, "github_docs", args.mode, refresh=args.refresh
    )

    print(f"\n  Embeddings A shape: {embeddings_a.shape}")
    print(f"  Embeddings B shape: {embeddings_b.shape}")

    # ── STEP 3: Compare ────────────────────────────────────────────────────
    print(f"\n[Step 3/4] Comparing (threshold={args.threshold}) ...")
    matches = compare_sources(
        docs_a, embeddings_a,
        docs_b, embeddings_b,
        threshold=args.threshold,
    )

    # ── STEP 4: Report ─────────────────────────────────────────────────────
    print("[Step 4/4] Generating report ...")
    print_results(matches, mode=args.mode)

    if args.save:
        save_json(matches, mode=args.mode, output_dir="output")

    if args.generate:
        from generator import generate_all_summaries
        generate_all_summaries(matches, mode=args.mode)

    return 0


if __name__ == "__main__":
    sys.exit(main())
