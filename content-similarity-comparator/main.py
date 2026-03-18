"""
main.py
-------
Entry point. Wires Loader → Embedder → Comparator → Reporter together.

Usage
-----
  python main.py                        # defaults: mode=dummy, threshold=0.75
  python main.py --mode dummy           # random vectors, no LLM
  python main.py --mode gemini          # Google Gemini cloud embeddings
  python main.py --mode ollama          # local Ollama llama3.2:3b
  python main.py --mode gemini --threshold 0.80
  python main.py --source_a data/source_a.txt --source_b data/source_b.txt

ANALOGY: main.py is the "director" on a film set. It doesn't act itself —
it calls each crew member (loader, embedder, comparator, reporter) in the
right order and passes their output to the next person.
"""

import argparse
import sys
from pathlib import Path

# ── Import our own modules ────────────────────────────────────────────────────
# These are the files we created in the same folder.
from loader     import parse_documents
from embedder   import get_embeddings
from comparator import compare_sources
from reporter   import print_results, save_json


def parse_args():
    """
    argparse : Python's built-in library for reading command-line arguments.
    e.g.  python main.py --mode gemini  →  args.mode == "gemini"
    """
    parser = argparse.ArgumentParser(
        description="Compare similar content between two sources using LLM embeddings."
    )
    parser.add_argument(
        "--mode",
        choices=["dummy", "gemini", "ollama"],
        default="dummy",
        help="Embedding backend to use (default: dummy)",
    )
    parser.add_argument(
        "--source",
        choices=["files", "live"],
        default="files",
        help="Data source: files or live APIs",
    )
    parser.add_argument(
        "--source_a",
        default="data/source_a.txt",
        help="Path to Source A text file (default: data/source_a.txt)",
    )
    parser.add_argument(
        "--source_b",
        default="data/source_b.txt",
        help="Path to Source B text file (default: data/source_b.txt)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Minimum cosine similarity to consider a match (default: 0.75)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to a JSON file in ./output/",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate LLM summary for each matched pair (RAG generation step)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("  Content Similarity Comparator")
    print(f"  Mode      : {args.mode.upper()}")
    print(f"  Source A  : {args.source_a}")
    print(f"  Source B  : {args.source_b}")
    print(f"  Threshold : {args.threshold}")
    print("=" * 60 + "\n")

    # ── STEP 1: Load documents ─────────────────────────────────────────────────
    print("[Step 1/4] Loading documents ...")
    if args.source == "live":
        from fetchers.jira_fetcher import fetch_jira_tickets
        from fetchers.github_fetcher import fetch_github_issues
        docs_a = fetch_jira_tickets()
        docs_b = fetch_github_issues()
    else:
        docs_a = parse_documents(args.source_a)
        docs_b = parse_documents(args.source_b)

    # Pull out just the text strings for embedding
    # Each doc's "full_text" = title + description (set in loader.py)
    texts_a = [d["full_text"] for d in docs_a]
    texts_b = [d["full_text"] for d in docs_b]

    # ── STEP 2: Generate embeddings ────────────────────────────────────────────
    print(f"\n[Step 2/4] Generating embeddings (mode={args.mode}) ...")
    embeddings_a = get_embeddings(texts_a, mode=args.mode)
    embeddings_b = get_embeddings(texts_b, mode=args.mode)

    print(f"  Embeddings A shape: {embeddings_a.shape}")  # (num_docs, vector_dim)
    print(f"  Embeddings B shape: {embeddings_b.shape}")

    # ── STEP 3: Compare ────────────────────────────────────────────────────────
    print(f"\n[Step 3/4] Comparing documents (threshold={args.threshold}) ...")
    matches = compare_sources(
        docs_a, embeddings_a,
        docs_b, embeddings_b,
        threshold=args.threshold,
    )

    # ── STEP 4: Report ─────────────────────────────────────────────────────────
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
