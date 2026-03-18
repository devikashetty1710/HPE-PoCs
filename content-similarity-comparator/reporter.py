"""
reporter.py
-----------
Pretty-prints comparison results to the console and saves them as JSON.

ANALOGY: This is the "announcer" at the end of the race. The comparator
did all the math; the reporter just reads out the results in a nice format
that humans can actually understand.

KEY TERMS:
  - JSON : JavaScript Object Notation — a text format for storing structured
           data. Looks like a Python dict. Universally readable by any language.
"""

import json
from pathlib import Path
from datetime import datetime


# ANSI colour codes — make terminal output prettier
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"


def _score_bar(score: float, width: int = 20) -> str:
    """Draw a simple ASCII progress bar for the similarity score."""
    filled = int(score * width)
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def _score_colour(score: float) -> str:
    if score >= 0.90:
        return GREEN
    elif score >= 0.80:
        return CYAN
    else:
        return YELLOW


def print_results(matches: list[dict], mode: str) -> None:
    """
    Print a formatted comparison report to the console.

    Parameters
    ----------
    matches : output from comparator.compare_sources()
    mode    : embedding mode used ("dummy" | "gemini" | "ollama")
    """
    print("\n" + "=" * 70)
    print(f"{BOLD}  CONTENT SIMILARITY REPORT{RESET}  |  mode: {CYAN}{mode.upper()}{RESET}")
    print("=" * 70)

    if not matches:
        print(f"\n{YELLOW}  No similar pairs found above the threshold.{RESET}")
        print("  Try lowering --threshold (default 0.75) or using a real LLM mode.\n")
        return

    for rank, m in enumerate(matches, start=1):
        score  = m["score"]
        doc_a  = m["doc_a"]
        doc_b  = m["doc_b"]
        colour = _score_colour(score)
        bar    = _score_bar(score)

        print(f"\n  {BOLD}#{rank}{RESET}  {colour}{bar}  {score:.4f}{RESET}")
        print(f"  {DIM}── Source A ──{RESET}")
        print(f"     ID    : {doc_a['id']}")
        print(f"     Title : {doc_a['title']}")
        print(f"  {DIM}── Source B ──{RESET}")
        print(f"     ID    : {doc_b['id']}")
        print(f"     Title : {doc_b['title']}")
        print()

    print("=" * 70 + "\n")


def save_json(matches: list[dict], mode: str, output_dir: str = ".") -> str:
    """
    Save the matches list to a JSON file.

    Returns the path of the saved file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"results_{mode}_{timestamp}.json"
    filepath  = Path(output_dir) / filename

    # Build a clean output structure
    output = {
        "mode"       : mode,
        "timestamp"  : timestamp,
        "total_matches": len(matches),
        "matches"    : [
            {
                "rank"   : i + 1,
                "score"  : m["score"],
                "source_a": {"id": m["doc_a"]["id"], "title": m["doc_a"]["title"]},
                "source_b": {"id": m["doc_b"]["id"], "title": m["doc_b"]["title"]},
            }
            for i, m in enumerate(matches)
        ],
    }

    filepath.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[Reporter] Results saved → {filepath}")
    return str(filepath)
