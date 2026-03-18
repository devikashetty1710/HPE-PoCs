"""
main.py — AI Document Retrieval Agent

Entry point. On startup:
  1. Configure logging.
  2. Verify FAISS index exists (run ingest.py if not).
  3. Build the ReAct agent.
  4. Enter an interactive query loop (or run demo queries in batch mode).

Usage:
    python main.py              # interactive mode
    python main.py --demo       # run built-in demo queries and exit
"""

import argparse
import logging
import os
import sys
import textwrap

from config.settings import settings
from agent.agent_core import build_agent

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

FAISS_DIR = "faiss_db"

# ------------------------------------------------------------------
# Demo queries
# ------------------------------------------------------------------

DEMO_QUERIES = [
    "What does the project overview say?",
    "What is the Hackfest 2026 strategy about?",
    "What is retrieval augmented generation?",
    "What is LangChain?",
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def print_banner() -> None:
    print("\n" + "=" * 65)
    print("  AI DOCUMENT RETRIEVAL AGENT")
    print(f"  LLM: {settings.LLM_PROVIDER.upper()} | Docs: {settings.LOCAL_DOCS_DIR}")
    print("=" * 65)


def print_separator() -> None:
    print("\n" + "-" * 65)


def check_faiss_index() -> bool:
    """Check if FAISS index exists. If not, tell user to run ingest.py."""
    if not os.path.exists(FAISS_DIR):
        print("\n⚠️  No FAISS index found.")
        print("Please run this first:  python ingest.py")
        print("Then start the agent again.")
        return False
    print(f"\n[FAISS index found at '{FAISS_DIR}/' — ready to search]")
    return True


def run_query(agent, query: str) -> None:
    print_separator()
    print(f"Query: {query}")
    print_separator()

    try:
        result = agent.invoke({"input": query})
        answer = result.get("output", "No answer returned.")

        print("\n" + "=" * 20 + " FINAL ANSWER " + "=" * 20)
        print(textwrap.fill(answer, width=80))
        print("=" * 54)

        # Show which tools were called
        steps = result.get("intermediate_steps", [])
        if steps:
            tool_names = [step[0].tool for step in steps]
            print(f"\n[Tools used: {' → '.join(tool_names)}]")

    except KeyboardInterrupt:
        print("\n[Query interrupted by user]")
    except Exception as exc:
        logger.error("Agent error: %s", exc)
        print(f"\nError running query: {exc}")


def interactive_loop(agent) -> None:
    print("\nType your question and press Enter. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        run_query(agent, user_input)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AI Document Retrieval Agent")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a set of demo queries and exit (non-interactive).",
    )
    args = parser.parse_args()

    print_banner()

    # Step 1: Verify FAISS index exists
    if not check_faiss_index():
        sys.exit(1)

    # Step 2: Build agent
    logger.info("Building agent...")
    try:
        agent = build_agent()
    except EnvironmentError as exc:
        print(f"\nConfiguration error: {exc}")
        print("Edit your .env file and try again.")
        sys.exit(1)

    # Step 3: Run queries
    if args.demo:
        print("\nRunning demo queries...\n")
        for query in DEMO_QUERIES:
            run_query(agent, query)
        print("\nDemo complete.")
    else:
        interactive_loop(agent)


if __name__ == "__main__":
    main()