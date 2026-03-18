"""
generator.py
------------
The 'G' in RAG — Generation.

Takes matched document pairs from comparator.py and feeds them into
Gemini to generate a human-readable analysis of WHY they match and
WHAT to do about it.

ANALOGY: The comparator was a detective who found two suspicious people
who look similar. The generator is the detective's report — it explains
in plain English who they are, why they look similar, and what action
to take.

FLOW:
  comparator.py → match dict → generator.py → LLM prompt → human summary
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()


def build_prompt(match: dict) -> str:
    """
    Build the prompt we send to Gemini for a single matched pair.

    Parameters
    ----------
    match : dict from comparator.compare_sources()
      {
        "score" : float,
        "doc_a" : {"id", "title", "content", "full_text"},
        "doc_b" : {"id", "title", "content", "full_text"},
      }

    Returns
    -------
    str — the full prompt text
    """
    doc_a = match["doc_a"]
    doc_b = match["doc_b"]
    score = match["score"]

    prompt = f"""You are a senior project manager reviewing tickets from two different systems.

Two tickets have been flagged as semantically similar with a similarity score of {score:.2%}.

--- TICKET FROM JIRA ---
ID          : {doc_a['id']}
Title       : {doc_a['title']}
Description : {doc_a.get('content', 'No description available.')}

--- TICKET FROM GITHUB ---
ID          : {doc_b['id']}
Title       : {doc_b['title']}
Description : {doc_b.get('content', 'No description available.')}

Please provide a concise analysis with the following structure:

1. SIMILARITY REASON  : In 1-2 sentences, explain what makes these tickets similar.
2. DUPLICATE CHECK    : Are these exact duplicates or just related issues? Explain briefly.
3. RECOMMENDED ACTION : Choose one — MERGE / LINK / CLOSE ONE / KEEP SEPARATE — and explain why.
4. SUGGESTED OWNER    : Which team should own the resolution — Backend, Frontend, DevOps, or QA?

Keep your response concise and actionable.
"""
    return prompt


def generate_summary(match: dict, mode: str = "gemini") -> str:
    """
    Generate a human-readable analysis for a single matched pair.

    Parameters
    ----------
    match : dict — a single match from comparator.compare_sources()
    mode  : str  — "gemini" or "ollama" (which LLM to use for generation)

    Returns
    -------
    str — the LLM's analysis
    """
    prompt = build_prompt(match)

    if mode == "gemini":
        return _generate_gemini(prompt)
    elif mode == "ollama":
        return _generate_ollama(prompt)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose: gemini | ollama")


def _generate_gemini(prompt: str) -> str:
    """
    Call Gemini's text generation API (not embeddings — actual chat/generation).

    This uses gemini-1.5-flash — fast and free tier friendly.
    """
    try:
        from google import genai
    except ImportError:
        raise ImportError("Run: pip install google-genai")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="models/gemini-2.0-flash",
        contents=prompt,
    )

    return response.text


def _generate_ollama(prompt: str, model: str = "llama3.2:3b") -> str:
    """
    Call local Ollama server for text generation.
    Uses /api/generate endpoint (different from /api/embeddings).
    """
    try:
        import requests
    except ImportError:
        raise ImportError("Run: pip install requests")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    endpoint = f"{base_url}/api/generate"

    payload = {
        "model" : model,
        "prompt": prompt,
        "stream": False,   # wait for full response, don't stream tokens
    }

    resp = requests.post(endpoint, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


def generate_all_summaries(matches: list[dict], mode: str = "gemini") -> None:
    """
    Generate and print summaries for all matched pairs.

    Parameters
    ----------
    matches : list of match dicts from comparator.compare_sources()
    mode    : "gemini" or "ollama"
    """
    if not matches:
        print("[Generator] No matches to analyse.")
        return

    print("\n" + "=" * 70)
    print(f"  RAG GENERATION REPORT  |  model: {mode.upper()}")
    print("=" * 70)

    for i, match in enumerate(matches, start=1):
        doc_a  = match["doc_a"]
        doc_b  = match["doc_b"]
        score  = match["score"]

        print(f"\n  [{i}/{len(matches)}] Analysing match  "
              f"{doc_a['id']} <-> {doc_b['id']}  (score: {score:.4f})")
        print(f"  Jira  : {doc_a['title']}")
        print(f"  GitHub: {doc_b['title']}")
        print(f"  {'─' * 60}")

        try:
            summary = generate_summary(match, mode=mode)
            # Indent each line for clean output
            for line in summary.strip().split("\n"):
                print(f"  {line}")
        except Exception as e:
            print(f"  [ERROR] Could not generate summary: {e}")

        print()
        # Small delay between API calls to avoid rate limits
        if i < len(matches):
            time.sleep(1)

    print("=" * 70 + "\n")
