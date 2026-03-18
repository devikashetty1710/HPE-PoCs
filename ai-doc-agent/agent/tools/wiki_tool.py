"""
agent/tools/wiki_tool.py

Wikipedia search — a lightweight fallback for general knowledge questions
that are not answered by local documents.
No API key required.
"""

from langchain_core.tools import Tool


def search_wikipedia(query: str) -> str:
    """Query Wikipedia and return a summary."""
    query = query.strip()
    if not query:
        return "Error: empty query provided to Wikipedia search."

    try:
        from langchain_community.utilities import WikipediaAPIWrapper

        wiki = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000)
        result = wiki.run(query)
        return f"[Source: Wikipedia]\n\n{result}" if result else "No Wikipedia results found."

    except Exception as exc:
        # Graceful fallback
        return f"Wikipedia search failed: {str(exc)}"


wiki_tool = Tool(
    name="WikipediaSearch",
    func=search_wikipedia,
    description=(
        "FALLBACK TOOL — use only when LocalDocumentSearch returns NOT_FOUND_LOCALLY. "
        "Searches Wikipedia for general knowledge, definitions, and factual information. "
        "Input: a concise search term or question, e.g. 'LangChain framework' or "
        "'what is retrieval augmented generation'. "
        "Do NOT use this for company-specific or private documents."
    ),
)
