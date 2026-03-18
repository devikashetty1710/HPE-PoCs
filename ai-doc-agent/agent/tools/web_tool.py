"""
agent/tools/web_tool.py

Fetches and extracts readable text from a given web URL.
Used as a FALLBACK when the answer is not found in local documents.
"""

import logging
from langchain_core.tools import Tool
from config.settings import settings

logger = logging.getLogger(__name__)


def load_url(url: str) -> str:
    """
    Fetch text content from a web URL using requests + BeautifulSoup.

    Falls back to LangChain's WebBaseLoader if bs4 fails.
    """
    url = url.strip().strip('"').strip("'")

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # --- Attempt 1: requests + BeautifulSoup ---
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; AI-Doc-Agent/1.0; "
                "+https://github.com/devikashetty1710/ai-doc-agent)"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        # Remove boilerplate tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)

        # Collapse excessive blank lines
        lines = [line for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        if not clean_text:
            raise ValueError("Extracted text is empty after cleaning.")

        return f"[Source: {url}]\n\n{clean_text[: settings.MAX_CONTENT_LENGTH]}"

    except Exception as primary_exc:
        logger.warning("Primary web fetch failed for '%s': %s", url, primary_exc)

    # --- Attempt 2: LangChain WebBaseLoader ---
    try:
        from langchain_community.document_loaders import WebBaseLoader

        loader = WebBaseLoader(url)
        docs = loader.load()
        content = " ".join(d.page_content for d in docs)
        return f"[Source: {url}]\n\n{content[: settings.MAX_CONTENT_LENGTH]}"

    except Exception as fallback_exc:
        logger.error("Fallback web fetch also failed for '%s': %s", url, fallback_exc)
        return (
            f"Error: Could not fetch content from '{url}'. "
            f"Primary error: {primary_exc}. "
            f"Fallback error: {fallback_exc}"
        )


web_tool = Tool(
    name="WebURLLoader",
    func=load_url,
    description=(
        "FALLBACK TOOL — use only when LocalDocumentSearch returns NOT_FOUND_LOCALLY. "
        "Fetches and extracts readable text content from a specific web URL. "
        "Input must be a full URL starting with https://, "
        "e.g. 'https://docs.langchain.com/introduction'. "
        "Do NOT use this for general knowledge questions — use WikipediaSearch instead."
    ),
)
