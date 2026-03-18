from agent.tools.local_search_tool import local_search_tool
from agent.tools.pdf_tool import pdf_tool
from agent.tools.text_tool import text_tool
from agent.tools.json_tool import json_tool
from agent.tools.web_tool import web_tool
from agent.tools.wiki_tool import wiki_tool

# Ordered list — the agent's tool descriptions enforce local-first priority,
# but ordering here also subtly influences LLM tool selection.
ALL_TOOLS = [
    local_search_tool,  # 1st — always try local docs first
    pdf_tool,           # 2nd — for explicit PDF file reads
    text_tool,          # 3rd — for explicit text/csv file reads
    json_tool,          # 4th — for explicit JSON file reads
    wiki_tool,          # 5th — general knowledge fallback
    web_tool,           # 6th — web URL fallback (explicit URLs only)
]

__all__ = [
    "local_search_tool",
    "pdf_tool",
    "text_tool",
    "json_tool",
    "web_tool",
    "wiki_tool",
    "ALL_TOOLS",
]
