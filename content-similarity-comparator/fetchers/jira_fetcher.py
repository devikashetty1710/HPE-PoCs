"""
jira_fetcher.py
---------------
Fetches real tickets from your Jira project using the Jira REST API.

ANALOGY: Think of this as a postal worker who goes to the Jira office,
shows their ID badge (API token), asks for all tickets in a project,
and brings them back in the same format our loader.py already understands.

KEY TERMS:
  - REST API   : a way to talk to a web service using URLs. Like visiting
                 a specific web address to get data back as JSON.
  - Basic Auth : sending your email + token with every request so the
                 server knows who you are. Like showing ID at a door.
  - JSON       : the format Jira sends data back in. Our code converts
                 it into the same dict format loader.py produces.
  - JQL        : Jira Query Language — like SQL but for Jira tickets.
                 e.g. 'project = DEV ORDER BY created DESC'
"""

import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()


def fetch_jira_tickets(max_results: int = 50) -> list[dict]:
    """
    Fetch tickets from your Jira project and return them in the same
    format as loader.parse_documents() so the rest of the pipeline
    works without any changes.

    Returns
    -------
    list[dict] with keys: id, title, content, full_text
    """

    # ── Read credentials from .env ────────────────────────────────────────
    base_url  = os.getenv("JIRA_BASE_URL")        # e.g. https://yourname.atlassian.net
    email     = os.getenv("JIRA_EMAIL")            # your Jira login email
    api_token = os.getenv("JIRA_API_TOKEN")        # token from id.atlassian.com
    project   = os.getenv("JIRA_PROJECT_KEY", "DEV")  # e.g. DEV

    # Validate all credentials are present
    if not all([base_url, email, api_token, project]):
        raise EnvironmentError(
            "Missing Jira credentials in .env!\n"
            "Need: JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY"
        )

    # ── Build the API request ─────────────────────────────────────────────
    # Jira REST API endpoint for searching issues
    # Docs: https://developer.atlassian.com/cloud/jira/platform/rest/v3/api-group-issue-search
    url = f"{base_url}/rest/api/3/search/jql"

    # JQL = Jira Query Language (like SQL for Jira)
    # This fetches all issues in your project, newest first
    jql = f"project = {project} ORDER BY created DESC"

    # Query parameters sent with the request
    params = {
        "jql"        : jql,
        "maxResults" : max_results,
        # Only fetch the fields we need (saves bandwidth)
        "fields"     : "summary,description,issuetype,status",
    }

    # HTTPBasicAuth sends email:token with every request (Jira requires this)
    auth = HTTPBasicAuth(email, api_token)

    # Headers tell Jira we want JSON back
    headers = {"Accept": "application/json"}

    print(f"[Jira Fetcher] Connecting to {base_url} ...")
    print(f"[Jira Fetcher] Fetching project '{project}' tickets ...")

    try:
        response = requests.get(url, params=params, auth=auth, headers=headers, timeout=30)
        response.raise_for_status()  # raises an error if status is 4xx or 5xx
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Cannot reach Jira at {base_url}. Check your JIRA_BASE_URL.")
    except requests.exceptions.HTTPError as e:
        raise PermissionError(
            f"Jira API error: {e}\n"
            "Check your JIRA_EMAIL and JIRA_API_TOKEN are correct."
        )

    # ── Parse the response ────────────────────────────────────────────────
    data   = response.json()
    issues = data.get("issues", [])

    print(f"[Jira Fetcher] Found {len(issues)} tickets in project '{project}'")

    documents = []
    for issue in issues:
        fields = issue["fields"]

        # Issue key = e.g. DEV-4, DEV-5
        doc_id = issue["key"]

        # Summary = the ticket title
        title = fields.get("summary", "").strip()

        # Description in Jira Cloud is in "Atlassian Document Format" (ADF)
        # It's a nested JSON structure. We extract plain text from it.
        content = extract_adf_text(fields.get("description"))

        # Skip tickets with no title
        if not title:
            continue

        documents.append({
            "id"       : doc_id,
            "title"    : title,
            "content"  : content,
            "full_text": f"{title}. {content}",  # same format as loader.py
        })

    print(f"[Jira Fetcher] Loaded {len(documents)} valid tickets")
    return documents


def extract_adf_text(adf: dict | None) -> str:
    """
    Jira Cloud stores descriptions in ADF (Atlassian Document Format).
    This is a nested JSON tree. We recursively walk it and extract
    all plain text nodes.

    Example ADF structure:
    {
      "type": "doc",
      "content": [
        {
          "type": "paragraph",
          "content": [
            {"type": "text", "text": "This is the description."}
          ]
        }
      ]
    }

    Parameters
    ----------
    adf : dict | None — the description field from Jira API

    Returns
    -------
    str — plain text extracted from the ADF tree
    """
    if not adf:
        return ""

    texts = []

    def walk(node):
        if isinstance(node, dict):
            # If this node has text, grab it
            if node.get("type") == "text":
                texts.append(node.get("text", ""))
            # Recurse into content children
            for child in node.get("content", []):
                walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(adf)
    return " ".join(texts).strip()
