"""
github_fetcher.py
-----------------
Fetches real issues from your GitHub repository using the GitHub REST API.

ANALOGY: Same postal worker, different office. This time they go to
GitHub, show their Personal Access Token as ID, grab all open issues
from your repo, and return them in the exact same dict format so
the comparator doesn't know or care where the data came from.

KEY TERMS:
  - Personal Access Token (PAT): a secret string GitHub gives you to
    authenticate API calls. Starts with 'ghp_'.
  - Bearer token: a way of sending your token in the request header.
    Like saying "I have this badge, let me in."
  - Pagination: GitHub returns max 100 issues per request. If you have
    more, you need to ask for the next "page". We handle this below.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()


def fetch_github_issues(max_results: int = 50) -> list[dict]:
    """
    Fetch open issues from your GitHub repository and return them
    in the same format as loader.parse_documents().

    Returns
    -------
    list[dict] with keys: id, title, content, full_text
    """

    # ── Read credentials from .env ────────────────────────────────────────
    token = os.getenv("GITHUB_TOKEN")    # ghp_...
    owner = os.getenv("GITHUB_OWNER")   # your GitHub username
    repo  = os.getenv("GITHUB_REPO")    # e.g. rag-test-issues

    if not all([token, owner, repo]):
        raise EnvironmentError(
            "Missing GitHub credentials in .env!\n"
            "Need: GITHUB_TOKEN, GITHUB_OWNER, GITHUB_REPO"
        )

    # ── Build the API request ─────────────────────────────────────────────
    # GitHub REST API endpoint for listing issues
    # Docs: https://docs.github.com/en/rest/issues/issues#list-repository-issues
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"

    # Request headers
    headers = {
        "Authorization": f"Bearer {token}",  # our PAT as a bearer token
        "Accept"       : "application/vnd.github+json",  # tells GitHub API version
        "X-GitHub-API-Version": "2022-11-28",
    }

    # Query parameters
    params = {
        "state"   : "open",         # only fetch open issues
        "per_page": min(max_results, 100),  # GitHub max is 100 per page
        "sort"    : "created",
        "direction": "desc",
    }

    print(f"[GitHub Fetcher] Connecting to GitHub API ...")
    print(f"[GitHub Fetcher] Fetching issues from '{owner}/{repo}' ...")

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot reach GitHub API. Check your internet connection.")
    except requests.exceptions.HTTPError as e:
        raise PermissionError(
            f"GitHub API error: {e}\n"
            "Check your GITHUB_TOKEN, GITHUB_OWNER, and GITHUB_REPO are correct."
        )

    # ── Parse the response ────────────────────────────────────────────────
    issues = response.json()

    # GitHub returns pull requests in the issues endpoint too!
    # Filter them out — we only want real issues
    issues = [i for i in issues if "pull_request" not in i]

    print(f"[GitHub Fetcher] Found {len(issues)} open issues in '{owner}/{repo}'")

    documents = []
    for issue in issues:
        doc_id  = f"GH-{issue['number']}"           # e.g. GH-1, GH-2
        title   = issue.get("title", "").strip()
        content = issue.get("body") or ""            # body can be None in GitHub
        content = content.strip()

        if not title:
            continue

        documents.append({
            "id"       : doc_id,
            "title"    : title,
            "content"  : content,
            "full_text": f"{title}. {content}",      # same format as loader.py
        })

    print(f"[GitHub Fetcher] Loaded {len(documents)} valid issues")
    return documents
