# AI POCs

A collection of Proof-of-Concept projects built using LLMs, RAG pipelines, and vector databases.  
Each project is self-contained with its own setup instructions and documentation.

---

## Projects

| # | Project | Description | Stack |
|---|---|---|---|
| 1 | [Content Similarity Comparator](./content-similarity-comparator) | Compares Jira tickets and GitHub issues using semantic embeddings | Python, ChromaDB, Gemini, Ollama |
| 2 | [AI Doc Agent](./ai-doc-agent) | Multi-source intelligent document retrieval agent with local-first RAG strategy | Python, LangChain, Gemini API, FAISS |

---

## Project Details

### 1. Content Similarity Comparator
> Compares Jira tickets and GitHub issues using semantic embeddings to identify duplicate or related content.

- **Stack:** Python, ChromaDB, Google Gemini API, Ollama (`llama3.2:3b`)
- **Key Features:**
  - Dual backend support — cloud (Gemini) and local (Ollama)
  - Cosine similarity scoring with tuned thresholds per model
  - Vector caching via ChromaDB
  - RAG-based generation layer
- **Folder:** [`content-similarity-comparator/`](./content-similarity-comparator)

---

### 2. AI Doc Agent
> End-to-end intelligent document querying system with multi-source retrieval, full source attribution, and a browser-based UI — built entirely on a free-tier stack.

- **Stack:** Python, LangChain, Google Gemini API, FAISS
- **Key Features:**
  - Local-first RAG strategy — private documents are always checked before the internet
  - Automatic Wikipedia fallback when answer is not found in local documents
  - Live web content fetching when a URL is provided in the query
  - Full source attribution and transparent reasoning on every response
  - Agent routing behaviour controlled entirely through prompt engineering
  - Clean browser-based UI with no infrastructure cost
- **Key Observations:**
  - Prompt engineering drives agent behaviour — tool selection, fallback logic, and stopping conditions are all controlled via the system prompt
  - Query phrasing directly affects retrieval quality — a fundamental RAG characteristic tied to chunk boundaries and embedding similarity
- **Folder:** [`ai-doc-agent/`](./ai-doc-agent)

---

<!--
  TO ADD A NEW POC:
  1. Add a new row in the Projects table above
  2. Add a new "### N. Project Name" section below
  3. Create the folder and add its own README.md
-->

---

## Tech Stack (Overall)

| Category | Tools |
|---|---|
| Language | Python |
| LLM APIs | Google Gemini API, Ollama |
| Vector DB | ChromaDB, FAISS |
| Frameworks | LangChain, Flask |
| Integrations | Jira API, GitHub API, Wikipedia API |

---

## Author

**Devika Shetty**  
[GitHub](https://github.com/devikashetty1710)
