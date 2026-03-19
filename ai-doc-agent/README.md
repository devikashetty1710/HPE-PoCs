# AI Document Retrieval Agent

**POC | LangChain + Groq (Llama 3.3 70B) + FAISS | Python 3.10+**

An intelligent AI agent that answers questions by searching your local documents first (PDFs, TXT, JSON, CSV, Markdown) and falling back to Wikipedia or the web only when the answer is not found locally. Includes a Streamlit chat UI.

---

## Table of Contents

1. [What This Agent Does](#what-this-agent-does)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Adding Your Own Documents](#adding-your-own-documents)
8. [Running the Agent](#running-the-agent)
9. [Tool Reference](#tool-reference)
10. [Troubleshooting](#troubleshooting)

---

## What This Agent Does

```
User Query
    │
    ▼
LocalDocumentSearch  (FAISS semantic search over local files)
    │
    ├── Found (score ≥ threshold) ──────────────► LLM synthesizes answer + source
    │
    └── NOT_FOUND_LOCALLY
            │
            ├── General knowledge ─────────────► WikipediaSearch
            │
            └── Specific URL in query ─────────► WebURLLoader
                                                   │
                                                   └── LLM synthesizes final answer
```

The agent uses the **ReAct** (Reasoning + Acting) pattern — it reasons step by step, picks a tool, observes the result, and iterates until it produces a final answer.

---

## Architecture

The agent is built on a **local-first RAG (Retrieval-Augmented Generation)** pipeline:

| Stage | Module | Description |
|-------|--------|-------------|
| Load | `ingest.py` | Scans `sample_docs/`, reads all supported file types |
| Chunk | `ingest.py` | Splits text into overlapping chunks |
| Embed & Index | `ingest.py` | Embeds chunks using `all-MiniLM-L6-v2`, stores in FAISS |
| Query | `local_search_tool.py` | Cosine similarity search at query time |
| Generate | `agent_core.py` | Groq/Llama synthesizes answer from retrieved chunks |
| Report | `app.py` / `main.py` | Returns answer with source attribution and tool trace |

---

## Project Structure

```
ai-doc-agent/
├── app.py                          ← Streamlit chat UI
├── main.py                         ← Terminal entry point
├── ingest.py                       ← Index documents into FAISS
├── config/
│   └── settings.py                 ← All env variables in one place
├── agent/
│   ├── agent_core.py               ← AgentExecutor + ReAct prompt
│   └── tools/
│       ├── local_search_tool.py    ← PRIMARY — FAISS semantic search
│       ├── pdf_tool.py             ← Read a specific PDF by path
│       ├── text_tool.py            ← Read a specific TXT/MD/CSV by path
│       ├── json_tool.py            ← Read a specific JSON by path
│       ├── wiki_tool.py            ← Wikipedia fallback
│       └── web_tool.py             ← Web URL fallback
├── vectorstore/
│   └── document_indexer.py         ← Document loading and chunking logic
├── sample_docs/                    ← Drop your documents here (not committed)
│   └── .gitkeep                    ← Keeps the empty folder in Git
├── faiss_db/                       ← Auto-created FAISS index (not committed)
├── .env                            ← Your API keys (never committed)
├── .env.example                    ← Template — copy this to .env
├── .gitignore
└── requirements.txt
```

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10 or higher (3.11 recommended) |
| pip | Latest |
| Git | Any recent version |
| RAM | 8 GB minimum |

**API Key required (free):**
- Groq API Key: https://console.groq.com → API Keys → Create

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/devikashetty1710/ai-doc-agent.git
cd ai-doc-agent
```

### Step 2 — Create and activate virtual environment

```bash
# Create
python -m venv venv

# Activate (Windows PowerShell)
venv\Scripts\Activate.ps1

# Activate (macOS / Linux)
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

### Step 4 — Create your .env file

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in your Groq API key:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your-key-here
```

> **Security:** `.env` is listed in `.gitignore` and will never be committed to Git.

---

## Adding Your Own Documents

Drop any supported file into the `sample_docs/` folder before running `ingest.py`.

**Supported formats:**

| Format | Extension |
|--------|-----------|
| PDF | `.pdf` (text-based, not scanned) |
| Text | `.txt` |
| Markdown | `.md` |
| JSON | `.json` |
| CSV | `.csv` |

> **Note:** The `sample_docs/` folder is listed in `.gitignore` — your personal documents are never pushed to GitHub. The folder itself is kept in Git via a `.gitkeep` file so the structure is preserved when someone clones the repo.

### Step 5 — Index your documents

After adding files to `sample_docs/`:

```bash
python ingest.py
```

This creates the `faiss_db/` folder with your indexed embeddings. Run this every time you add new documents.

---

## Running the Agent

### Option A — Streamlit UI (recommended)

```bash
streamlit run app.py
```

Opens at **http://localhost:8501** — chat interface with source badges, tool trace, and document upload.

### Option B — Terminal

```bash
python main.py
```

Interactive query loop in the terminal with full ReAct trace output.

---

## Tool Reference

| Priority | Tool | Type | When Used |
|----------|------|------|-----------|
| 1 | `LocalDocumentSearch` | Local (Primary) | Every query — always called first |
| 2 | `PDFReader` | Local | User explicitly names a PDF file |
| 3 | `TextFileReader` | Local | User explicitly names a TXT/MD/CSV file |
| 4 | `JSONFileReader` | Local | User explicitly names a JSON file |
| 5 | `WikipediaSearch` | Web Fallback | NOT_FOUND_LOCALLY + general knowledge |
| 6 | `WebURLLoader` | Web Fallback | NOT_FOUND_LOCALLY + specific URL provided |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No FAISS index found` | Run `python ingest.py` first |
| `GROQ_API_KEY is not set` | Add your key to `.env` |
| `429 rate limit` | Wait 60 seconds and try again (Groq free tier limit) |
| `ModuleNotFoundError` | Activate venv: `venv\Scripts\Activate.ps1` then `pip install -r requirements.txt` |
| Scanned PDF returns empty | OCR not supported — use text-based PDFs only |
| Agent loops without answering | Query is ambiguous — try rephrasing with more specific keywords |

---
