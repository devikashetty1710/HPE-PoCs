# Content Similarity Comparator

A Python-based proof of concept that compares content between Jira tickets and GitHub issues using AI embeddings to detect semantically similar items. Supports Google Gemini (cloud) and Ollama with llama3.2:3b (local) as embedding and generation backends.

---

## Project Structure

```
content_comparator/
├── main.py                  # Entry point — wires all modules together
├── main_chroma.py           # ChromaDB-backed vector store comparator
├── comparator.py            # Core similarity comparison logic
├── embedder.py              # Embedding generation (Gemini / Ollama)
├── vector_store.py          # Vector store operations
├── loader.py                # Document loading utilities
├── generator.py             # LLM-based response generation
├── reporter.py              # Output and report formatting
├── fetchers/
│   ├── github_fetcher.py    # Fetches issues from GitHub
│   └── jira_fetcher.py      # Fetches tickets from Jira
├── data/
│   ├── source_a.txt         # Sample source A (dummy data)
│   └── source_b.txt         # Sample source B (dummy data)
├── .env.example             # Environment variable template
├── .gitignore
└── requirements.txt
```

---

## Prerequisites

- Python 3.10 or higher
- Ollama installed and running locally with the llama3.2:3b model:
  ```bash
  ollama pull llama3.2:3b
  ollama serve
  ```
- Google Gemini API key (required for Gemini mode)
- Jira project access and API token (required for live mode)
- GitHub personal access token (required for live mode)

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/devikashetty1710/content-similarity-comparator.git
cd content-similarity-comparator
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```

Open `.env` and fill in your actual credentials:
```
GEMINI_API_KEY=your_gemini_api_key_here
JIRA_URL=your_jira_url_here
JIRA_EMAIL=your_jira_email_here
JIRA_API_TOKEN=your_jira_token_here
GITHUB_TOKEN=your_github_token_here
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Usage

### Basic run with dummy data (no API credentials required)
```bash
python main.py --mode dummy
```

### Run with Gemini embeddings (cloud)
```bash
pip install google-genai
python main.py --mode gemini
```

### Run with Ollama embeddings (local)
```bash
python main.py --mode ollama
```

### Run with live Jira and GitHub data

Using Gemini:
```bash
python main.py --mode gemini --source live
```

Using Ollama:
```bash
python main.py --mode ollama --source live
```

### Run with a custom similarity threshold

Gemini with threshold 0.75:
```bash
python main.py --mode gemini --source live --threshold 0.75
```

Ollama with threshold 0.60:
```bash
python main.py --mode ollama --source live --threshold 0.60
```

### Generate LLM summary for matched pairs
```bash
python main.py --mode gemini --source live --generate
```

### Save results to a JSON file
```bash
python main.py --mode gemini --source live --save
```

### Run with ChromaDB vector store
```bash
python main_chroma.py
```

---

## Notes

- The `chroma_db/` directory is auto-generated locally when running `main_chroma.py` and is excluded from version control.
- Use `--mode dummy` to test the pipeline without any API credentials.
- The default similarity threshold is `0.75` if the `--threshold` argument is not specified.
- The `--generate` flag triggers the RAG generation step, producing an LLM summary for each matched pair.
