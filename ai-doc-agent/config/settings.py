"""
config/settings.py

Central configuration loaded from environment variables.
All modules import from here — no scattered os.getenv() calls.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- LLM ---
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # --- ChromaDB ---
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "local_docs")

    # --- Document Sources ---
    LOCAL_DOCS_DIR: str = os.getenv("LOCAL_DOCS_DIR", "./sample_docs")
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", 4000))

    # --- Agent ---
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", 8))
    VERBOSE: bool = os.getenv("VERBOSE", "true").lower() == "true"

    @classmethod
    def validate(cls) -> None:
        if cls.LLM_PROVIDER == "gemini" and not cls.GOOGLE_API_KEY:
            raise EnvironmentError("GOOGLE_API_KEY is not set.")
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            raise EnvironmentError("GROQ_API_KEY is not set.")


settings = Settings()
