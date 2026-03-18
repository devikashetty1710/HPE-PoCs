# ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DOCS_DIR = "sample_docs"
FAISS_DIR = "faiss_db"


def load_all_documents():
    all_docs = []

    for filename in os.listdir(DOCS_DIR):
        filepath = os.path.join(DOCS_DIR, filename)
        print(f"Loading: {filename}")

        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.endswith(".txt"):
                loader = TextLoader(filepath)
            elif filename.endswith(".md"):
                loader = TextLoader(filepath)
            elif filename.endswith(".csv"):
                loader = CSVLoader(filepath)
            elif filename.endswith(".json"):
                loader = TextLoader(filepath)
            else:
                print(f"  Skipping unsupported file: {filename}")
                continue

            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = filename

            all_docs.extend(docs)
            print(f"  Loaded {len(docs)} document(s) from {filename}")

        except Exception as e:
            print(f"  Error loading {filename}: {e}")

    return all_docs


def ingest():
    print("\n--- Starting document ingestion ---\n")
    docs = load_all_documents()

    if not docs:
        print("No documents found in sample_docs/ folder.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print(f"\nTotal chunks created: {len(chunks)}")

    print("Embedding and storing in FAISS...")
    print("Downloading HuggingFace model on first run — please wait...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    vectorstore.save_local(FAISS_DIR)
    print(f"\nIngestion complete. {len(chunks)} chunks saved to {FAISS_DIR}/")


if __name__ == "__main__":
    ingest()