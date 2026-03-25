from pathlib import Path
import argparse
import sys
import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

HF_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_MODEL_NAME = os.getenv("HUGGING_FACE_EMBEDDING_MODEL") or DEFAULT_EMBED_MODEL
LLM_MODEL_NAME = os.getenv("LLM_MODEL")

INDEX_DIR = ROOT_DIR / "data_index"
DATA_DIR = ROOT_DIR / "data"


def load_documents(data_dir: Path):
    docs = []
    for path in data_dir.rglob("*"):
        if path.is_dir():
            continue
        try:
            if path.suffix.lower() in [".txt", ".md"]:
                docs.extend(TextLoader(str(path), encoding="utf-8").load())
            elif path.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(path)).load())
        except Exception as e:
            print(f"[skip] {path.name}: {e}", file=sys.stderr)
    if not docs:
        raise RuntimeError(f"No documents found in {data_dir}. Put .txt/.md/.pdf files there.")
    return docs

def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    vs = FAISS.from_documents(chunks, embeddings)
    return vs

def main():
    parser = argparse.ArgumentParser(description="Ingest documents and build FAISS index.")
    args = parser.parse_args()



    print(f"Loading documents from {DATA_DIR}")
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents. Building index…")

    vs = build_vectorstore(docs)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))

    # Persist embedding model name for safety
    (INDEX_DIR / "embeddings_model.txt").write_text(EMBED_MODEL_NAME, encoding="utf-8")

    print(f"Index saved to {INDEX_DIR.resolve()}")

if __name__ == "__main__":
    main()
