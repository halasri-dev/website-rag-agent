"""Load data/pages.jsonl -> chunk -> embed locally -> store in Chroma."""
import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from usage import log_usage

PAGES = Path("data/pages.jsonl")
DB_DIR = "data/chroma_db"
COLLECTION = "litestream_docs"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
MIN_WORDS = 100      # drop tiny navigation pages
CHUNK_SIZE = 1000    # characters
CHUNK_OVERLAP = 150


def load_pages():
    with PAGES.open(encoding="utf-8") as f:
        pages = [json.loads(line) for line in f]
    kept = [p for p in pages if p["words"] >= MIN_WORDS]
    print(f"Pages: {len(pages)} total, {len(kept)} kept (>= {MIN_WORDS} words)")
    return kept


def main():
    pages = load_pages()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = []
    for p in pages:
        for i, chunk in enumerate(splitter.split_text(p["text"])):
            docs.append(
                Document(
                    page_content=f'{p["title"]}\n{chunk}',
                    metadata={
                        "url": p["url"],
                        "title": p["title"],
                        "chunk_id": f'{p["url"]}#{i}',
                    },
                )
            )
    print(f"Chunks: {len(docs)}")

    # Count embedding tokens (for the cost analysis)
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    total_tokens = sum(
        len(tok(d.page_content, add_special_tokens=False)["input_ids"]) for d in docs
    )
    print(f"Embedding tokens: {total_tokens:,}")

    # Rebuild the database from scratch so re-running is always clean
    if Path(DB_DIR).exists():
        shutil.rmtree(DB_DIR)

    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True}
    )
    Chroma.from_documents(
        docs, emb, collection_name=COLLECTION, persist_directory=DB_DIR
    )
    log_usage(
        kind="ingestion",
        model=EMBED_MODEL,
        input_tokens=total_tokens,
        output_tokens=0,
        note=f"{len(pages)} pages, {len(docs)} chunks, local embeddings",
    )
    print(f"Stored in {DB_DIR}")


if __name__ == "__main__":
    main()