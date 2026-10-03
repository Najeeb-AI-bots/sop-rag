"""
ingest.py - load SOP docs, chunk them (with overlap), embed, and store.

Pipeline: read .md files -> split into overlapping chunks -> build a persistent
Chroma collection of embeddings. BM25 keyword index is built at query time in
retrieve.py (it needs the same chunk list, which we persist to chunks.json).

Design: embeddings via sentence-transformers (free, local, no API key). If the
heavy libs are not installed, ingest still produces the chunk list so retrieval
can run keyword-only — the logic stays testable everywhere.
"""

import os
import re
import json
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SOP_DIR = os.path.join(ROOT, "data", "sops")
STORE_DIR = os.path.join(ROOT, "data", "store")
CHUNKS_JSON = os.path.join(STORE_DIR, "chunks.json")
CHROMA_DIR = os.path.join(STORE_DIR, "chroma")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "sops"


def chunk_text(text, source, max_chars=600, overlap=100):
    """Split on blank lines into passages, then pack into ~max_chars chunks
    with a small overlap so a claim split across a boundary isn't lost."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            # start new buffer, carrying an overlap tail from the previous chunk
            tail = buf[-overlap:] if buf else ""
            buf = (tail + "\n" + p).strip()
    if buf:
        chunks.append(buf)
    return [{"source": source, "text": c} for c in chunks]


def load_chunks():
    """Read every SOP .md and return a flat list of chunk dicts."""
    all_chunks = []
    for path in sorted(glob.glob(os.path.join(SOP_DIR, "*.md"))):
        source = os.path.basename(path)
        text = open(path, encoding="utf-8").read()
        all_chunks.extend(chunk_text(text, source))
    return all_chunks


def build(persist=True):
    """Build the chunk list (+ optional Chroma embedding store)."""
    os.makedirs(STORE_DIR, exist_ok=True)
    chunks = load_chunks()
    for i, c in enumerate(chunks):
        c["id"] = f"chunk-{i}"

    if persist:
        with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)

    # Try to build the embedding store; degrade gracefully if libs missing.
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBED_MODEL)
        embeddings = model.encode([c["text"] for c in chunks]).tolist()

        client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
        col = client.create_collection(COLLECTION)
        col.add(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks],
            metadatas=[{"source": c["source"]} for c in chunks],
        )
        status = f"Embedded {len(chunks)} chunks into Chroma ({EMBED_MODEL})."
    except Exception as e:
        status = (f"Built {len(chunks)} chunks (keyword-only mode). "
                  f"Embedding store skipped: {type(e).__name__}: {e}")

    return {"n_chunks": len(chunks), "status": status, "chunks": chunks}


if __name__ == "__main__":
    result = build()
    print(result["status"])
    print(f"Total chunks: {result['n_chunks']}")
