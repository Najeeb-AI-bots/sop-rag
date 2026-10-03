"""
retrieve.py - hybrid retrieval: dense (embeddings) + BM25 (keyword),
fused with Reciprocal Rank Fusion (RRF), then reranked with a cross-encoder.

Why hybrid: SOP questions mix fuzzy policy language (needs semantic/dense) with
exact identifiers like GUIDs, category codes, case IDs (needs keyword/BM25).
Dense alone misses exact strings; BM25 alone misses paraphrases. RRF gets both.

Graceful degradation: if sentence-transformers / chromadb / the reranker are not
installed, this falls back to BM25-only (and a pure-Python BM25 if rank_bm25 is
missing), so retrieval logic is testable in any environment.
"""

import os
import re
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STORE_DIR = os.path.join(ROOT, "data", "store")
CHUNKS_JSON = os.path.join(STORE_DIR, "chunks.json")
CHROMA_DIR = os.path.join(STORE_DIR, "chroma")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION = "sops"


def _tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def load_chunks():
    if os.path.exists(CHUNKS_JSON):
        return json.load(open(CHUNKS_JSON, encoding="utf-8"))
    # fall back to building the chunk list on the fly
    from rag.ingest import load_chunks as _lc
    chunks = _lc()
    for i, c in enumerate(chunks):
        c["id"] = f"chunk-{i}"
    return chunks


# --------------------------------------------------------------------------- #
#  Dense retrieval (embeddings via Chroma). Returns [] if unavailable.
# --------------------------------------------------------------------------- #
def dense_search(query, chunks, k=20):
    try:
        import chromadb
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
        qemb = model.encode([query]).tolist()
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        col = client.get_collection(COLLECTION)
        res = col.query(query_embeddings=qemb, n_results=min(k, len(chunks)))
        return list(res["ids"][0])  # ranked chunk ids
    except Exception:
        return []


# --------------------------------------------------------------------------- #
#  BM25 keyword retrieval. Uses rank_bm25 if present, else a pure-Python BM25.
# --------------------------------------------------------------------------- #
def bm25_search(query, chunks, k=20):
    corpus = [_tokens(c["text"]) for c in chunks]
    q = _tokens(query)
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(q)
    except Exception:
        scores = _bm25_scores(q, corpus)
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i]["id"] for i in ranked[:k]]


def _bm25_scores(q, corpus, k1=1.5, b=0.75):
    """Minimal pure-Python BM25 fallback (no external dependency)."""
    N = len(corpus)
    avgdl = sum(len(d) for d in corpus) / N if N else 0
    # document frequency
    df = {}
    for doc in corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    scores = [0.0] * N
    for i, doc in enumerate(corpus):
        tf = {}
        for t in doc:
            tf[t] = tf.get(t, 0) + 1
        dl = len(doc)
        s = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / avgdl) if avgdl else 1
            s += idf * (tf[term] * (k1 + 1)) / denom
        scores[i] = s
    return scores


# --------------------------------------------------------------------------- #
#  RRF fusion + rerank
# --------------------------------------------------------------------------- #
def rrf_fuse(rankings, k=60):
    """Reciprocal Rank Fusion: combine multiple ranked id-lists into one."""
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0) + 1.0 / (k + rank + 1)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def rerank(query, chunk_ids, chunks, top_n=5):
    """Cross-encoder rerank of candidates; falls back to input order."""
    by_id = {c["id"]: c for c in chunks}
    cands = [by_id[cid] for cid in chunk_ids if cid in by_id]
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(RERANK_MODEL)
        pairs = [[query, c["text"]] for c in cands]
        scores = ce.predict(pairs)
        order = sorted(range(len(cands)), key=lambda i: scores[i], reverse=True)
        return [cands[i] for i in order[:top_n]]
    except Exception:
        return cands[:top_n]


def retrieve(query, top_n=5, candidate_k=20):
    """Full hybrid pipeline -> top_n reranked chunks (list of chunk dicts)."""
    chunks = load_chunks()
    dense = dense_search(query, chunks, k=candidate_k)
    sparse = bm25_search(query, chunks, k=candidate_k)
    rankings = [r for r in (dense, sparse) if r]
    fused = rrf_fuse(rankings) if rankings else [c["id"] for c in chunks]
    reranked = rerank(query, fused[:candidate_k], chunks, top_n=top_n)
    mode = "hybrid" if dense and sparse else ("bm25-only" if sparse else "fallback")
    return {"chunks": reranked, "mode": mode}


if __name__ == "__main__":
    out = retrieve("a seller's product was removed because the listing expired")
    print("mode:", out["mode"])
    for c in out["chunks"]:
        print(f"  [{c['source']}] {c['text'][:70]}...")
