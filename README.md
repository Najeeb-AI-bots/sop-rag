# 📚 SOP-RAG — a Production RAG Assistant for SOPs & Policy

> Ask a natural-language policy question → get an answer grounded **only** in the indexed SOPs, with **cited sources** — and an honest **"Not found in current SOPs"** refusal when the answer isn't there. Hybrid retrieval + reranking + a real evaluation suite.

**Built by:** [Mohammed Abdul Najeeb](https://github.com/Najeeb-AI-bots) · Operations Manager → AI-Transformation builder

> 💡 Grounded in real contact-center operations: the demo corpus is a set of **synthetic** SOP/policy docs (expired products, stranded inventory, escalation tiers, customer reviews, NPT coding, browse-node updates). No real data.

---

## Why this project

A plain LLM hallucinates policy answers and can't cite a source — useless in an audit/compliance context. **RAG fixes both:** it retrieves the real policy from your own documents and answers using only that, with citations, or refuses. This repo is a worked, measured example — not a "GPT wrapper."

## What makes it production-grade (not a demo)

| Capability | How |
|---|---|
| **Hybrid retrieval** | semantic (embeddings) **+** keyword (BM25), fused with Reciprocal Rank Fusion (RRF) |
| **Reranking** | a cross-encoder re-scores the top ~20 candidates → best 5 |
| **Grounding** | answer composed from retrieved chunks **only** |
| **Citations** | every answer names its source SOP file |
| **Refusal** | out-of-scope questions get "Not found in current SOPs" instead of a hallucination |
| **Evaluation** | a golden set scores recall@k, faithfulness, citation rate, refusal accuracy, latency |

### Why **hybrid** specifically
SOP questions mix two kinds of matching:
- **Fuzzy policy language** ("listing expired" ≈ "past its expiry window") → needs **semantic** embeddings.
- **Exact identifiers** (knowledge-link GUIDs, category codes, case IDs) → needs **keyword** BM25.

Dense alone misses exact strings; BM25 alone misses paraphrases. **Hybrid gets both** — the right architecture for a policy/compliance corpus.

## 📊 Evaluation (the "how do you know it works?" signal)

`python -m rag.evaluate` runs the golden set and writes [`evals/eval_report.md`](evals/eval_report.md):

| Metric | Target | Result (keyword-only fallback) |
|---|---|---|
| Recall@5 | ≥ 85% | **100%** |
| Faithfulness | ≥ 90% | **100%** |
| Citation rate | 100% | **100%** |
| Refusal accuracy | ≥ 90% | **100%** |

> Results shown are from the pure-Python BM25 fallback (so they reproduce anywhere). With the full embedding + reranker stack installed, retrieval quality is equal or better.

## Architecture

```
INGEST (offline):  SOP .md → chunk (overlap) → embed → Chroma vector store
QUERY:             question ─┬─► dense (embeddings) ─┐
                             └─► BM25 (keyword)  ─────┴─► RRF fuse → rerank → top-5
                                                                              │
                             grounded prompt ("use ONLY context, cite, refuse")
                                                                              │
                                                                     cited answer
EVAL:              golden set → recall@k · faithfulness · refusal acc · latency
```

## Run locally

```bash
pip install -r requirements.txt
python -m rag.ingest        # build the chunk list + vector store
python -m rag.evaluate      # run the eval suite → evals/eval_report.md
streamlit run app.py        # launch the UI
```

> **No API key needed.** The default engine is a free, local, grounded composer (embeddings + BM25). Add an Anthropic key in the sidebar to have Claude write the prose — grounding, citations, and refusal behave identically either way.

> **Graceful degradation:** if `sentence-transformers` / `chromadb` aren't installed, the app automatically falls back to BM25 keyword retrieval (with a pure-Python BM25 if `rank-bm25` is also missing), so it runs anywhere.

## Project layout

```
sop-rag/
├── app.py                 # Streamlit UI: Ask · Evaluation · How-it-works tabs
├── rag/
│   ├── ingest.py          # load → chunk (overlap) → embed → store
│   ├── retrieve.py        # hybrid dense+BM25 → RRF → rerank
│   ├── generate.py        # grounded answer: cite or refuse
│   └── evaluate.py        # golden-set metrics → eval_report.md
├── data/sops/             # synthetic SOP corpus (.md)
├── evals/golden_set.jsonl # Q → expected-source pairs (incl. refusal cases)
└── requirements.txt · README.md · LICENSE
```

## Skills demonstrated

RAG system design · hybrid retrieval (dense + BM25 + RRF) · reranking · grounding & citation · hallucination refusal · **LLM evaluation** (golden sets, recall@k, faithfulness) · free/no-key architecture with optional BYO-key.

## License

MIT.
