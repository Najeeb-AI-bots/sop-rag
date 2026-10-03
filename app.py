"""
SOP-RAG - LCI SOP & Policy Assistant (Streamlit)

A production-style Retrieval-Augmented Generation app: ask a natural-language
policy question, get an answer grounded ONLY in the indexed SOPs, with cited
sources - and a clear "Not found in current SOPs" refusal when the answer
isn't there. Includes a live evaluation tab (recall@k, faithfulness, refusal
accuracy, latency).

Free, no-key by default (local embeddings + BM25 + grounded composer).
Optional Anthropic key enables Claude-written prose. Synthetic SOP data only.
Built by Mohammed Abdul Najeeb.
"""

import os
import time
import streamlit as st

from rag.retrieve import retrieve
from rag.generate import answer

st.set_page_config(page_title="SOP-RAG Assistant", page_icon="📚", layout="wide")

st.markdown("""
<style>
  .stApp { background: linear-gradient(160deg,#0f1420 0%,#1b2a3a 100%); }
  h1,h2,h3,h4,p,label,.stMarkdown { color:#e8ecf3 !important; }
  .hero { background:linear-gradient(135deg,#1B2A3A 0%,#5B3CC4 100%);
    border-radius:18px; padding:24px 30px; margin-bottom:20px;
    box-shadow:0 10px 30px rgba(0,0,0,.35); }
  .hero h1 { color:#fff !important; margin:0; font-size:28px; font-weight:800; }
  .hero p { color:rgba(255,255,255,.9)!important; margin:6px 0 0; font-size:13px; }
  .card { background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.12);
    border-radius:14px; padding:16px 18px; margin-bottom:14px; }
  .src { background:rgba(91,60,196,.18); border:1px solid rgba(91,60,196,.5);
    border-radius:8px; padding:3px 10px; font-size:12px; margin-right:6px; color:#c9b8ff; }
  .stButton>button { background:linear-gradient(135deg,#5B3CC4,#0b84ff); color:#fff;
    border:none; border-radius:10px; font-weight:700; padding:9px 18px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>📚 SOP-RAG - LCI SOP & Policy Assistant</h1>
  <p>Ask a policy question - get a grounded, cited answer from the indexed SOPs,
  or an honest "Not found" refusal. Hybrid retrieval (semantic + keyword) +
  reranking + evaluation. Synthetic demo data.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Settings")
    provider = st.selectbox("Answer engine",
                            ["Free (grounded, no key)", "Anthropic Claude (your key)"])
    api_key = ""
    if provider.startswith("Anthropic"):
        api_key = st.text_input("Anthropic API key", type="password",
                                help="Used only in this session, never stored.")
    top_k = st.slider("Chunks to retrieve (top-k)", 3, 10, 5)
    st.markdown("---")
    st.caption("Hybrid = dense embeddings + BM25 keyword, fused (RRF) and "
               "reranked. Degrades to keyword-only if ML libs aren't installed. "
               "Built by Mohammed Abdul Najeeb.")

tab_ask, tab_eval, tab_about = st.tabs(["🔎 Ask", "📊 Evaluation", "ℹ️ How it works"])

# ----------------------------- ASK TAB -----------------------------
with tab_ask:
    st.subheader("Ask a policy / SOP question")
    examples = [
        "A seller's product was removed because the listing expired. Which policy and NRR Column R name applies?",
        "Tier 1 OnCall couldn't resolve within SLA - what do I do and what must the email include?",
        "A customer review violates policy - can an LCI agent action it directly?",
        "What is the policy for counterfeit luxury handbags?",  # refusal demo
    ]
    pick = st.selectbox("Try an example (or type your own below)", [""] + examples)
    q = st.text_input("Your question", value=pick)

    if st.button("🔎 Answer", type="primary") and q.strip():
        t0 = time.time()
        with st.spinner("Retrieving & grounding…"):
            retrieved = retrieve(q, top_n=top_k)
            out = answer(q, retrieved, api_key=api_key)
        dt = (time.time() - t0) * 1000

        refused = out["answer"].strip().startswith("Not found")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if refused:
            st.warning("🚫 " + out["answer"])
        else:
            st.markdown("#### Answer")
            st.write(out["answer"])
        st.markdown('</div>', unsafe_allow_html=True)

        meta = f"retrieval: **{retrieved['mode']}** · engine: **{out['engine']}** · {dt:.0f} ms"
        st.caption(meta)

        if out["citations"]:
            st.markdown("**Sources**")
            st.markdown(" ".join(f"<span class='src'>{s}</span>" for s in out["citations"]),
                        unsafe_allow_html=True)

        with st.expander("See retrieved chunks (what the model saw)"):
            for c in retrieved["chunks"]:
                st.markdown(f"**[{c['source']}]**")
                st.text(c["text"])

# ----------------------------- EVAL TAB -----------------------------
with tab_eval:
    st.subheader("📊 Evaluation — how do we know it works?")
    st.caption("Runs the golden set (in evals/golden_set.jsonl) and scores "
               "retrieval + grounding. This is the 'senior' signal most demos skip.")
    if st.button("Run evaluation"):
        from rag.evaluate import run, write_report
        with st.spinner("Scoring golden set…"):
            metrics, details = run(api_key=api_key)
            write_report(metrics, details)
        c = st.columns(4)
        c[0].metric(f"Recall@{5}", f"{metrics['recall@5']:.0%}", "target 85%")
        c[1].metric("Faithfulness", f"{metrics['faithfulness']:.0%}", "target 90%")
        c[2].metric("Citation rate", f"{metrics['citation_rate']:.0%}", "target 100%")
        c[3].metric("Refusal acc.", f"{metrics['refusal_accuracy']:.0%}", "target 90%")
        st.caption(f"mode: {metrics['mode']} · p50 {metrics['p50_latency_s']*1000:.0f} ms "
                   f"· p95 {metrics['p95_latency_s']*1000:.0f} ms · "
                   f"{metrics['n_questions']} questions")
        st.success("Report written to evals/eval_report.md")

# ----------------------------- ABOUT TAB -----------------------------
with tab_about:
    st.markdown("""
### How this RAG system works
1. **Ingest** — SOP docs are split into overlapping chunks, embedded, and stored
   in a vector database.
2. **Retrieve (hybrid)** — your question runs through BOTH semantic (embeddings)
   and keyword (BM25) search; the two ranked lists are fused with Reciprocal
   Rank Fusion (RRF). Hybrid matters because SOPs mix fuzzy policy language
   (needs semantic) with exact codes/GUIDs/case-IDs (needs keyword).
3. **Rerank** — a cross-encoder re-scores the ~20 candidates down to the best 5.
4. **Generate (grounded)** — the answer is composed using ONLY the retrieved
   chunks, cites its sources, and refuses ("Not found in current SOPs") when the
   answer isn't in the context.
5. **Evaluate** — a golden set measures recall@k, faithfulness, citation rate,
   refusal accuracy, and latency.

All SOP content here is **synthetic** — no real data.
""")
