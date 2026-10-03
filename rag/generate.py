"""
generate.py - grounded answer generation.

Takes the retrieved chunks and produces an answer that:
  1. uses ONLY the retrieved context (no outside knowledge),
  2. cites the source file of each chunk it used,
  3. REFUSES ("Not found in current SOPs") when the context is irrelevant.

Two engines, same contract:
  - Anthropic Claude (if an API key is provided) - fluent prose.
  - Free grounded composer (default, no key) - extractive, deterministic.
Both enforce grounding + citation + refusal, so the trust behavior is identical.
"""

import re

REFUSAL = "Not found in current SOPs."
# If the best content-word overlap is below this, refuse (out-of-scope).
# Chosen from the golden set: in-scope >= 0.44, out-of-scope <= 0.25.
MIN_RELEVANCE = 0.35

# Generic question/stop words are ignored when judging relevance, so words
# like "what/is/the/policy" don't create false relevance on out-of-scope
# questions (which otherwise prevents refusal from ever triggering).
STOPWORDS = set(
    "a an the of to is are was for and or in on at by with what which how do "
    "does i you we it this that applies apply name should can could would when "
    "who whom its into from be as get got my your their his her they them s "
    "whats".split()
)


def _tokens(t):
    return set(re.findall(r"[a-z0-9]+", t.lower()))


def _overlap_score(query, text):
    q, d = _tokens(query) - STOPWORDS, _tokens(text)
    return len(q & d) / len(q) if q else 0.0


def _grounded_free(query, chunks):
    """Deterministic, no-key composer: stitch the most relevant chunk(s) with
    citations. Refuses if nothing is relevant enough."""
    scored = sorted(((_overlap_score(query, c["text"]), c) for c in chunks),
                    key=lambda x: x[0], reverse=True)
    if not scored or scored[0][0] < MIN_RELEVANCE:
        return {"answer": REFUSAL, "citations": [], "grounded": True, "engine": "free"}

    used = [c for s, c in scored if s >= MIN_RELEVANCE][:2]
    parts, cites = [], []
    for c in used:
        parts.append(f"{c['text'].strip()}")
        if c["source"] not in cites:
            cites.append(c["source"])
    answer = "\n\n".join(parts)
    answer += "\n\nSource(s): " + ", ".join(cites)
    return {"answer": answer, "citations": cites, "grounded": True, "engine": "free"}


def _grounded_claude(query, chunks, api_key):
    try:
        import anthropic
        context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
        system = (
            "You are an SOP & policy assistant for a contact-center operations "
            "team. Answer the question using ONLY the context below. Cite the "
            "source file in square brackets for every claim. If the answer is "
            "not contained in the context, reply EXACTLY with: "
            f"'{REFUSAL}' and nothing else."
        )
        user = f"Context:\n{context}\n\nQuestion: {query}"
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-5", max_tokens=600,
            system=system, messages=[{"role": "user", "content": user}])
        text = msg.content[0].text.strip()
        cites = sorted(set(re.findall(r"\[([^\]]+\.md)\]", text)))
        return {"answer": text, "citations": cites,
                "grounded": REFUSAL not in text, "engine": "claude"}
    except Exception:
        return _grounded_free(query, chunks)


def answer(query, retrieved, api_key=""):
    """Produce a grounded answer dict from retrieved chunks."""
    chunks = retrieved.get("chunks", [])
    if not chunks:
        return {"answer": REFUSAL, "citations": [], "grounded": True,
                "engine": "none", "mode": retrieved.get("mode", "fallback")}
    if api_key:
        out = _grounded_claude(query, chunks, api_key)
    else:
        out = _grounded_free(query, chunks)
    out["mode"] = retrieved.get("mode", "fallback")
    return out


if __name__ == "__main__":
    from rag.retrieve import retrieve
    q = "What is the NRR Column R name for an expired listing and its GUID?"
    r = retrieve(q)
    print(answer(q, r)["answer"])
