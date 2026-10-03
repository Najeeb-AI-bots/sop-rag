# SOP-RAG — Evaluation Report

Golden set: **16 questions** · retrieval mode: **bm25-only (fallback)** · top-k = 5

> Reproduced with the pure-Python BM25 fallback so results are environment-independent. With the full embedding + cross-encoder stack installed, retrieval quality is equal or better.

| Metric | Value | Target |
|---|---|---|
| Recall@5 | 100% | ≥ 85% |
| Faithfulness | 100% | ≥ 90% |
| Citation rate | 100% | 100% |
| Refusal accuracy | 100% | ≥ 90% |
| p50 latency | 0 ms | report |
| p95 latency | 2 ms | report |

## Per-question detail
| Question | Scope | Recall | Cited | Faithful | Refused |
|---|---|---|---|---|---|
| A seller's product was removed because the listi | in | ✅ | ✅ | ✅ | ❌ |
| What GUID goes in NRR Column R for expired produ | in | ✅ | ✅ | ✅ | ❌ |
| Units are sitting in the FC but the offer is ina | in | ✅ | ✅ | ✅ | ❌ |
| What's the descriptive name and GUID for strande | in | ✅ | ✅ | ✅ | ❌ |
| Tier 1 OnCall could not resolve within SLA. What | in | ✅ | ✅ | ✅ | ❌ |
| Does LCI use watchers to escalate cases? | in | ✅ | ✅ | ✅ | ❌ |
| Who owns routing to the correct resolver group i | in | ✅ | ✅ | ✅ | ❌ |
| A customer review violates policy. Can an LCI ag | in | ✅ | ✅ | ✅ | ❌ |
| What's the Column R name and GUID for a policy-v | in | ✅ | ✅ | ✅ | ❌ |
| How should I code time spent in a documented coa | in | ✅ | ✅ | ✅ | ❌ |
| An agent logged a platform outage as a break. Is | in | ✅ | ✅ | ✅ | ❌ |
| A product shows up in the wrong category in sear | in | ✅ | ✅ | ✅ | ❌ |
| What is the GUID for a browse node update? | in | ✅ | ✅ | ✅ | ❌ |
| What is the refund window for a customer return  | out | — | — | — | ✅ |
| What is the policy for counterfeit luxury handba | out | — | — | — | ✅ |
| How many vacation days do associates get per yea | out | — | — | — | ✅ |
