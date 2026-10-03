"""
evaluate.py - the eval suite (the "how do you know it works?" signal).

Runs the golden set and computes:
  - Recall@k        : did the expected source appear in the top-k retrieved?
  - Citation rate   : % of in-scope answers that cite at least one source
  - Faithfulness    : % of in-scope answers whose every cited source was in
                      the retrieved context (no uncited/outside claims)
  - Refusal accuracy: % of out-of-scope questions correctly refused
  - p50 / p95 latency per query (seconds)

Writes evals/eval_report.md. Run: python -m rag.evaluate
"""

import os
import json
import time
import statistics

from rag.retrieve import retrieve
from rag.generate import answer, REFUSAL

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLDEN = os.path.join(ROOT, "evals", "golden_set.jsonl")
REPORT = os.path.join(ROOT, "evals", "eval_report.md")

TOP_K = 5


def load_golden():
    rows = []
    for line in open(GOLDEN, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def run(api_key=""):
    rows = load_golden()
    in_scope = [r for r in rows if r["in_scope"]]
    out_scope = [r for r in rows if not r["in_scope"]]

    recall_hits = 0
    citation_hits = 0
    faithful_hits = 0
    refusal_hits = 0
    latencies = []
    details = []

    for r in rows:
        t0 = time.time()
        retrieved = retrieve(r["q"], top_n=TOP_K)
        out = answer(r["q"], retrieved, api_key=api_key)
        latencies.append(time.time() - t0)

        retrieved_sources = {c["source"] for c in retrieved["chunks"]}
        refused = out["answer"].strip().startswith(REFUSAL)

        if r["in_scope"]:
            # recall@k
            hit = r["expected_source"] in retrieved_sources
            recall_hits += int(hit)
            # citation rate
            cited = len(out["citations"]) > 0 and not refused
            citation_hits += int(cited)
            # faithfulness: every cited source was actually retrieved
            faithful = cited and all(c in retrieved_sources for c in out["citations"])
            faithful_hits += int(faithful)
            details.append((r["q"][:50], "in", hit, cited, faithful, refused))
        else:
            # refusal accuracy
            refusal_hits += int(refused)
            details.append((r["q"][:50], "out", None, None, None, refused))

    n_in = len(in_scope) or 1
    n_out = len(out_scope) or 1
    metrics = {
        "recall@%d" % TOP_K: recall_hits / n_in,
        "citation_rate": citation_hits / n_in,
        "faithfulness": faithful_hits / n_in,
        "refusal_accuracy": refusal_hits / n_out,
        "p50_latency_s": statistics.median(latencies) if latencies else 0,
        "p95_latency_s": (sorted(latencies)[int(0.95 * len(latencies)) - 1]
                          if latencies else 0),
        "mode": retrieve("expired", top_n=1)["mode"],
        "n_questions": len(rows),
    }
    return metrics, details


def write_report(metrics, details):
    lines = ["# SOP-RAG — Evaluation Report", ""]
    lines.append(f"Golden set: **{metrics['n_questions']} questions** · "
                 f"retrieval mode: **{metrics['mode']}** · top-k = {TOP_K}")
    lines.append("")
    lines.append("| Metric | Value | Target |")
    lines.append("|---|---|---|")
    lines.append(f"| Recall@{TOP_K} | {metrics['recall@%d' % TOP_K]:.0%} | ≥ 85% |")
    lines.append(f"| Faithfulness | {metrics['faithfulness']:.0%} | ≥ 90% |")
    lines.append(f"| Citation rate | {metrics['citation_rate']:.0%} | 100% |")
    lines.append(f"| Refusal accuracy | {metrics['refusal_accuracy']:.0%} | ≥ 90% |")
    lines.append(f"| p50 latency | {metrics['p50_latency_s']*1000:.0f} ms | report |")
    lines.append(f"| p95 latency | {metrics['p95_latency_s']*1000:.0f} ms | report |")
    lines.append("")
    lines.append("## Per-question detail")
    lines.append("| Question | Scope | Recall | Cited | Faithful | Refused |")
    lines.append("|---|---|---|---|---|---|")
    for q, scope, hit, cited, faithful, refused in details:
        def m(x):
            return "—" if x is None else ("✅" if x else "❌")
        lines.append(f"| {q} | {scope} | {m(hit)} | {m(cited)} | {m(faithful)} | {m(refused)} |")
    report = "\n".join(lines) + "\n"
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    return report


if __name__ == "__main__":
    metrics, details = run()
    report = write_report(metrics, details)
    print(report)
