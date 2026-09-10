"""Retrieval evaluation: Recall@k, MRR, dedup rate and latency.

Usage: python -m app.research.evaluator

Loads gold labels from data/research/retrieval_gold.jsonl and compares
retrieval configurations over the local corpus (BM25 / vector / hybrid /
hybrid + LLM rerank). Doc-level relevance labels are used for scoring.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.research.reranker import LLMReranker
from app.research.retriever import LocalRetriever

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "research"
GOLD_PATH = DATA_ROOT / "retrieval_gold.jsonl"


def load_gold(path: Optional[Path] = None) -> List[dict]:
    path = Path(path) if path else GOLD_PATH
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            queries.append(json.loads(line))
    return queries


async def evaluate(
    retriever: LocalRetriever,
    queries: List[dict],
    top_k: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run the eval set through one retriever; returns aggregate metrics."""
    recalls: List[float] = []
    mrrs: List[float] = []
    doc_rates: List[float] = []
    latencies: List[float] = []
    per_query: List[dict] = []

    for item in queries:
        started = time.perf_counter()
        evidence = await retriever.search(item["query"], top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)

        retrieved_docs = []
        for ev in evidence:
            if ev.doc_id and ev.doc_id not in retrieved_docs:
                retrieved_docs.append(ev.doc_id)

        relevant = set(item["relevant_doc_ids"])
        hits = [doc for doc in retrieved_docs if doc in relevant]
        recall = len(hits) / len(relevant) if relevant else 1.0
        mrr = 0.0
        for rank, doc in enumerate(retrieved_docs, 1):
            if doc in relevant:
                mrr = 1.0 / rank
                break
        doc_rate = len(retrieved_docs) / top_k if top_k else 0.0

        recalls.append(recall)
        mrrs.append(mrr)
        doc_rates.append(doc_rate)
        per_query.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "recall": round(recall, 3),
                "mrr": round(mrr, 3),
                "retrieved_docs": retrieved_docs[:top_k],
                "latency_ms": round(latencies[-1], 1),
            }
        )
        if verbose:
            print(
                f"{item['query_id']} recall={recall:.2f} mrr={mrr:.2f} "
                f"docs={retrieved_docs[:top_k]}"
            )

    def mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "recall_at_k": round(mean(recalls), 3),
        "mrr": round(mean(mrrs), 3),
        "unique_doc_rate": round(mean(doc_rates), 3),
        "avg_latency_ms": round(mean(latencies), 1),
        "queries": per_query,
    }


async def run_ablation(
    queries: List[dict], top_k: int = 5, verbose: bool = False
) -> List[Dict[str, Any]]:
    """Compare retrieval configurations and print a summary table."""
    configs: Dict[str, LocalRetriever] = {
        "bm25": LocalRetriever(mode="bm25"),
        "vector": LocalRetriever(mode="vector"),
        "hybrid": LocalRetriever(mode="hybrid"),
        "hybrid+rerank": LocalRetriever(mode="hybrid", reranker=LLMReranker()),
    }
    # Build the corpus index once (vector cache is shared on disk).
    first = next(iter(configs.values()))
    chunk_count = await first.index_corpus()
    print(f"Corpus indexed: {chunk_count} chunks\n")

    results = []
    print(
        f"{'config':<16}{'Recall@k':>10}{'MRR':>8}{'uniq/top_k':>12}{'latency_ms':>12}"
    )
    for name, retriever in configs.items():
        metrics = await evaluate(retriever, queries, top_k=top_k, verbose=verbose)
        results.append({"config": name, **{k: v for k, v in metrics.items() if k != "queries"}})
        print(
            f"{name:<16}{metrics['recall_at_k']:>10.3f}{metrics['mrr']:>8.3f}"
            f"{metrics['unique_doc_rate']:>12.3f}{metrics['avg_latency_ms']:>12.1f}"
        )
    return results


async def main() -> None:
    queries = load_gold()
    results = await run_ablation(queries)
    out_path = DATA_ROOT / "retrieval_eval_results.json"
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
