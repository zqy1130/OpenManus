"""Unit tests for app.research.evaluator (retriever mocked)."""

import pytest

from app.research.evaluator import evaluate
from app.research.models import Evidence


class FakeRetriever:
    """Returns evidence with preset doc ids in order."""

    def __init__(self, doc_ids_by_query=None):
        self.doc_ids_by_query = doc_ids_by_query or {}

    async def search(self, query, top_k=5, filters=None):
        doc_ids = self.doc_ids_by_query.get(query, [])
        return [
            Evidence(
                url="",
                snippet=f"chunk of {doc_id}",
                source="local",
                doc_id=doc_id,
                chunk_id=f"{doc_id}#0",
            )
            for doc_id in doc_ids
        ]


QUERIES = [
    {"query_id": "q1", "query": "a", "relevant_doc_ids": ["d1", "d2"]},
    {"query_id": "q2", "query": "b", "relevant_doc_ids": ["d9"]},
]


@pytest.mark.asyncio
async def test_recall_and_mrr():
    retriever = FakeRetriever(
        {"a": ["d1", "d3", "d4"], "b": ["d7", "d8", "d9"]}
    )
    metrics = await evaluate(retriever, QUERIES, top_k=3)
    # q1: recall = 1/2, mrr = 1/1
    # q2: recall = 1/1, mrr = 1/3
    assert metrics["recall_at_k"] == pytest.approx((0.5 + 1.0) / 2, abs=1e-3)
    assert metrics["mrr"] == pytest.approx((1.0 + 1 / 3) / 2, abs=1e-3)
    assert metrics["unique_doc_rate"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["avg_latency_ms"] >= 0


@pytest.mark.asyncio
async def test_missed_query():
    retriever = FakeRetriever({"a": ["d3"]})
    metrics = await evaluate(retriever, QUERIES, top_k=3)
    assert metrics["recall_at_k"] == pytest.approx(0.0)  # q1 misses, q2 returns []
    assert metrics["mrr"] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_duplicate_docs_counted_once():
    retriever = FakeRetriever({"a": ["d1", "d1", "d2"]})
    metrics = await evaluate(retriever, [QUERIES[0]], top_k=3)
    assert metrics["recall_at_k"] == pytest.approx(1.0)
    assert metrics["unique_doc_rate"] == pytest.approx(2 / 3, abs=1e-3)
