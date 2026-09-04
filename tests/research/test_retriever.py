"""Unit tests for app.research.retriever."""

from pathlib import Path

import pytest

from app.research.models import EventType, Evidence
from app.research.retriever import Retriever, dedup_evidence
from app.research.trace import TraceWriter


class FakeSearchResult:
    def __init__(self, url, title="", description="", raw_content=None, source="fake"):
        self.url = url
        self.title = title
        self.description = description
        self.raw_content = raw_content
        self.source = source


class FakeWebSearch:
    """Stand-in for WebSearch with configurable results/error/exception."""

    def __init__(self, results=None, error=None, exc=None):
        self.results = results or []
        self.error = error
        self.exc = exc
        self.calls = []

    async def execute(self, query, num_results=5, fetch_content=False):
        self.calls.append({"query": query, "num_results": num_results, "fetch_content": fetch_content})
        if self.exc:
            raise self.exc
        return type(
            "Response",
            (),
            {"results": self.results, "error": self.error},
        )()


def make_evidence(url, snippet="text"):
    return Evidence(url=url, snippet=snippet)


class TestDedup:
    def test_dedups_by_url(self):
        evs = [make_evidence("https://a.com"), make_evidence("https://a.com")]
        assert len(dedup_evidence(evs)) == 1

    def test_dedups_by_content_hash(self):
        a = Evidence(snippet="same content")
        b = Evidence(snippet="same content")
        assert dedup_evidence([a, b]) == [a]

    def test_keeps_distinct(self):
        evs = [make_evidence("https://a.com", "x"), make_evidence("https://b.com", "y")]
        assert len(dedup_evidence(evs)) == 2


class TestRetriever:
    @pytest.mark.asyncio
    async def test_converts_and_dedups(self, tmp_path: Path):
        fake = FakeWebSearch(
            results=[
                FakeSearchResult("https://a.com", "A", "snippet a"),
                FakeSearchResult("https://a.com", "A", "snippet a"),
                FakeSearchResult("https://b.com", "B", "snippet b"),
            ]
        )
        with TraceWriter("t", output_root=tmp_path) as trace:
            retriever = Retriever(trace=trace, top_k=5, web_search=fake)
            evidence = await retriever.search("test query", fetch_content=False)

        assert len(evidence) == 2
        assert evidence[0].retrieval_query == "test query"
        assert fake.calls[0]["num_results"] == 5

        events = TraceWriter.load_trace(tmp_path / "t" / "trace.jsonl")
        retrieval_events = [e for e in events if e.event_type == EventType.RETRIEVAL]
        assert any(e.payload.get("evidence_count") == 2 for e in retrieval_events)

    @pytest.mark.asyncio
    async def test_error_response_returns_empty(self, tmp_path: Path):
        fake = FakeWebSearch(error="All engines failed")
        with TraceWriter("t", output_root=tmp_path) as trace:
            retriever = Retriever(trace=trace, web_search=fake)
            evidence = await retriever.search("q")

        assert evidence == []
        events = TraceWriter.load_trace(tmp_path / "t" / "trace.jsonl")
        assert any(
            e.event_type == EventType.ERROR and "All engines failed" in str(e.payload)
            for e in events
        )

    @pytest.mark.asyncio
    async def test_exception_returns_empty(self, tmp_path: Path):
        fake = FakeWebSearch(exc=RuntimeError("network down"))
        with TraceWriter("t", output_root=tmp_path) as trace:
            retriever = Retriever(trace=trace, web_search=fake)
            evidence = await retriever.search("q")

        assert evidence == []
        events = TraceWriter.load_trace(tmp_path / "t" / "trace.jsonl")
        assert any(
            e.event_type == EventType.ERROR and "network down" in str(e.payload)
            for e in events
        )

    @pytest.mark.asyncio
    async def test_works_without_trace(self):
        fake = FakeWebSearch(results=[FakeSearchResult("https://a.com", "A", "s")])
        retriever = Retriever(trace=None, web_search=fake)
        evidence = await retriever.search("q")
        assert len(evidence) == 1
