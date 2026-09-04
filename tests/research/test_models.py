"""Unit tests for app.research.models."""

import pytest
from pydantic import ValidationError

from app.research.models import (
    Claim,
    ClaimStatus,
    EventType,
    Evidence,
    ResearchTask,
    TaskStatus,
    TraceEvent,
)


class FakeSearchResult:
    """Duck-typed stand-in for WebSearch.SearchResult."""

    def __init__(self, url, title, description, raw_content=None, source="google"):
        self.url = url
        self.title = title
        self.description = description
        self.raw_content = raw_content
        self.source = source


class TestEvidence:
    def test_from_search_result_maps_fields(self):
        result = FakeSearchResult(
            "https://example.com/a", "Example", "A snippet", source="duckduckgo"
        )
        evidence = Evidence.from_search_result(result, query="test query")
        assert evidence.url == "https://example.com/a"
        assert evidence.title == "Example"
        assert evidence.snippet == "A snippet"
        assert evidence.source == "duckduckgo"
        assert evidence.retrieval_query == "test query"
        assert evidence.full_text is None

    def test_content_hash_is_deterministic_and_content_based(self):
        a = Evidence(snippet="same text")
        b = Evidence(snippet="same text")
        c = Evidence(snippet="other text")
        assert a.content_hash == b.content_hash
        assert a.content_hash != c.content_hash
        assert len(a.content_hash) == 64

    def test_full_text_takes_precedence_for_hash(self):
        a = Evidence(snippet="s", full_text="full")
        b = Evidence(snippet="s", full_text="full")
        c = Evidence(snippet="s")
        assert a.content_hash == b.content_hash
        assert a.content_hash != c.content_hash

    def test_explicit_hash_is_preserved(self):
        evidence = Evidence(snippet="text", content_hash="custom-hash")
        assert evidence.content_hash == "custom-hash"


class TestClaim:
    def test_defaults(self):
        claim = Claim(text="The sky is blue")
        assert claim.status == ClaimStatus.UNSUPPORTED
        assert claim.supporting_evidence_ids == []
        assert claim.contradicting_evidence_ids == []
        assert claim.confidence is None

    def test_confidence_is_bounded(self):
        with pytest.raises(ValidationError):
            Claim(text="x", confidence=1.5)
        with pytest.raises(ValidationError):
            Claim(text="x", confidence=-0.1)


class TestResearchTask:
    def test_defaults(self):
        task = ResearchTask(question="What is X?")
        assert task.task_id.startswith("task-")
        assert task.status == TaskStatus.PENDING
        assert task.budget.max_steps == 12
        assert task.constraints == []
        assert task.created_at.tzinfo is not None

    def test_task_ids_are_unique(self):
        t1 = ResearchTask(question="q1")
        t2 = ResearchTask(question="q2")
        assert t1.task_id != t2.task_id


class TestTraceEvent:
    def test_json_serialization(self):
        event = TraceEvent(task_id="t1", event_type=EventType.RETRIEVAL, span_id="s1")
        dumped = event.model_dump(mode="json")
        assert dumped["event_type"] == "retrieval"
        assert dumped["task_id"] == "t1"
        assert dumped["span_id"] == "s1"
        assert isinstance(dumped["timestamp"], str)
