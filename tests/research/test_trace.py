"""Unit tests for app.research.trace.TraceWriter."""

import json
from pathlib import Path

from app.research.models import EventType, LLMUsage
from app.research.trace import TraceWriter


def test_writes_jsonl_and_roundtrips(tmp_path: Path):
    writer = TraceWriter("task-abc", output_root=tmp_path)
    writer.add_event(EventType.SYSTEM, payload={"msg": "hello"})
    writer.close()

    trace_path = tmp_path / "task-abc" / "trace.jsonl"
    assert trace_path.exists()

    loaded = TraceWriter.load_trace(trace_path)
    payloads = [e.payload for e in loaded if e.event_type == EventType.SYSTEM]
    assert payloads == [{"msg": "hello"}]
    assert all(e.task_id == "task-abc" for e in loaded)


def test_span_hierarchy_and_replay(tmp_path: Path):
    writer = TraceWriter("task-1", output_root=tmp_path)
    span = writer.start_span("retrieval")
    event = writer.add_event(EventType.RETRIEVAL, payload={"query": "q"})
    assert event.span_id == span
    writer.end_span(span)
    after = writer.add_event(EventType.SYSTEM, payload={})
    assert after.span_id != span
    writer.close()

    events = TraceWriter.load_trace(tmp_path / "task-1" / "trace.jsonl")
    tree = TraceWriter.build_span_tree(events)

    assert tree[span]["name"] == "retrieval"
    assert tree[span]["parent_span_id"] is not None
    assert tree[span]["ended_at"] is not None
    assert [e.event_type for e in tree[span]["events"]] == [EventType.RETRIEVAL]
    # The root span contains the event recorded after end_span.
    root_id = tree[span]["parent_span_id"]
    assert any(
        e.event_type == EventType.SYSTEM for e in tree[root_id]["events"]
    )


def test_nested_spans_default_to_current_parent(tmp_path: Path):
    writer = TraceWriter("t", output_root=tmp_path)
    phase = writer.start_span("planning")
    step = writer.start_span("step-1")
    writer.end_span(step)
    writer.end_span(phase)
    writer.close()

    tree = TraceWriter.build_span_tree(
        TraceWriter.load_trace(tmp_path / "t" / "trace.jsonl")
    )
    assert tree[step]["parent_span_id"] == phase
    assert step in tree[phase]["children"]


def test_redacts_sensitive_keys(tmp_path: Path):
    writer = TraceWriter("t", output_root=tmp_path)
    writer.add_event(
        EventType.LLM,
        payload={
            "model": "gpt-x",
            "api_key": "sk-123456",
            "nested": {"authorization": "Bearer abc", "query": "ok"},
        },
        usage=LLMUsage(model="gpt-x", input_tokens=10, output_tokens=5),
    )
    writer.close()

    raw = (tmp_path / "t" / "trace.jsonl").read_text(encoding="utf-8")
    assert "sk-123456" not in raw
    assert "Bearer abc" not in raw
    assert "[REDACTED]" in raw

    last = json.loads(raw.strip().splitlines()[-1])
    assert last["payload"]["nested"]["query"] == "ok"
    assert last["usage"]["input_tokens"] == 10


def test_manifest_summary(tmp_path: Path):
    writer = TraceWriter("t", output_root=tmp_path)
    writer.add_event(
        EventType.LLM,
        payload={},
        usage=LLMUsage(
            model="m", input_tokens=100, output_tokens=50, estimated_cost_usd=0.01
        ),
    )
    writer.add_event(EventType.ERROR, payload={"reason": "timeout"})
    path = writer.write_manifest("completed")
    writer.close()

    assert path == tmp_path / "t" / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["task_id"] == "t"
    assert summary["status"] == "completed"
    assert summary["tokens"] == {"input": 100, "output": 50}
    assert summary["error_count"] == 1
    assert summary["duration_s"] >= 0
