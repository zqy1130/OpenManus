"""Unit tests for app.research.success_judge and model routing."""

import json

import pytest

from app.research.llm import resolve_model
from app.research.success_judge import SuccessJudge, TaskVerdict


class TestParseVerdict:
    def test_full_verdict(self):
        content = json.dumps(
            {
                "key_facts": [
                    {"fact": "a", "covered": True, "correct": True},
                    {"fact": "b", "covered": False, "correct": False},
                ],
                "task_success": False,
                "score": 0.5,
                "reason": "one fact missing",
            }
        )
        verdict = SuccessJudge._parse_verdict(content, ["a", "b"])
        assert verdict.task_success is False
        assert verdict.score == 0.5
        assert len(verdict.key_facts) == 2
        assert verdict.key_facts[0].covered is True

    def test_malformed_fails_closed(self):
        verdict = SuccessJudge._parse_verdict("not json", ["a"])
        assert verdict.task_success is False
        assert verdict.reason == "judge parse failed"

    def test_score_clamped(self):
        content = json.dumps({"task_success": True, "score": 99})
        verdict = SuccessJudge._parse_verdict(content, [])
        assert verdict.score == 1.0


class TestRouting:
    def test_off_returns_none(self):
        assert resolve_model("planning", False) is None
        assert resolve_model("unknown_stage", True) is None

    def test_route_map(self):
        assert resolve_model("planning", True) == "qwen3.7-max"
        assert resolve_model("validation", True) == "qwen-plus"
        assert resolve_model("synthesis", True) == "qwen3.7-max"
        assert resolve_model("success_judge", True) == "qwen-plus"


class TestAggregate:
    def test_aggregate_metrics(self):
        from app.research.run_eval import aggregate

        rows = [
            {
                "task_id": "t1",
                "status": "completed",
                "citation_correctness": 1.0,
                "unsupported_claim_rate": 0.0,
                "task_success": True,
                "gold_score": 1.0,
                "llm_latency_ms": [100.0, 200.0, 300.0],
                "error_type": "unknown",
            },
            {
                "task_id": "t2",
                "status": "completed",
                "citation_correctness": 0.8,
                "unsupported_claim_rate": 0.1,
                "task_success": False,
                "gold_score": 0.5,
                "llm_latency_ms": [50.0],
                "error_type": "unknown",
            },
            {
                "task_id": "t3",
                "status": "failed: APIError",
                "error_type": "retrieval_error",
            },
        ]
        summary = aggregate(rows)
        assert summary["task_count"] == 3
        assert summary["completed"] == 2
        assert summary["task_success_rate"] == 0.5
        assert summary["avg_gold_score"] == 0.75
        assert summary["avg_citation_correctness"] == pytest.approx(0.9)
        assert summary["avg_unsupported_claim_rate"] == pytest.approx(0.05)
        assert summary["llm_latency_p50_ms"] == 200.0  # nearest-rank: values[int(0.5*4)]
        assert summary["llm_latency_p95_ms"] == 300.0
        assert summary["failure_types"] == {"retrieval_error": 1}

    def test_classify_failure(self):
        from app.research.models import EventType, TraceEvent
        from app.research.run_eval import classify_failure

        events = [
            TraceEvent(
                task_id="t", event_type=EventType.ERROR,
                payload={"stage": "planning", "error": "x"},
            )
        ]
        assert classify_failure(events) == "planning_error"

        events = [
            TraceEvent(
                task_id="t", event_type=EventType.ERROR,
                payload={"query": "q", "error": "boom", "error_type": "rate_limit"},
            )
        ]
        assert classify_failure(events) == "rate_limit"
