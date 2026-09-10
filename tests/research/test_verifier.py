"""Unit tests for app.research.verifier (LLM mocked)."""

import json
from pathlib import Path

import pytest

import app.research.verifier as verifier_module
from app.research.models import EventType, Evidence
from app.research.trace import TraceWriter
from app.research.verifier import Validator, ValidationResult

USAGE = __import__("app.research.models", fromlist=["LLMUsage"]).LLMUsage(
    model="test", input_tokens=10, output_tokens=5
)


class FakeLLM:
    def __init__(self, response):
        self.response = response

    async def __call__(self, messages, max_tokens=4096, temperature=0.0):
        return self.response, USAGE


class TestParseResult:
    def test_valid_json(self):
        result = Validator._parse_result(
            json.dumps(
                {
                    "step_complete": False,
                    "evidence_sufficiency": 4,
                    "refined_query": "better query",
                    "reason": "missing facts",
                }
            )
        )
        assert result.step_complete is False
        assert result.evidence_sufficiency == 4.0
        assert result.refined_query == "better query"

    def test_malformed_fails_open(self):
        result = Validator._parse_result("not json")
        assert result.step_complete is True
        assert result.reason == "validator parse failed"

    def test_non_dict_fails_open(self):
        result = Validator._parse_result("[1, 2, 3]")
        assert result.step_complete is True

    def test_missing_fields_defaulted(self):
        result = Validator._parse_result(json.dumps({"step_complete": True}))
        assert result.evidence_sufficiency == 0.0
        assert result.refined_query == ""

    def test_sufficiency_out_of_range_fails_open(self):
        result = Validator._parse_result(
            json.dumps({"step_complete": False, "evidence_sufficiency": 99})
        )
        assert result.step_complete is True  # ValidationError -> fail open


class TestValidateStep:
    @pytest.mark.asyncio
    async def test_records_metrics_in_trace(self, tmp_path: Path, monkeypatch):
        response = json.dumps(
            {
                "step_complete": True,
                "evidence_sufficiency": 8,
                "refined_query": "",
                "reason": "covered",
            }
        )
        monkeypatch.setattr(verifier_module, "call_llm", FakeLLM(response))
        with TraceWriter("t", output_root=tmp_path) as trace:
            validator = Validator(trace=trace)
            result = await validator.validate_step(
                "who won", [Evidence(url="", snippet="text")], "question?"
            )
        assert result.step_complete is True
        assert result.evidence_sufficiency == 8.0

        events = TraceWriter.load_trace(tmp_path / "t" / "trace.jsonl")
        verify_events = [e for e in events if e.event_type == EventType.VERIFICATION]
        assert len(verify_events) == 1
        assert verify_events[0].payload["step_complete"] is True
        assert verify_events[0].payload["evidence_sufficiency"] == 8.0
