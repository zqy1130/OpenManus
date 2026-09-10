"""Unit tests for app.research.orchestrator (LLM and retriever mocked)."""

import json
from pathlib import Path

import pytest

import app.research.orchestrator as orchestrator_module
from app.research.models import EventType, Evidence, LLMUsage, ResearchTask, TaskStatus
from app.research.orchestrator import ResearchOrchestrator
from app.research.trace import TraceWriter

PLAN_JSON = json.dumps(
    [
        {"goal": "who won", "search_query": "nobel physics 2024 winner"},
        {"goal": "main contribution", "search_query": "nobel physics 2024 contribution"},
    ]
)

REPORT = """# Question

Who won?

# Key Findings

The winners were X and Y [E1][E2].

# References

- [E1] A (https://a.com)
- [E2] B (https://b.com)
"""

USAGE = LLMUsage(model="test-model", input_tokens=100, output_tokens=50)


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, messages, max_tokens=4096, temperature=0.0):
        self.calls.append(messages)
        return self.responses.pop(0)


class FakeRetriever:
    def __init__(self, evidence=None):
        self.evidence = evidence or []
        self.queries = []

    async def search(self, query, top_k=None, fetch_content=True):
        self.queries.append(query)
        return self.evidence


def make_evidence(url, snippet=None):
    return Evidence(url=url, snippet=snippet if snippet is not None else f"content of {url}")


@pytest.fixture
def task():
    return ResearchTask(question="Who won the 2024 Nobel Prize in Physics?")


class TestParsePlan:
    def test_plain_json_array(self):
        plan = ResearchOrchestrator._parse_plan(PLAN_JSON)
        assert len(plan) == 2
        assert plan[0]["search_query"] == "nobel physics 2024 winner"

    def test_json_in_markdown_fence(self):
        plan = ResearchOrchestrator._parse_plan(f"```json\n{PLAN_JSON}\n```")
        assert len(plan) == 2

    def test_json_wrapped_in_prose(self):
        plan = ResearchOrchestrator._parse_plan(f"Here is the plan: {PLAN_JSON} thanks")
        assert len(plan) == 2

    def test_dict_form(self):
        plan = ResearchOrchestrator._parse_plan(json.dumps({"sub_questions": json.loads(PLAN_JSON)}))
        assert len(plan) == 2

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            ResearchOrchestrator._parse_plan("no json here")

    def test_empty_array_raises(self):
        with pytest.raises(ValueError):
            ResearchOrchestrator._parse_plan("[]")


class TestCheckCitations:
    def test_valid_and_invalid(self):
        stats = ResearchOrchestrator.check_citations("[E1] and [E2] and [E9]", 3)
        assert stats["cited_evidence"] == [1, 2]
        assert stats["invalid_citations"] == [9]
        assert stats["citation_coverage"] == pytest.approx(2 / 3, abs=1e-3)

    def test_empty_report(self):
        stats = ResearchOrchestrator.check_citations("no citations", 3)
        assert stats["cited_evidence"] == []
        assert stats["citation_coverage"] == 0.0


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_end_to_end(self, tmp_path: Path, task, monkeypatch):
        monkeypatch.setattr(
            orchestrator_module,
            "call_llm",
            FakeLLM([(PLAN_JSON, USAGE), (REPORT, USAGE)]),
        )
        retriever = FakeRetriever(
            evidence=[make_evidence("https://a.com"), make_evidence("https://b.com")]
        )
        with TraceWriter(task.task_id, output_root=tmp_path) as trace:
            orchestrator = ResearchOrchestrator(task=task, trace=trace, retriever=retriever)
            summary = await orchestrator.run()

        assert task.status == TaskStatus.COMPLETED
        assert summary["sub_question_count"] == 2
        assert summary["evidence_count"] == 2
        assert summary["citation_coverage"] == pytest.approx(1.0)
        assert summary["invalid_citations"] == []

        task_dir = tmp_path / task.task_id
        report = (task_dir / "report.md").read_text(encoding="utf-8")
        assert report == REPORT
        evidence = json.loads((task_dir / "evidence.json").read_text(encoding="utf-8"))
        assert len(evidence) == 2

        manifest = json.loads((task_dir / "summary.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "completed"
        assert manifest["tokens"] == {"input": 200, "output": 100}

    @pytest.mark.asyncio
    async def test_plan_parse_failure_falls_back(self, tmp_path: Path, task, monkeypatch):
        monkeypatch.setattr(
            orchestrator_module,
            "call_llm",
            FakeLLM([("not json at all", USAGE), (REPORT, USAGE)]),
        )
        retriever = FakeRetriever(evidence=[])
        with TraceWriter(task.task_id, output_root=tmp_path) as trace:
            orchestrator = ResearchOrchestrator(task=task, trace=trace, retriever=retriever)
            summary = await orchestrator.run()

        assert task.status == TaskStatus.COMPLETED
        assert summary["sub_question_count"] == 1
        assert retriever.queries == [task.question]

    @pytest.mark.asyncio
    async def test_validator_retry_loop(self, tmp_path: Path, task, monkeypatch):
        """Incomplete step triggers one refined-query retry, then completes."""
        from app.research.verifier import ValidationResult

        class FakeValidator:
            def __init__(self):
                self.seen = set()

            async def validate_step(self, goal, evidence, question):
                first_time = goal not in self.seen
                self.seen.add(goal)
                if first_time:
                    return ValidationResult(
                        step_complete=False,
                        evidence_sufficiency=3.0,
                        refined_query="refined query",
                        reason="missing facts",
                    )
                return ValidationResult(
                    step_complete=True, evidence_sufficiency=9.0, reason="covered"
                )

        monkeypatch.setattr(
            orchestrator_module,
            "call_llm",
            FakeLLM([(PLAN_JSON, USAGE), (REPORT, USAGE)]),
        )
        retriever = FakeRetriever(evidence=[make_evidence("https://a.com")])
        with TraceWriter(task.task_id, output_root=tmp_path) as trace:
            orchestrator = ResearchOrchestrator(
                task=task,
                trace=trace,
                retriever=retriever,
                validator=FakeValidator(),
                max_retries_per_step=2,
            )
            await orchestrator.run()

        # each of the 2 steps: first attempt judged incomplete -> retried
        assert retriever.queries == [
            "nobel physics 2024 winner",
            "refined query",
            "nobel physics 2024 contribution",
            "refined query",
        ]
        events = TraceWriter.load_trace(tmp_path / task.task_id / "trace.jsonl")
        verify_events = [e for e in events if e.event_type == EventType.VERIFICATION]
        metrics = [e for e in verify_events if "step_index" in e.payload]
        assert len(metrics) == 2
        assert metrics[0].payload["retries_used"] == 1
        assert metrics[0].payload["step_complete"] is None  # last attempt not re-validated
        assert metrics[0].payload["evidence_sufficiency"] == 3.0

    @pytest.mark.asyncio
    async def test_validator_retry_exhausted(self, tmp_path: Path, task, monkeypatch):
        """Validator keeps asking for more; loop stops at max_retries."""
        from app.research.verifier import ValidationResult

        class NeverDoneValidator:
            async def validate_step(self, goal, evidence, question):
                return ValidationResult(
                    step_complete=False,
                    evidence_sufficiency=2.0,
                    refined_query="another query",
                    reason="still missing",
                )

        monkeypatch.setattr(
            orchestrator_module,
            "call_llm",
            FakeLLM([(PLAN_JSON, USAGE), (REPORT, USAGE)]),
        )
        retriever = FakeRetriever(evidence=[])
        with TraceWriter(task.task_id, output_root=tmp_path) as trace:
            orchestrator = ResearchOrchestrator(
                task=task,
                trace=trace,
                retriever=retriever,
                validator=NeverDoneValidator(),
                max_retries_per_step=2,
            )
            await orchestrator.run()

        assert retriever.queries == [
            "nobel physics 2024 winner",
            "another query",
            "nobel physics 2024 contribution",
            "another query",
        ]
        events = TraceWriter.load_trace(tmp_path / task.task_id / "trace.jsonl")
        metrics = [
            e
            for e in events
            if e.event_type == EventType.VERIFICATION and "step_index" in e.payload
        ]
        assert metrics[0].payload["retries_used"] == 1
        assert metrics[0].payload["step_complete"] is None  # never completed

    @pytest.mark.asyncio
    async def test_synthesis_failure_marks_failed(self, tmp_path: Path, task, monkeypatch):
        class FlakyLLM(FakeLLM):
            def __init__(self):
                super().__init__([(PLAN_JSON, USAGE)])

            async def __call__(self, messages, max_tokens=4096, temperature=0.0):
                if len(self.calls) >= 1:
                    raise RuntimeError("llm down")
                return await super().__call__(messages, max_tokens, temperature)

        monkeypatch.setattr(orchestrator_module, "call_llm", FlakyLLM())
        with TraceWriter(task.task_id, output_root=tmp_path) as trace:
            orchestrator = ResearchOrchestrator(
                task=task, trace=trace, retriever=FakeRetriever()
            )
            with pytest.raises(RuntimeError):
                await orchestrator.run()

        assert task.status == TaskStatus.FAILED
        manifest = json.loads(
            (tmp_path / task.task_id / "summary.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "failed"
        assert "llm down" in manifest["error"]
