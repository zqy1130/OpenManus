"""Unit tests for app.research.claim_check (LLM mocked, fully offline)."""

import json
from pathlib import Path

import pytest

import app.research.claim_check as cc_module
from app.research.claim_check import ReportAuditor

USAGE = __import__("app.research.models", fromlist=["LLMUsage"]).LLMUsage(
    model="test", input_tokens=10, output_tokens=5
)


class FakeLLM:
    """Returns responses in order; extraction first, then check batches."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def __call__(self, messages, max_tokens=4096, temperature=0.0):
        self.calls.append(messages)
        return self.responses.pop(0)


REPORT = """# Question
Who won?

# Key Findings
- The winners were Alice and Bob [E1][E2].
- The award ceremony was held in Stockholm. [E1]
- Some claim with no citation at all.

# References
- [E1] A (https://a.com)
- [E2] B (https://b.com)
"""

EVIDENCE = [
    {"title": "A", "url": "https://a.com", "snippet": "Alice and Bob won the prize."},
    {"title": "B", "url": "https://b.com", "snippet": "Bob is a scientist."},
]

EXTRACT_RESPONSE = json.dumps(
    [
        {"text": "The winners were Alice and Bob", "evidence_ids": ["1", "2"]},
        {"text": "The award ceremony was held in Stockholm", "evidence_ids": ["1"]},
        {"text": "Some claim with no citation at all", "evidence_ids": []},
    ]
)

CHECK_RESPONSE = json.dumps(
    [
        {"claim_index": 0, "verdict": "supported", "reason": "evidence states it"},
        {"claim_index": 1, "verdict": "irrelevant", "reason": "no location info"},
        {"claim_index": 2, "verdict": "unsupported", "reason": "no citations"},
    ]
)


def make_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task-test"
    task_dir.mkdir()
    (task_dir / "report.md").write_text(REPORT, encoding="utf-8")
    (task_dir / "evidence.json").write_text(
        json.dumps(EVIDENCE, ensure_ascii=False), encoding="utf-8"
    )
    return task_dir


@pytest.fixture
def task_dir(tmp_path: Path) -> Path:
    return make_task_dir(tmp_path)


class TestAuditTask:
    @pytest.mark.asyncio
    async def test_metrics_and_audit_file(self, task_dir: Path, monkeypatch):
        monkeypatch.setattr(
            cc_module, "call_llm", FakeLLM([(EXTRACT_RESPONSE, USAGE), (CHECK_RESPONSE, USAGE)])
        )
        metrics = await ReportAuditor().audit_task(task_dir)

        assert metrics["claim_count"] == 3
        assert metrics["claims_with_citations"] == 2
        assert metrics["supported"] == 1
        assert metrics["irrelevant"] == 1
        assert metrics["unsupported"] == 1
        assert metrics["citation_correctness"] == pytest.approx(0.5, abs=1e-3)
        assert metrics["unsupported_claim_rate"] == pytest.approx(1 / 3, abs=1e-3)

        audit = json.loads((task_dir / "audit.json").read_text(encoding="utf-8"))
        assert len(audit["claims"]) == 3
        assert [c["verdict"] for c in audit["checks"]] == [
            "supported",
            "irrelevant",
            "unsupported",
        ]

    @pytest.mark.asyncio
    async def test_extraction_failure_yields_empty(self, task_dir: Path, monkeypatch):
        monkeypatch.setattr(
            cc_module,
            "call_llm",
            FakeLLM([("no json at all", USAGE), ("still no json", USAGE)]),
        )
        metrics = await ReportAuditor().audit_task(task_dir)
        assert metrics["claim_count"] == 0
        assert metrics["citation_correctness"] is None

    def test_parse_claims_accepts_dict_form(self):
        content = json.dumps(
            {"claims": [{"text": "a fact", "evidence_ids": ["1"]}]}
        )
        claims = ReportAuditor._parse_claims(content)
        assert len(claims) == 1
        assert claims[0]["evidence_ids"] == ["1"]

    @pytest.mark.asyncio
    async def test_unjudged_excluded_from_correctness(self, task_dir: Path, monkeypatch):
        """Claims with citations but no parsed verdict must not count as wrong."""
        monkeypatch.setattr(
            cc_module,
            "call_llm",
            FakeLLM(
                [
                    (EXTRACT_RESPONSE, USAGE),
                    ("not json", USAGE),
                    ("still not json", USAGE),
                ]
            ),
        )
        metrics = await ReportAuditor().audit_task(task_dir)
        assert metrics["unsupported"] == 1  # only the no-citation claim
        assert metrics["citation_correctness"] is None  # nothing judged
        assert metrics["claim_count"] == 3

    @pytest.mark.asyncio
    async def test_invalid_citation_label_is_passed_through(self, task_dir: Path, monkeypatch):
        extract = json.dumps(
            [{"text": "claims something", "evidence_ids": ["99"]}]
        )
        check = json.dumps(
            [{"claim_index": 0, "verdict": "irrelevant", "reason": "bad citation"}]
        )
        captured = {}

        async def spy(messages, max_tokens=4096, temperature=0.0):
            if "claim extractor" in messages[0]["content"]:
                return extract, USAGE
            captured["user"] = messages[1]["content"]
            return check, USAGE

        monkeypatch.setattr(cc_module, "call_llm", spy)
        metrics = await ReportAuditor().audit_task(task_dir)
        assert "(invalid citation)" in captured["user"]
        assert metrics["irrelevant"] == 1


class TestListTaskDirs:
    def test_only_dirs_with_reports(self, tmp_path: Path):
        make_task_dir(tmp_path)
        (tmp_path / "empty-dir").mkdir()
        (tmp_path / "no-report").mkdir()
        (tmp_path / "no-report" / "other.txt").write_text("x", encoding="utf-8")
        dirs = cc_module.list_task_dirs(tmp_path)
        assert [d.name for d in dirs] == ["task-test"]
