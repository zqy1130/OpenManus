"""Minimal research orchestrator: plan -> search -> synthesize -> report."""

import json
import re
import time
from typing import Any, Dict, List, Optional

from app.research.llm import call_llm, extract_json
from app.research.models import (
    EventType,
    Evidence,
    ResearchTask,
    TaskStatus,
)
from app.research.retriever import Retriever, dedup_evidence
from app.research.trace import TraceWriter

PLAN_SYSTEM_PROMPT = """You are a research planner. Your job is to break down a research question into focused sub-questions that can be answered through web search.

Rules:
- Produce 3 to 5 sub-questions that together cover the original question.
- Each sub-question must be self-contained and answerable with 1-2 searches.
- Provide an effective web search query for each sub-question.

Respond with ONLY a JSON array of objects in this format:
[{"goal": "what this sub-question aims to find out", "search_query": "the web search query to run"}]"""

SYNTHESIS_SYSTEM_PROMPT = """You are a research analyst. Write a research report based ONLY on the evidence provided. Follow these rules:

1. Write the report in the same language as the research question.
2. Structure the report with these Markdown sections: # Question, # Method, # Key Findings, # Conflicting Views, # Limitations, # References.
3. Cite evidence inline as [E1], [E2] etc. Every key factual claim must carry at least one citation.
4. Never invent facts that are not in the evidence. If evidence is insufficient, explicitly write "insufficient evidence" instead of guessing.
5. If sources conflict, present both sides and mark the conflict explicitly.
6. In References, list every cited evidence id with its title and URL.

The research question and the evidence list follow in the user message."""


class ResearchOrchestrator:
    """Runs the minimal research pipeline for one task.

    plan (LLM) -> retrieve per sub-question -> synthesize (LLM) -> report.md
    Everything is traced via the TraceWriter: spans for planning, per-step
    retrieval and synthesis, with per-call LLM usage recorded.
    """

    def __init__(
        self,
        task: ResearchTask,
        trace: TraceWriter,
        retriever: Optional[Retriever] = None,
        top_k: int = 5,
        max_sub_questions: int = 5,
    ):
        self.task = task
        self.trace = trace
        self.retriever = retriever or Retriever(trace=trace, top_k=top_k)
        self.top_k = top_k
        self.max_sub_questions = max_sub_questions

    async def run(self) -> Dict[str, Any]:
        """Execute the pipeline and return a summary dict."""
        self.task.status = TaskStatus.RUNNING
        started = time.perf_counter()
        try:
            plan = await self._plan()
            evidence = await self._retrieve(plan)
            report = await self._synthesize(evidence)
            citation_stats = self.check_citations(report, len(evidence))
            report_path = self._write_outputs(report, evidence)
            self.task.status = TaskStatus.COMPLETED
            extra = {
                "sub_question_count": len(plan),
                "evidence_count": len(evidence),
                "duration_s": round(time.perf_counter() - started, 1),
                **citation_stats,
            }
            self.trace.write_manifest("completed", extra=extra)
            return {
                "task_id": self.task.task_id,
                "report_path": str(report_path),
                **extra,
            }
        except Exception as e:
            self.task.status = TaskStatus.FAILED
            self.trace.add_event(
                EventType.ERROR,
                payload={"stage": "orchestrator", "error": str(e)},
            )
            self.trace.write_manifest("failed", extra={"error": str(e)})
            raise

    # ------------------------------------------------------------------ stages

    async def _plan(self) -> List[Dict[str, str]]:
        span = self.trace.start_span("planning")
        try:
            print("[planning] 生成研究计划...", flush=True)
            content, usage = await call_llm(
                [
                    {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": self.task.question},
                ],
                max_tokens=2000,
            )
            self.trace.add_event(
                EventType.PLANNING, payload={"raw": content}, usage=usage
            )
            try:
                plan = self._parse_plan(content)
            except ValueError as e:
                self.trace.add_event(
                    EventType.ERROR,
                    payload={"stage": "planning", "error": str(e)},
                )
                plan = [
                    {
                        "goal": self.task.question,
                        "search_query": self.task.question,
                    }
                ]
            self.trace.add_event(EventType.PLANNING, payload={"plan": plan})
            print(f"[planning] 拆分为 {len(plan)} 个子问题：", flush=True)
            for i, item in enumerate(plan, 1):
                print(f"  {i}. {item['goal']}", flush=True)
            return plan
        finally:
            self.trace.end_span(span)

    @staticmethod
    def _parse_plan(content: str) -> List[Dict[str, str]]:
        data = extract_json(content)
        if isinstance(data, dict):
            data = data.get("sub_questions") or data.get("plan") or [data]
        if not isinstance(data, list):
            raise ValueError(f"Plan is not a JSON array: {content[:200]}")
        plan = [
            item
            for item in data
            if isinstance(item, dict) and item.get("search_query")
        ]
        if not plan:
            raise ValueError("Plan contains no usable sub-questions")
        return plan

    async def _retrieve(self, plan: List[Dict[str, str]]) -> List[Evidence]:
        evidence: List[Evidence] = []
        for i, item in enumerate(plan, 1):
            span = self.trace.start_span(f"step-{i}")
            try:
                print(
                    f"[step {i}/{len(plan)}] 检索: \"{item['search_query']}\"",
                    flush=True,
                )
                found = await self.retriever.search(
                    item["search_query"], top_k=self.top_k, fetch_content=True
                )
                print(f"        -> {len(found)} 条证据", flush=True)
                evidence.extend(found)
            finally:
                self.trace.end_span(span)
        print(f"[retrieval] 去重后共 {len(dedup_evidence(evidence))} 条证据", flush=True)
        return dedup_evidence(evidence)

    async def _synthesize(self, evidence: List[Evidence]) -> str:
        span = self.trace.start_span("synthesis")
        try:
            print(f"[synthesis] 基于 {len(evidence)} 条证据生成报告...", flush=True)
            content, usage = await call_llm(
                [
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Research question: {self.task.question}\n\n"
                        + self._evidence_context(evidence),
                    },
                ],
                max_tokens=6000,
            )
            self.trace.add_event(
                EventType.SYNTHESIS,
                payload={"evidence_count": len(evidence)},
                usage=usage,
            )
            return content
        finally:
            self.trace.end_span(span)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _evidence_context(evidence: List[Evidence], max_chars: int = 1000) -> str:
        lines = []
        for i, ev in enumerate(evidence, 1):
            text = ev.snippet or (ev.full_text or "")[:max_chars]
            lines.append(
                f"[E{i}] {ev.title}\n  URL: {ev.url}\n  Source: {ev.source}\n  {text}"
            )
        return "\n\n".join(lines)

    @staticmethod
    def check_citations(report: str, evidence_count: int) -> Dict[str, Any]:
        """Validate [E#] citations in the report against the evidence list."""
        cited = set()
        invalid = set()
        for match in re.finditer(r"\[E(\d+)\]", report):
            index = int(match.group(1))
            if 1 <= index <= evidence_count:
                cited.add(index)
            else:
                invalid.add(index)
        return {
            "cited_evidence": sorted(cited),
            "invalid_citations": sorted(invalid),
            "citation_coverage": (
                round(len(cited) / evidence_count, 3) if evidence_count else 0.0
            ),
        }

    def _write_outputs(self, report: str, evidence: List[Evidence]):
        report_path = self.trace.task_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        evidence_path = self.trace.task_dir / "evidence.json"
        evidence_path.write_text(
            json.dumps(
                [ev.model_dump(mode="json") for ev in evidence],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return report_path
