"""Full evaluation runner.

Usage: python -m app.research.run_eval --limit 5 [--route-models] [--no-memory]

Runs tasks from data/research/tasks.jsonl through the orchestrator, then
claim-audits each report and judges it against the gold answer. Writes
aggregated metrics (task_success_rate, citation correctness, latency
percentiles, tokens, failure classification) to JSON and prints a table.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.research.claim_check import ReportAuditor
from app.research.memory import LessonStore
from app.research.models import ResearchTask
from app.research.orchestrator import ResearchOrchestrator
from app.research.retriever import Retriever
from app.research.success_judge import SuccessJudge
from app.research.trace import TraceWriter
from app.research.verifier import Validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "research"
TASKS_PATH = DATA_ROOT / "tasks.jsonl"


def load_tasks(limit: Optional[int] = None) -> List[dict]:
    tasks = [
        json.loads(line)
        for line in TASKS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return tasks[:limit] if limit else tasks


def classify_failure(events: List[Any]) -> str:
    for event in events:
        if event.event_type.value == "error":
            stage = str(event.payload.get("stage", ""))
            if stage in ("planning",):
                return "planning_error"
            if event.payload.get("error_type") == "rate_limit":
                return "rate_limit"
            if event.payload.get("error_type"):
                return "retrieval_error"
    return "unknown"


async def run_one(task: dict, args) -> Dict[str, Any]:
    research_task = ResearchTask(
        question=task["question"],
        metadata={"category": task.get("category", ""), "task_id": task["id"]},
    )
    with TraceWriter(research_task.task_id) as trace:
        orchestrator = ResearchOrchestrator(
            task=research_task,
            trace=trace,
            retriever=Retriever(trace=trace, top_k=args.top_k),
            top_k=args.top_k,
            validator=Validator(trace=trace, route_models=args.route_models)
            if not args.no_validator
            else None,
            lesson_store=LessonStore() if not args.no_memory else None,
            use_lessons=not args.no_memory,
            summarize_evidence=not args.no_summarize,
            route_models=args.route_models,
        )
        try:
            await orchestrator.run()
            status = "completed"
        except Exception as e:
            status = f"failed: {type(e).__name__}"

        events = TraceWriter.load_trace(trace.trace_path)
        report_path = trace.task_dir / "report.md"
        report = (
            report_path.read_text(encoding="utf-8")
            if report_path.exists()
            else ""
        )

        row: Dict[str, Any] = {
            "task_id": task["id"],
            "category": task.get("category"),
            "status": status,
        }
        if report:
            audit = await ReportAuditor(route_models=args.route_models).audit_task(
                trace.task_dir
            )
            row.update(
                {
                    "citation_correctness": audit["citation_correctness"],
                    "unsupported_claim_rate": audit["unsupported_claim_rate"],
                    "claim_count": audit["claim_count"],
                }
            )
            verdict = await SuccessJudge(route_models=args.route_models).judge(
                task["question"], task, report
            )
            row.update(
                {
                    "task_success": verdict.task_success,
                    "gold_score": verdict.score,
                    "judge_reason": verdict.reason,
                }
            )
        latencies = [
            e.usage.latency_ms
            for e in events
            if e.usage is not None and e.usage.latency_ms is not None
        ]
        row["llm_latency_ms"] = latencies
        row["error_type"] = classify_failure(events)
        return row


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [r for r in rows if r["status"] == "completed"]
    judged = [r for r in completed if r.get("task_success") is not None]
    corrects = [r["citation_correctness"] for r in completed if r.get("citation_correctness") is not None]
    unsups = [r["unsupported_claim_rate"] for r in completed if r.get("unsupported_claim_rate") is not None]
    latencies = sorted(
        ms for r in completed for ms in r.get("llm_latency_ms", [])
    )

    def pct(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        return values[min(len(values) - 1, int(q * len(values)))]

    from collections import Counter

    return {
        "task_count": len(rows),
        "completed": len(completed),
        "task_success_rate": round(
            sum(1 for r in judged if r["task_success"]) / len(judged), 3
        )
        if judged
        else None,
        "avg_gold_score": round(
            sum(r["gold_score"] for r in judged) / len(judged), 3
        )
        if judged
        else None,
        "avg_citation_correctness": round(sum(corrects) / len(corrects), 3)
        if corrects
        else None,
        "avg_unsupported_claim_rate": round(sum(unsups) / len(unsups), 3)
        if unsups
        else None,
        "llm_latency_p50_ms": round(pct(latencies, 0.5), 1),
        "llm_latency_p95_ms": round(pct(latencies, 0.95), 1),
        "failure_types": dict(Counter(r["error_type"] for r in rows if r["status"] != "completed")),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full research eval")
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--route-models", action="store_true", help="Stage-based model routing")
    parser.add_argument("--no-memory", action="store_true", help="Disable procedural memory")
    parser.add_argument("--no-summarize", action="store_true")
    parser.add_argument("--no-validator", action="store_true")
    parser.add_argument("--arm", type=str, default="default", help="Label for results file")
    args = parser.parse_args()

    tasks = load_tasks(args.limit)
    print(f"Running {len(tasks)} tasks (arm={args.arm} route={args.route_models})")
    rows = []
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task['id']}: {task['question'][:50]}")
        rows.append(await run_one(task, args))

    summary = aggregate(rows)
    out = {"arm": args.arm, "route_models": args.route_models, "summary": summary, "tasks": rows}
    out_path = DATA_ROOT / f"eval_results_{args.arm}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== EVAL SUMMARY =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
