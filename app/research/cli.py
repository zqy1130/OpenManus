"""Research agent CLI entry: python -m app.research.cli --question "..." """

import argparse
import asyncio

from app.research.models import ResearchTask
from app.research.orchestrator import ResearchOrchestrator
from app.research.retriever import Retriever
from app.research.trace import TraceWriter


async def run_research(question: str, top_k: int, max_sub_questions: int) -> None:
    task = ResearchTask(question=question)
    with TraceWriter(task.task_id) as trace:
        orchestrator = ResearchOrchestrator(
            task=task,
            trace=trace,
            retriever=Retriever(trace=trace, top_k=top_k),
            top_k=top_k,
            max_sub_questions=max_sub_questions,
        )
        summary = await orchestrator.run()
    print(f"Report: outputs/research/{task.task_id}/report.md")
    print(
        f"Evidence: {summary['evidence_count']} | "
        f"Sub-questions: {summary['sub_question_count']} | "
        f"Citation coverage: {summary['citation_coverage']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the research agent on a question")
    parser.add_argument("--question", type=str, required=True, help="Research question")
    parser.add_argument("--top-k", type=int, default=5, help="Results per sub-question")
    parser.add_argument(
        "--max-sub-questions", type=int, default=5, help="Max sub-questions"
    )
    args = parser.parse_args()
    asyncio.run(run_research(args.question, args.top_k, args.max_sub_questions))


if __name__ == "__main__":
    main()
