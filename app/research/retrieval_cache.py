"""Retrieval result cache: makes evaluation runs deterministic.

Web search results vary between runs; that variance dominates end-to-end
eval noise. The cache stores evidence per query string so repeated runs
receive identical evidence.

Modes:
- "off":  no caching (production behavior, live search every time)
- "on":   serve from cache when present, otherwise search and store
- "read": only serve from cache, never hit the network (strict replay)

Seeding: seed_from_outputs() imports evidence from previous runs'
evidence.json files, so a warm cache can be built from history for free.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

from app.research.models import Evidence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "research"
DEFAULT_CACHE_PATH = DATA_ROOT / "retrieval_cache.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "research"


class RetrievalCache:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_CACHE_PATH
        self._lock = threading.Lock()
        self._data: Optional[Dict[str, List[dict]]] = None

    def _load(self) -> Dict[str, List[dict]]:
        if self._data is not None:
            return self._data
        data: Dict[str, List[dict]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    query = entry.get("query")
                    items = entry.get("evidence")
                    if query and isinstance(items, list):
                        data[query] = items
                except json.JSONDecodeError:
                    continue
        self._data = data
        return data

    def get(self, query: str) -> Optional[List[Evidence]]:
        with self._lock:
            items = self._load().get(query)
        if items is None:
            return None
        return [Evidence.model_validate(item) for item in items]

    def put(self, query: str, evidence: List[Evidence]) -> None:
        items = [ev.model_dump(mode="json") for ev in evidence]
        with self._lock:
            data = self._load()
            data[query] = items
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                for q, evs in data.items():
                    f.write(
                        json.dumps({"query": q, "evidence": evs}, ensure_ascii=False)
                        + "\n"
                    )

    def seed_from_outputs(
        self, output_root: Optional[Path] = None
    ) -> int:
        """Import evidence from all previous task dirs; returns count added."""
        root = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
        count = 0
        for evidence_path in sorted(root.glob("*/evidence.json")):
            try:
                items = json.loads(evidence_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for item in items:
                query = item.get("retrieval_query", "")
                if not query:
                    continue
                with self._lock:
                    data = self._load()
                    data.setdefault(query, [])
                    if not any(
                        existing.get("url") == item.get("url")
                        and existing.get("content_hash") == item.get("content_hash")
                        for existing in data[query]
                    ):
                        data[query].append(item)
                        count += 1
        self.put_many_loaded()
        return count

    def put_many_loaded(self) -> None:
        with self._lock:
            data = self._load()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                for q, evs in data.items():
                    f.write(
                        json.dumps({"query": q, "evidence": evs}, ensure_ascii=False)
                        + "\n"
                    )

    def __len__(self) -> int:
        return len(self._load())


class PlanCache:
    """Caches the planner's sub-question plan per research question.

    Plans vary between runs (LLM sampling), which changes the retrieval
    queries and defeats retrieval caching. Caching plans closes that gap.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = (
            Path(path) if path else DATA_ROOT / "plan_cache.jsonl"
        )
        self._lock = threading.Lock()
        self._data: Optional[Dict[str, List[dict]]] = None

    def _load(self) -> Dict[str, List[dict]]:
        if self._data is not None:
            return self._data
        data: Dict[str, List[dict]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    question = entry.get("question")
                    plan = entry.get("plan")
                    if question and isinstance(plan, list):
                        data[question] = plan
                except json.JSONDecodeError:
                    continue
        self._data = data
        return data

    def get(self, question: str) -> Optional[List[dict]]:
        with self._lock:
            return self._load().get(question)

    def put(self, question: str, plan: List[dict]) -> None:
        with self._lock:
            data = self._load()
            data[question] = plan
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                for q, p in data.items():
                    f.write(
                        json.dumps({"question": q, "plan": p}, ensure_ascii=False)
                        + "\n"
                    )

    def seed_from_outputs(self, output_root: Optional[Path] = None) -> int:
        """Import plans from previous task traces; returns count added."""
        from app.research.models import EventType, TraceEvent

        root = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
        count = 0
        for report_path in sorted(root.glob("*/report.md")):
            task_dir = report_path.parent
            trace_path = task_dir / "trace.jsonl"
            if not trace_path.exists():
                continue
            lines = report_path.read_text(encoding="utf-8").splitlines()
            question = lines[1] if len(lines) > 1 else ""
            if not question:
                continue
            plan = None
            try:
                for line in trace_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    event = TraceEvent.model_validate(json.loads(line))
                    if event.event_type == EventType.PLANNING and "plan" in event.payload:
                        plan = event.payload["plan"]
            except (json.JSONDecodeError, KeyError):
                continue
            if plan and self.get(question) is None:
                self.put(question, plan)
                count += 1
        return count

    def __len__(self) -> int:
        return len(self._load())
