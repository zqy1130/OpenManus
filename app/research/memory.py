"""Cross-task procedural memory: lessons learned from previous runs.

Lessons are strategy-level conclusions (what worked, what failed), never
raw webpage content, so prompt injection cannot propagate across tasks.
Stored as JSONL at data/research/lessons.jsonl and retrieved with BM25.
"""

import json
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from app.research.bm25 import BM25Index
from app.research.models import new_id, utc_now

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "research"
DEFAULT_LESSONS_PATH = DATA_ROOT / "lessons.jsonl"


class Lesson(BaseModel):
    lesson_id: str = Field(default_factory=lambda: new_id("lesson-"))
    task_id: str = ""
    category: str = Field(default="", description="Task category from the eval set")
    question: str = Field(default="", description="The question this lesson came from")
    strategy: str = Field(..., description="What was tried / what worked / what failed")
    outcome: str = Field(default="unknown", description="success | failure | mixed")
    created_at: str = Field(default_factory=lambda: utc_now().isoformat())

    @property
    def search_text(self) -> str:
        return f"{self.category} {self.question} {self.strategy}"


class LessonStore:
    """Append-only JSONL store with BM25 retrieval over lessons."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else DEFAULT_LESSONS_PATH

    def load(self) -> List[Lesson]:
        if not self.path.exists():
            return []
        lessons = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    lessons.append(Lesson.model_validate(json.loads(line)))
                except Exception:
                    continue
        return lessons

    def add(self, lessons: List[Lesson]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            for lesson in lessons:
                f.write(lesson.model_dump_json() + "\n")
        return len(lessons)

    def search(self, query: str, top_k: int = 3) -> List[Lesson]:
        lessons = self.load()
        if not lessons:
            return []
        index = BM25Index()
        index.build(
            [
                {"id": lesson.lesson_id, "text": lesson.search_text}
                for lesson in lessons
            ]
        )
        hits = index.search(query, top_k=top_k)
        by_id = {lesson.lesson_id: lesson for lesson in lessons}
        return [by_id[lesson_id] for lesson_id, _ in hits if lesson_id in by_id]
