"""Unit tests for app.research.memory."""

import json
from pathlib import Path

from app.research.memory import Lesson, LessonStore


def make_lesson(strategy, category="fact_lookup", question="q"):
    return Lesson(
        task_id="task-x", category=category, question=question, strategy=strategy
    )


class TestLessonStore:
    def test_add_and_load_roundtrip(self, tmp_path: Path):
        store = LessonStore(path=tmp_path / "lessons.jsonl")
        store.add([make_lesson("search official announcements first")])
        store.add([make_lesson("add year to queries to reduce ambiguity")])
        loaded = store.load()
        assert len(loaded) == 2
        assert loaded[0].strategy.startswith("search official")

    def test_search_retrieves_relevant(self, tmp_path: Path):
        store = LessonStore(path=tmp_path / "lessons.jsonl")
        store.add(
            [
                make_lesson("add year numbers to queries to reduce ambiguity"),
                make_lesson("use official product pages for release dates"),
                make_lesson("break long comparisons into pairwise sub-questions"),
            ]
        )
        hits = store.search("when was the model released what year", top_k=2)
        assert hits
        assert any("year" in h.strategy for h in hits)

    def test_search_empty_store(self, tmp_path: Path):
        store = LessonStore(path=tmp_path / "none.jsonl")
        assert store.search("anything") == []

    def test_corrupted_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "lessons.jsonl"
        store = LessonStore(path=path)
        store.add([make_lesson("valid lesson")])
        with path.open("a", encoding="utf-8") as f:
            f.write("{not valid json}\n")
        loaded = store.load()
        assert len(loaded) == 1
