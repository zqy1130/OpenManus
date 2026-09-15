"""Unit tests for app.research.retrieval_cache (fully offline)."""

import json
from pathlib import Path

import pytest

from app.research.models import Evidence
from app.research.retrieval_cache import RetrievalCache


def make_evidence(url, query="q"):
    return Evidence(url=url, snippet=f"content of {url}", retrieval_query=query)


class TestRetrievalCache:
    def test_put_and_get_roundtrip(self, tmp_path: Path):
        cache = RetrievalCache(path=tmp_path / "cache.jsonl")
        evidence = [make_evidence("https://a.com", "q1")]
        cache.put("q1", evidence)
        assert len(cache) == 1

        reloaded = RetrievalCache(path=tmp_path / "cache.jsonl")
        got = reloaded.get("q1")
        assert got is not None
        assert got[0].url == "https://a.com"
        assert got[0].retrieval_query == "q1"

    def test_missing_query_returns_none(self, tmp_path: Path):
        cache = RetrievalCache(path=tmp_path / "cache.jsonl")
        assert cache.get("absent") is None

    def test_seed_from_outputs(self, tmp_path: Path):
        output_root = tmp_path / "outputs"
        task_dir = output_root / "task-x"
        task_dir.mkdir(parents=True)
        evidence = [
            make_evidence("https://a.com", "seed query").model_dump(mode="json"),
            make_evidence("https://b.com", "seed query").model_dump(mode="json"),
            make_evidence("https://c.com", "other query").model_dump(mode="json"),
        ]
        (task_dir / "evidence.json").write_text(
            json.dumps(evidence, ensure_ascii=False), encoding="utf-8"
        )

        cache = RetrievalCache(path=tmp_path / "cache.jsonl")
        added = cache.seed_from_outputs(output_root)
        assert added == 3
        assert len(cache) == 2
        assert len(cache.get("seed query")) == 2

    def test_seed_dedups_duplicates(self, tmp_path: Path):
        output_root = tmp_path / "outputs"
        task_dir = output_root / "task-x"
        task_dir.mkdir(parents=True)
        ev = make_evidence("https://a.com", "dup query").model_dump(mode="json")
        for name in ("task-x", "task-y"):
            d = output_root / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "evidence.json").write_text(
                json.dumps([ev], ensure_ascii=False), encoding="utf-8"
            )
        cache = RetrievalCache(path=tmp_path / "cache.jsonl")
        added = cache.seed_from_outputs(output_root)
        assert added == 1
        assert len(cache.get("dup query")) == 1

    def test_corrupted_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "cache.jsonl"
        path.write_text("{broken json}\n", encoding="utf-8")
        cache = RetrievalCache(path=path)
        assert cache.get("anything") is None


class TestRetrieverCacheModes:
    @pytest.mark.asyncio
    async def test_read_mode_never_hits_network(self, tmp_path: Path):
        from app.research.retriever import Retriever

        cache = RetrievalCache(path=tmp_path / "cache.jsonl")
        cache.put("cached query", [make_evidence("https://a.com", "cached query")])

        class ExplodingWebSearch:
            async def execute(self, **kwargs):
                raise RuntimeError("network should not be called")

        retriever = Retriever(
            web_search=ExplodingWebSearch(), cache=cache, cache_mode="read"
        )
        got = await retriever.search("cached query")
        assert len(got) == 1

        missed = await retriever.search("uncached query")
        assert missed == []

    @pytest.mark.asyncio
    async def test_on_mode_writes_after_search(self, tmp_path: Path):
        from app.research.retriever import Retriever

        cache = RetrievalCache(path=tmp_path / "cache.jsonl")

        class FakeSearchResult:
            def __init__(self, url, title="", description="", raw_content=None, source="fake"):
                self.url = url
                self.title = title
                self.description = description
                self.raw_content = raw_content
                self.source = source

        class FakeWebSearch:
            async def execute(self, query, num_results=5, fetch_content=False):
                return type(
                    "Response",
                    (),
                    {
                        "results": [
                            FakeSearchResult(
                                "https://a.com", "A", "description text"
                            )
                        ],
                        "error": None,
                    },
                )()

        retriever = Retriever(
            web_search=FakeWebSearch(), cache=cache, cache_mode="on"
        )
        got = await retriever.search("fresh query")
        assert len(got) == 1
        # second call is served from cache
        got2 = await retriever.search("fresh query")
        assert len(got2) == 1
        assert cache.get("fresh query") is not None

    def test_invalid_mode_rejected(self):
        from app.research.retriever import Retriever

        with pytest.raises(ValueError):
            Retriever(cache_mode="bogus")
