"""Unit tests for app.research.retriever.LocalRetriever (no API calls)."""

import json
from pathlib import Path

import pytest

from app.research.models import EventType
from app.research.retriever import LocalRetriever
from app.research.trace import TraceWriter

CORPUS_HTML = """<html><head><title>Alpha Test</title></head>
<body><div id="mw-content-text">
<p>AlphaFold predicts protein structures from amino acid sequences.
This paragraph has plenty of text so that it becomes at least one chunk.
More sentences follow to make sure the chunk is reasonably long and searchable.</p>
<h2>History</h2>
<p>The first version of AlphaFold won the CASP competition in 2018.</p>
</div></body></html>"""


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    (tmp_path / "01_alphafold.html").write_text(CORPUS_HTML, encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            [
                {
                    "doc_id": "alphafold",
                    "title": "AlphaFold",
                    "url": "https://example.com/alphafold",
                    "file": "01_alphafold.html",
                }
            ]
        ),
        encoding="utf-8",
    )
    return tmp_path


class TestLocalRetrieverBM25:
    @pytest.mark.asyncio
    async def test_index_and_search(self, corpus_dir: Path):
        retriever = LocalRetriever(mode="bm25", corpus_dir=corpus_dir)
        count = await retriever.index_corpus()
        assert count >= 1

        evidence = await retriever.search("protein structure prediction", top_k=3)
        assert evidence
        assert all(ev.source == "local" for ev in evidence)
        assert all(ev.doc_id == "alphafold" for ev in evidence)
        assert evidence[0].chunk_id.startswith("alphafold#")
        assert evidence[0].retrieval_query == "protein structure prediction"
        assert evidence[0].score > 0

    @pytest.mark.asyncio
    async def test_filters_by_doc_id(self, corpus_dir: Path):
        retriever = LocalRetriever(mode="bm25", corpus_dir=corpus_dir)
        await retriever.index_corpus()
        evidence = await retriever.search(
            "protein", top_k=5, filters={"doc_ids": ["other_doc"]}
        )
        assert evidence == []

        evidence = await retriever.search(
            "protein", top_k=5, filters={"doc_id": "alphafold"}
        )
        assert evidence

    @pytest.mark.asyncio
    async def test_trace_events(self, corpus_dir: Path, tmp_path: Path):
        with TraceWriter("t", output_root=tmp_path) as trace:
            retriever = LocalRetriever(trace=trace, mode="bm25", corpus_dir=corpus_dir)
            await retriever.index_corpus()
            await retriever.search("protein", top_k=3)

        events = TraceWriter.load_trace(tmp_path / "t" / "trace.jsonl")
        retrieval = [e for e in events if e.event_type == EventType.RETRIEVAL]
        assert any(e.payload.get("engine") == "local" for e in retrieval)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            LocalRetriever(mode="bogus")


class TestFusion:
    def test_rrf_combines_rankings(self):
        bm25 = [("a", 5.0), ("b", 4.0)]
        vector = [("b", 0.9), ("c", 0.8)]
        fused = LocalRetriever._rrf_fuse(bm25, vector)
        ids = [chunk_id for chunk_id, _ in fused]
        assert ids[0] == "b"  # top-1 in vector, top-2 in bm25
        assert set(ids) == {"a", "b", "c"}

    def test_doc_filter_variants(self):
        assert LocalRetriever._doc_filter(None) is None
        assert LocalRetriever._doc_filter({}) is None
        assert LocalRetriever._doc_filter({"doc_ids": ["x"]}) == {"x"}
        assert LocalRetriever._doc_filter({"doc_id": "y"}) == {"y"}
