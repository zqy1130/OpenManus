"""Web and local retrieval, returning standardized Evidence objects."""

from typing import Any, Dict, List, Optional

from app.logger import logger
from app.research.bm25 import BM25Index
from app.research.documents import Chunk, load_corpus
from app.research.embeddings import DEFAULT_EMBEDDING_MODEL, VectorIndex
from app.research.models import Evidence, EventType
from app.research.trace import TraceWriter
from app.tool.web_search import WebSearch

RRF_K = 60


def dedup_evidence(evidence: List[Evidence]) -> List[Evidence]:
    """Remove duplicate evidence by URL first, then by content hash."""
    seen_urls = set()
    seen_hashes = set()
    out: List[Evidence] = []
    for ev in evidence:
        if ev.url and ev.url in seen_urls:
            continue
        if ev.content_hash and ev.content_hash in seen_hashes:
            continue
        if ev.url:
            seen_urls.add(ev.url)
        if ev.content_hash:
            seen_hashes.add(ev.content_hash)
        out.append(ev)
    return out


class Retriever:
    """Search the web via OpenManus WebSearch and return deduplicated Evidence.

    Retrieval failures are recorded in the trace and yield an empty list
    instead of raising, so the orchestrator can keep going.
    """

    def __init__(
        self,
        trace: Optional[TraceWriter] = None,
        top_k: int = 5,
        web_search: Optional[Any] = None,
    ):
        self.trace = trace
        self.top_k = top_k
        self._web_search = web_search if web_search is not None else WebSearch()

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        fetch_content: bool = True,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Evidence]:
        """Run one web search and return standardized, deduplicated evidence.

        ``filters`` is accepted for interface parity with the local
        retriever; the web engines do not support corpus-level filtering.
        """
        top_k = top_k or self.top_k
        span_id: Optional[str] = None
        if self.trace:
            span_id = self.trace.start_span("retrieval")
            self.trace.add_event(
                EventType.RETRIEVAL,
                payload={"query": query, "top_k": top_k, "fetch_content": fetch_content},
            )
        try:
            response = await self._web_search.execute(
                query=query, num_results=top_k, fetch_content=fetch_content
            )
            results = getattr(response, "results", None) or []
            if getattr(response, "error", None):
                if self.trace:
                    self.trace.add_event(
                        EventType.ERROR,
                        payload={"query": query, "error": response.error},
                    )
                return []
            evidence = dedup_evidence(
                [Evidence.from_search_result(r, query=query) for r in results]
            )
            if self.trace:
                self.trace.add_event(
                    EventType.RETRIEVAL,
                    payload={"query": query, "evidence_count": len(evidence)},
                )
            return evidence
        except Exception as e:
            logger.warning(f"Retrieval failed for query '{query}': {e}")
            if self.trace:
                self.trace.add_event(
                    EventType.ERROR,
                    payload={"query": query, "error": str(e)},
                )
            return []
        finally:
            if self.trace and span_id:
                self.trace.end_span(span_id)


class LocalRetriever:
    """Hybrid BM25 + dense retrieval over the local document corpus.

    Both rankings are merged with Reciprocal Rank Fusion; results come
    back as Evidence objects carrying doc_id, chunk_id and retrieval
    scores. Vector index construction hits the embedding API once and is
    cached under data/research/embeddings/<model>/.
    """

    def __init__(
        self,
        trace: Optional[TraceWriter] = None,
        corpus_dir: Optional[Any] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        reranker: Optional[Any] = None,
        mode: str = "hybrid",
    ):
        if mode not in ("hybrid", "bm25", "vector"):
            raise ValueError(f"Unknown retrieval mode: {mode}")
        self.trace = trace
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.mode = mode
        self._bm25 = BM25Index()
        self._vector = VectorIndex(model=embedding_model)
        self._chunks: Dict[str, Chunk] = {}
        self._doc_titles: Dict[str, str] = {}
        self._corpus_dir = corpus_dir

    async def index_corpus(self, force_rebuild: bool = False) -> int:
        """Load and index the corpus; returns the number of chunks."""
        documents = load_corpus(self._corpus_dir)
        for doc in documents:
            self._doc_titles[doc.doc_id] = doc.title
            for chunk in doc.chunks:
                self._chunks[chunk.chunk_id] = chunk
        self._bm25.build(
            [{"id": c.chunk_id, "text": c.text} for c in self._chunks.values()]
        )
        if not force_rebuild and self._vector.exists():
            self._vector = VectorIndex.load(model=self.embedding_model)
        else:
            await self._vector.build(
                [c.chunk_id for c in self._chunks.values()],
                [c.text for c in self._chunks.values()],
            )
        return len(self._chunks)

    # ------------------------------------------------------------------ search

    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Evidence]:
        """Hybrid search with RRF fusion; filters can carry doc_ids."""
        if not self._chunks:
            await self.index_corpus()

        if self.mode == "bm25":
            fused = self._bm25.search(query, top_k=top_k * 2)
        elif self.mode == "vector":
            fused = await self._vector.search(query, top_k=top_k * 2)
        else:
            bm25_hits = self._bm25.search(query, top_k=top_k * 2)
            vector_hits = await self._vector.search(query, top_k=top_k * 2)
            fused = self._rrf_fuse(bm25_hits, vector_hits)

        doc_filter = self._doc_filter(filters)
        evidence: List[Evidence] = []
        for chunk_id, score in fused:
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                continue
            if doc_filter is not None and chunk.doc_id not in doc_filter:
                continue
            evidence.append(self._to_evidence(query, chunk, score))
            if len(evidence) >= top_k:
                break

        if self.reranker is not None and evidence:
            evidence = await self.reranker.rerank(query, evidence)

        if self.trace:
            span = self.trace.start_span("retrieval")
            try:
                self.trace.add_event(
                    EventType.RETRIEVAL,
                    payload={
                        "query": query,
                        "engine": "local",
                        "evidence_count": len(evidence),
                    },
                )
            finally:
                self.trace.end_span(span)
        return evidence

    @staticmethod
    def _rrf_fuse(
        bm25_hits: List[Any], vector_hits: List[Any]
    ) -> List[tuple]:
        scores: Dict[str, float] = {}
        for rank, (chunk_id, _) in enumerate(bm25_hits, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (chunk_id, _) in enumerate(vector_hits, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked

    @staticmethod
    def _doc_filter(filters: Optional[Dict[str, Any]]) -> Optional[set]:
        if not filters:
            return None
        doc_ids = filters.get("doc_ids") or []
        if filters.get("doc_id"):
            doc_ids = list(doc_ids) + [filters["doc_id"]]
        return set(doc_ids) if doc_ids else None

    def _to_evidence(self, query: str, chunk: Chunk, score: float) -> Evidence:
        return Evidence(
            url="",
            title=self._doc_titles.get(chunk.doc_id, chunk.doc_id),
            snippet=chunk.text[:600],
            full_text=chunk.text,
            source="local",
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            retrieval_query=query,
            score=round(score, 5),
        )

