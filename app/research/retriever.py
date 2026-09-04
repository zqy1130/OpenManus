"""Web retrieval that turns search results into standardized Evidence objects."""

from typing import Any, List, Optional

from app.logger import logger
from app.research.models import Evidence, EventType
from app.research.trace import TraceWriter
from app.tool.web_search import WebSearch


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
        self, query: str, top_k: Optional[int] = None, fetch_content: bool = True
    ) -> List[Evidence]:
        """Run one search and return standardized, deduplicated evidence."""
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
