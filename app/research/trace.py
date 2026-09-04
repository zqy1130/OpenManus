import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.research.models import EventType, LLMUsage, TraceEvent, new_id, utc_now

# Computed locally instead of importing app.config: importing app.config would
# instantiate the whole Config singleton and fail when config.toml is invalid
# (e.g. a fresh checkout with only the example config, which lacks [daytona]).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "research"


class TraceWriter:
    """Append-only JSONL trace writer with lightweight span management.

    Each task gets its own directory under the output root:

        <output_root>/<task_id>/trace.jsonl
        <output_root>/<task_id>/summary.json

    Span start/end events are recorded in the same JSONL stream, so the whole
    execution hierarchy can be reconstructed from the trace file alone.
    Payload values under sensitive keys (api keys, passwords, ...) are
    redacted before writing.
    """

    REDACT_KEYS = {
        "api_key",
        "apikey",
        "api-key",
        "authorization",
        "access_token",
        "refresh_token",
        "password",
        "secret",
    }

    def __init__(self, task_id: str, output_root: Optional[Path] = None) -> None:
        self.task_id = task_id
        self.output_root = Path(output_root) if output_root else DEFAULT_OUTPUT_ROOT
        self.task_dir = self.output_root / task_id
        self.trace_path = self.task_dir / "trace.jsonl"
        self.task_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._file = self.trace_path.open("a", encoding="utf-8")
        self._span_stack: List[str] = []
        self._span_parent: Dict[str, Optional[str]] = {}
        self._span_names: Dict[str, str] = {}
        self._span_count = 0
        self._event_count = 0
        self._tokens_in = 0
        self._tokens_out = 0
        self._cost_usd = 0.0
        self._errors = 0
        self._started_at = utc_now()

        self._root_span_id: Optional[str] = None
        self._root_span_id = self.start_span("task", parent_span_id=None)

    # ------------------------------------------------------------------ spans

    @property
    def current_span_id(self) -> Optional[str]:
        """Id of the innermost open span, if any."""
        return self._span_stack[-1] if self._span_stack else None

    def start_span(self, name: str, parent_span_id: Optional[str] = None) -> str:
        """Open a new span; without an explicit parent it nests in the current span."""
        with self._lock:
            if parent_span_id is None and self._span_stack:
                parent_span_id = self._span_stack[-1]
            span_id = new_id("span-")
            self._span_parent[span_id] = parent_span_id
            self._span_names[span_id] = name
            self._span_stack.append(span_id)
            self._span_count += 1
            self._write_event(
                EventType.SPAN_START,
                span_id=span_id,
                parent_span_id=parent_span_id,
                span_name=name,
                payload={},
            )
            return span_id

    def end_span(self, span_id: Optional[str] = None) -> str:
        """Close a span; without an explicit id the innermost open span is closed."""
        with self._lock:
            if span_id is None:
                span_id = self._span_stack.pop() if self._span_stack else self._root_span_id
            elif span_id in self._span_stack:
                self._span_stack.remove(span_id)
            self._write_event(
                EventType.SPAN_END,
                span_id=span_id,
                parent_span_id=self._span_parent.get(span_id),
                span_name=self._span_names.get(span_id),
                payload={},
            )
            return span_id

    # ------------------------------------------------------------------ events

    def add_event(
        self,
        event_type: EventType,
        payload: Optional[Dict[str, Any]] = None,
        usage: Optional[LLMUsage] = None,
        span_id: Optional[str] = None,
        span_name: Optional[str] = None,
    ) -> TraceEvent:
        """Record an event in the current span (or the root span if none is open)."""
        with self._lock:
            if span_id is None:
                span_id = self.current_span_id or self._root_span_id
            if span_name is None:
                span_name = self._span_names.get(span_id)
            return self._write_event(
                event_type,
                span_id=span_id,
                parent_span_id=self._span_parent.get(span_id),
                span_name=span_name,
                payload=payload or {},
                usage=usage,
            )

    def _write_event(
        self,
        event_type: EventType,
        *,
        span_id: Optional[str],
        parent_span_id: Optional[str],
        span_name: Optional[str],
        payload: Dict[str, Any],
        usage: Optional[LLMUsage] = None,
    ) -> TraceEvent:
        event = TraceEvent(
            task_id=self.task_id,
            event_type=event_type,
            span_id=span_id or "",
            parent_span_id=parent_span_id,
            span_name=span_name,
            payload=self.redact(payload),
            usage=usage,
        )
        self._file.write(
            json.dumps(event.model_dump(mode="json"), ensure_ascii=False) + "\n"
        )
        self._file.flush()
        self._event_count += 1
        if usage:
            self._tokens_in += usage.input_tokens
            self._tokens_out += usage.output_tokens
            if usage.estimated_cost_usd:
                self._cost_usd += usage.estimated_cost_usd
        if event_type == EventType.ERROR:
            self._errors += 1
        return event

    # ------------------------------------------------------------------ redaction

    @classmethod
    def redact(cls, obj: Any) -> Any:
        """Recursively replace values under sensitive keys with '[REDACTED]'."""
        if isinstance(obj, dict):
            return {
                key: ("[REDACTED]" if cls._is_sensitive(key) else cls.redact(value))
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [cls.redact(item) for item in obj]
        return obj

    @staticmethod
    def _is_sensitive(key: str) -> bool:
        lowered = key.lower()
        if lowered in TraceWriter.REDACT_KEYS:
            return True
        return any(
            marker in lowered
            for marker in ("secret", "password", "api_key", "apikey", "api-key", "authorization")
        )

    # ------------------------------------------------------------------ summary

    def write_manifest(self, status: str, extra: Optional[Dict[str, Any]] = None) -> Path:
        """Write a summary.json with aggregate stats for this task."""
        finished_at = utc_now()
        summary = {
            "task_id": self.task_id,
            "status": status,
            "event_count": self._event_count,
            "span_count": self._span_count,
            "tokens": {"input": self._tokens_in, "output": self._tokens_out},
            "estimated_cost_usd": round(self._cost_usd, 6),
            "error_count": self._errors,
            "started_at": self._started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "duration_s": round((finished_at - self._started_at).total_seconds(), 3),
        }
        if extra:
            summary.update(extra)
        path = self.task_dir / "summary.json"
        with self._lock:
            path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return path

    # ------------------------------------------------------------------ replay

    @staticmethod
    def load_trace(trace_path: Path) -> List[TraceEvent]:
        """Read a trace.jsonl file back into TraceEvent objects."""
        events: List[TraceEvent] = []
        for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(TraceEvent.model_validate(json.loads(line)))
        return events

    @staticmethod
    def build_span_tree(events: List[TraceEvent]) -> Dict[str, Dict[str, Any]]:
        """Reconstruct the span hierarchy from recorded events.

        Returns a mapping of span_id -> {name, parent_span_id, children,
        events, started_at, ended_at}.
        """
        spans: Dict[str, Dict[str, Any]] = {}
        for event in events:
            if event.event_type != EventType.SPAN_START:
                continue
            spans[event.span_id] = {
                "name": event.span_name,
                "parent_span_id": event.parent_span_id,
                "children": [],
                "events": [],
                "started_at": event.timestamp,
                "ended_at": None,
            }
        for event in events:
            if event.span_id not in spans:
                continue
            if event.event_type == EventType.SPAN_END:
                spans[event.span_id]["ended_at"] = event.timestamp
            elif event.event_type != EventType.SPAN_START:
                spans[event.span_id]["events"].append(event)
        for span_id, span in spans.items():
            parent = span.get("parent_span_id")
            if parent and parent in spans:
                spans[parent]["children"].append(span_id)
        return spans

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Flush and close the trace file."""
        with self._lock:
            if not self._file.closed:
                self._file.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *args) -> None:
        self.close()
