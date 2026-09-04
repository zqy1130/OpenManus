import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(timezone.utc)


def new_id(prefix: str = "") -> str:
    """Generate a unique short id with an optional prefix."""
    suffix = uuid.uuid4().hex[:12]
    return f"{prefix}{suffix}" if prefix else suffix


class TaskStatus(str, Enum):
    """Lifecycle states of a research task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StepStatus(str, Enum):
    """Lifecycle states of a single research step."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ClaimStatus(str, Enum):
    """Verification status of a claim against its evidence."""

    SUPPORTED = "supported"
    DISPUTED = "disputed"
    UNSUPPORTED = "unsupported"


class EventType(str, Enum):
    """Types of events recorded in the execution trace."""

    SPAN_START = "span_start"
    SPAN_END = "span_end"
    LLM = "llm"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    PLANNING = "planning"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    ERROR = "error"
    SYSTEM = "system"


class Budget(BaseModel):
    """Resource limits for a research task."""

    max_steps: int = Field(default=12, description="Maximum number of research steps")
    max_retries_per_step: int = Field(
        default=3, description="Maximum retries for a single step"
    )
    timeout_s: Optional[float] = Field(
        default=None, description="Overall task timeout in seconds"
    )
    max_tokens: Optional[int] = Field(
        default=None, description="Total LLM token budget across the task"
    )


class ResearchTask(BaseModel):
    """A research question plus its constraints, budget and lifecycle state."""

    task_id: str = Field(default_factory=lambda: new_id("task-"))
    question: str = Field(..., description="The research question to answer")
    constraints: List[str] = Field(
        default_factory=list,
        description="User constraints such as language, scope or source requirements",
    )
    budget: Budget = Field(default_factory=Budget)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extra task-level metadata"
    )


class ResearchStep(BaseModel):
    """A single sub-question produced by the planner, with dependencies."""

    step_id: str = Field(default_factory=lambda: new_id("step-"))
    index: int = Field(default=0, description="Position of this step in the plan")
    goal: str = Field(..., description="What this step aims to find out")
    depends_on: List[str] = Field(
        default_factory=list,
        description="step_ids that must finish before this step starts",
    )
    success_criteria: str = Field(
        default="", description="Condition that defines a successful step"
    )
    max_retries: int = Field(default=3, description="Maximum retries for this step")
    retry_count: int = Field(default=0, description="Number of retries already used")
    status: StepStatus = StepStatus.PENDING
    error: Optional[str] = Field(
        default=None, description="Last error message if the step failed"
    )
    input_evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence ids consumed as input"
    )
    output_evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence ids produced by this step"
    )
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class Evidence(BaseModel):
    """A single retrieved source, hashable and traceable to its origin."""

    evidence_id: str = Field(default_factory=lambda: new_id("E-"))
    url: str = ""
    title: str = ""
    snippet: str = Field(default="", description="Short text snippet from the source")
    full_text: Optional[str] = Field(
        default=None, description="Fetched page content if available"
    )
    source: str = Field(
        default="", description="Search engine name or 'local' for local documents"
    )
    doc_id: Optional[str] = Field(
        default=None, description="Local document id (used from Phase 2)"
    )
    chunk_id: Optional[str] = Field(
        default=None, description="Local chunk id (used from Phase 2)"
    )
    retrieval_query: str = Field(
        default="", description="Query that retrieved this evidence"
    )
    score: Optional[float] = Field(default=None, description="Retrieval score")
    source_quality: Optional[float] = Field(
        default=None, description="Credibility score assigned by the verifier"
    )
    retrieved_at: datetime = Field(default_factory=utc_now)
    content_hash: str = Field(
        default="", description="SHA-256 of the content, used for deduplication"
    )

    @model_validator(mode="after")
    def fill_content_hash(self) -> "Evidence":
        if not self.content_hash:
            content = self.full_text or self.snippet
            if content:
                self.content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return self

    @classmethod
    def from_search_result(cls, result: Any, query: str = "") -> "Evidence":
        """Build an Evidence from a WebSearch.SearchResult-like object.

        Duck-typed on purpose so models.py does not depend on app.tool.web_search.
        """
        return cls(
            url=getattr(result, "url", ""),
            title=getattr(result, "title", ""),
            snippet=getattr(result, "description", "") or "",
            full_text=getattr(result, "raw_content", None),
            source=getattr(result, "source", ""),
            retrieval_query=query,
        )


class Claim(BaseModel):
    """A conclusion produced by the synthesizer, tied to evidence ids."""

    claim_id: str = Field(default_factory=lambda: new_id("C-"))
    text: str = Field(..., description="The claim or conclusion text")
    supporting_evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence ids supporting this claim"
    )
    contradicting_evidence_ids: List[str] = Field(
        default_factory=list, description="Evidence ids contradicting this claim"
    )
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    status: ClaimStatus = ClaimStatus.UNSUPPORTED
    generated_at: datetime = Field(default_factory=utc_now)


class LLMUsage(BaseModel):
    """Token, latency and cost accounting for a single LLM call."""

    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: Optional[float] = None
    estimated_cost_usd: Optional[float] = None


class TraceEvent(BaseModel):
    """A single event in the execution trace, attached to a span.

    Spans form an OpenTelemetry-style hierarchy (task -> phase -> step),
    so a full run can be replayed from the JSONL trace alone.
    """

    event_id: str = Field(default_factory=new_id)
    task_id: str = ""
    event_type: EventType = Field(..., description="Type of the recorded event")
    span_id: str = Field(default="", description="Span this event belongs to")
    parent_span_id: Optional[str] = Field(
        default=None, description="Parent span of the span this event belongs to"
    )
    span_name: Optional[str] = Field(
        default=None, description="Human-readable span name"
    )
    timestamp: datetime = Field(default_factory=utc_now)
    payload: Dict[str, Any] = Field(default_factory=dict)
    usage: Optional[LLMUsage] = Field(
        default=None, description="Token/latency/cost info for LLM events"
    )
