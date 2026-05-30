"""Shared Pydantic schemas for graph state, tools, memory, and responses."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field


RouteKind = Literal["structured", "unstructured", "out_of_scope", "recommendation"]
ToolStatus = Literal["active", "deprecated"]


class ReasoningStep(BaseModel):
    """A user-visible trace item from routing, tool use, or generation."""

    kind: str
    title: str
    detail: str = ""


class AgentResponse(BaseModel):
    """Final response returned to CLI, Streamlit, and tests."""

    answer: str
    route: RouteKind
    reasoning: list[ReasoningStep] = Field(default_factory=list)
    suggested_query: str | None = None


AgentEventKind = Literal[
    "route",
    "tool",
    "observation",
    "final",
    "fallback",
    "memory",
    "recommendation",
    "cancelled",
    "error",
]


class AgentEvent(BaseModel):
    """Incremental user-visible event emitted while a turn is running."""

    kind: AgentEventKind
    title: str
    detail: str = ""
    answer_delta: str = ""
    final_response: AgentResponse | None = None


class RouterDecision(BaseModel):
    """Router output before the agent chooses tools."""

    route: RouteKind
    reason: str


class ToolExample(BaseModel):
    """Example invocation for a versioned Python tool spec."""

    input: dict[str, Any]
    output_summary: str


class ToolSpec(BaseModel):
    """Versioned metadata and callable contract for a tool."""

    name: str
    version: str
    description: str
    args_schema: type[BaseModel]
    output_schema: type[BaseModel]
    callable: Callable[..., BaseModel]
    examples: list[ToolExample] = Field(default_factory=list)
    return_summary: str
    status: ToolStatus = "active"

    model_config = {"arbitrary_types_allowed": True}


class CategoryList(BaseModel):
    """Available dataset categories."""

    categories: list[str]


class IntentList(BaseModel):
    """Available dataset intents, optionally filtered by category."""

    intents: list[str]


class DatasetRow(BaseModel):
    """A compact dataset record shown to users."""

    row_id: int
    category: str
    intent: str
    instruction: str
    response: str


class SearchRowsResult(BaseModel):
    """Search result with an ID that later tools can reuse."""

    search_id: str
    total_matches: int
    rows: list[DatasetRow]


class CountRowsResult(BaseModel):
    """Count result for category, intent, query, or previous search filters."""

    count: int
    filters: dict[str, Any]


class ExamplesResult(BaseModel):
    """Example rows with paging metadata."""

    rows: list[DatasetRow]
    offset: int
    next_offset: int | None
    total_matches: int


class IntentDistributionResult(BaseModel):
    """Intent counts for a category or the whole dataset."""

    distribution: list[dict[str, int | str]]


class SummaryResult(BaseModel):
    """Dataset summary generated from matching records."""

    summary: str
    record_count: int


class ProfileFact(BaseModel):
    """A durable user profile fact."""

    id: int | None = None
    user_uuid: str
    kind: str
    fact: str
    canonical_key: str | None = None
    source: str
    confidence: float = 0.5
    status: str = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ProfileResult(BaseModel):
    """Profile facts visible to the current user."""

    user_uuid: str
    facts: list[ProfileFact]


class RecommendationResult(BaseModel):
    """A suggested query that has not been executed yet."""

    query: str
    reason: str
    pending: bool = True


class RecommendationRefinementResult(BaseModel):
    """Structured interpretation of a user's requested change to a pending recommendation."""

    refined_query: str | None = None
    reason: str
    unclear: bool = False
