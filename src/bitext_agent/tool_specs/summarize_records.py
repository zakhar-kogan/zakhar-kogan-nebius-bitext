"""Versioned `summarize_records` tool spec and implementation."""

from pydantic import BaseModel, Field

from bitext_agent.schemas import SummaryResult, ToolExample, ToolSpec
from bitext_agent.tool_specs import ToolRuntimeContext


class SummarizeRecordsArgs(BaseModel):
    """Arguments for record summarization."""

    category: str | None = Field(default=None, description="Optional category filter.")
    intent: str | None = Field(default=None, description="Optional intent filter.")
    query: str | None = Field(default=None, description="Optional phrase filter.")


def build_spec(context: ToolRuntimeContext) -> ToolSpec:
    """Build the summarize_records tool spec."""

    def summarize_records(
        category: str | None = None, intent: str | None = None, query: str | None = None
    ) -> SummaryResult:
        total, rows = context.repository.sample_for_summary(category=category, intent=intent, query=query)
        if not rows:
            return SummaryResult(summary="No matching records were found.", record_count=0)
        intents = sorted({row.intent for row in rows})[:8]
        response_terms = _response_patterns([row.response for row in rows])
        summary = (
            f"Found {total} matching records. Common intents include {', '.join(intents)}. "
            f"Agent responses usually {response_terms}."
        )
        return SummaryResult(summary=summary, record_count=total)

    return ToolSpec(
        name="summarize_records",
        version="1.0.0",
        description="Summarize customer needs and representative response patterns for matching records.",
        args_schema=SummarizeRecordsArgs,
        output_schema=SummaryResult,
        callable=summarize_records,
        examples=[ToolExample(input={"category": "FEEDBACK"}, output_summary="Grounded summary.")],
        return_summary="Short grounded summary and record count.",
    )


def _response_patterns(responses: list[str]) -> str:
    joined = " ".join(responses).lower()
    patterns: list[str] = []
    if "apolog" in joined:
        patterns.append("acknowledge or apologize for the issue")
    if "assist" in joined or "help" in joined:
        patterns.append("offer assistance")
    if "please" in joined or "provide" in joined:
        patterns.append("ask for clarifying details")
    if not patterns:
        patterns.append("provide a direct support response")
    return ", ".join(patterns)

