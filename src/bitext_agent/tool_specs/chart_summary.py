"""Versioned `chart_summary` tool spec and implementation."""

from typing import Literal

from pydantic import BaseModel, Field

from bitext_agent.schemas import ChartArtifact, ChartSummaryResult, ToolExample, ToolSpec
from bitext_agent.tool_specs import ToolRuntimeContext


ChartKind = Literal["category_distribution", "intent_distribution", "top_intents"]


class ChartSummaryArgs(BaseModel):
    """Arguments for building chart metadata from deterministic dataset aggregations."""

    chart_kind: ChartKind = Field(description="Dataset aggregation to visualize.")
    category: str | None = Field(default=None, description="Category for intent_distribution charts.")
    limit: int = Field(default=10, ge=1, le=25, description="Maximum rows for top_intents charts.")


def build_spec(context: ToolRuntimeContext) -> ToolSpec:
    """Build the chart_summary tool spec."""

    def chart_summary(
        chart_kind: ChartKind,
        category: str | None = None,
        limit: int = 10,
    ) -> ChartSummaryResult:
        if chart_kind == "category_distribution":
            rows = context.repository.category_distribution()
            artifact = ChartArtifact(
                title="Rows by category",
                x="category",
                y="count",
                rows=rows,
            )
        elif chart_kind == "intent_distribution":
            rows = context.repository.intent_distribution(category=category)
            title = f"Intent distribution for {category.upper()}" if category else "Intent distribution"
            artifact = ChartArtifact(title=title, x="intent", y="count", rows=rows)
        else:
            rows = context.repository.top_intents(limit=limit)
            artifact = ChartArtifact(title="Top intents", x="intent", y="count", rows=rows)
        return ChartSummaryResult(artifact=artifact)

    return ToolSpec(
        name="chart_summary",
        version="1.0.0",
        description=(
            "Return chart metadata and rows for visual answers. Use this when the user asks for "
            "a chart, graph, visual breakdown, or image-like dataset visualization."
        ),
        args_schema=ChartSummaryArgs,
        output_schema=ChartSummaryResult,
        callable=chart_summary,
        examples=[
            ToolExample(
                input={"chart_kind": "category_distribution"},
                output_summary="Bar chart artifact for rows by category.",
            )
        ],
        return_summary="Chart title, axes, type, and rows.",
    )
