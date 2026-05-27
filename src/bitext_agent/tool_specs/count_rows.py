"""Versioned `count_rows` tool spec and implementation."""

from pydantic import BaseModel, Field

from bitext_agent.schemas import CountRowsResult, ToolExample, ToolSpec
from bitext_agent.tool_specs import ToolRuntimeContext


class CountRowsArgs(BaseModel):
    """Arguments for counting rows."""

    category: str | None = Field(default=None, description="Optional exact category filter.")
    intent: str | None = Field(default=None, description="Optional exact intent filter.")
    search_id: str | None = Field(default=None, description="Optional previous search ID to count.")


def build_spec(context: ToolRuntimeContext) -> ToolSpec:
    """Build the count_rows tool spec."""

    def count_rows(
        category: str | None = None, intent: str | None = None, search_id: str | None = None
    ) -> CountRowsResult:
        count, filters = context.repository.count_rows(category=category, intent=intent, search_id=search_id)
        return CountRowsResult(count=count, filters=filters)

    return ToolSpec(
        name="count_rows",
        version="1.0.0",
        description=(
            "Count dataset rows matching an exact category, exact intent, or previous search result. "
            "If no filters or search_id are provided, this returns the total dataset row count; "
            "do not use an unfiltered count for vague concepts such as angry customers. "
            "For natural-language concepts, call search_rows first and then count_rows with that search_id. "
            "For questions like 'How many refund requests did we get?', count category='REFUND'."
        ),
        args_schema=CountRowsArgs,
        output_schema=CountRowsResult,
        callable=count_rows,
        examples=[ToolExample(input={"category": "REFUND"}, output_summary="Number of REFUND rows.")],
        return_summary="Count and filters used.",
    )
