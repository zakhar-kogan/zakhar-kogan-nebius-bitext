"""Versioned `show_examples` tool spec and implementation."""

from pydantic import BaseModel, Field

from bitext_agent.schemas import ExamplesResult, ToolExample, ToolSpec
from bitext_agent.tool_specs import ToolRuntimeContext


class ShowExamplesArgs(BaseModel):
    """Arguments for showing example records."""

    category: str | None = Field(default=None, description="Optional category filter.")
    intent: str | None = Field(default=None, description="Optional intent filter.")
    search_id: str | None = Field(default=None, description="Optional previous search ID.")
    n: int = Field(default=3, ge=1, le=25, description="Number of examples to return.")
    offset: int = Field(default=0, ge=0, description="Offset for follow-up paging.")


def build_spec(context: ToolRuntimeContext) -> ToolSpec:
    """Build the show_examples tool spec."""

    def show_examples(
        category: str | None = None,
        intent: str | None = None,
        search_id: str | None = None,
        n: int = 3,
        offset: int = 0,
    ) -> ExamplesResult:
        rows, next_offset, total = context.repository.show_examples(category, intent, search_id, n, offset)
        return ExamplesResult(
            rows=[row.model_dump(mode="json") for row in rows],
            offset=offset,
            next_offset=next_offset,
            total_matches=total,
        )

    return ToolSpec(
        name="show_examples",
        version="1.0.0",
        description="Show example customer instructions and responses, with offset paging for 'show more'.",
        args_schema=ShowExamplesArgs,
        output_schema=ExamplesResult,
        callable=show_examples,
        examples=[ToolExample(input={"category": "SHIPPING", "n": 3}, output_summary="3 examples.")],
        return_summary="Example rows and next offset.",
    )
