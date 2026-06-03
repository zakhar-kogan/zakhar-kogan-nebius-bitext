"""Versioned `category_distribution` tool spec and implementation."""

from pydantic import BaseModel

from bitext_agent.schemas import CategoryDistributionResult, ToolExample, ToolSpec
from bitext_agent.tool_specs import ToolRuntimeContext


class CategoryDistributionArgs(BaseModel):
    """Arguments for category distributions."""


def build_spec(context: ToolRuntimeContext) -> ToolSpec:
    """Build the category_distribution tool spec."""

    def category_distribution() -> CategoryDistributionResult:
        return CategoryDistributionResult(distribution=context.repository.category_distribution())

    return ToolSpec(
        name="category_distribution",
        version="1.0.0",
        description="Count rows by top-level support category for charting or tabular answers.",
        args_schema=CategoryDistributionArgs,
        output_schema=CategoryDistributionResult,
        callable=category_distribution,
        examples=[ToolExample(input={}, output_summary="Category/count rows.")],
        return_summary="List of category/count pairs.",
    )
