"""Versioned `list_categories` tool spec and implementation."""

from pydantic import BaseModel

from bitext_agent.schemas import CategoryList, ToolExample, ToolSpec
from bitext_agent.tool_specs import ToolRuntimeContext


class ListCategoriesArgs(BaseModel):
    """No arguments; returns all dataset categories."""


def build_spec(context: ToolRuntimeContext) -> ToolSpec:
    """Build the list_categories tool spec."""

    def list_categories() -> CategoryList:
        return CategoryList(categories=context.repository.list_categories())

    return ToolSpec(
        name="list_categories",
        version="1.0.0",
        description="List all top-level customer support categories in the Bitext dataset.",
        args_schema=ListCategoriesArgs,
        output_schema=CategoryList,
        callable=list_categories,
        examples=[ToolExample(input={}, output_summary="Returns categories such as ACCOUNT or REFUND.")],
        return_summary="Sorted list of category labels.",
    )

