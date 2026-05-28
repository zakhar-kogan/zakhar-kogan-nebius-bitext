"""Versioned `search_rows` tool spec and implementation."""

from pydantic import BaseModel, Field

from bitext_agent.schemas import SearchRowsResult, ToolExample, ToolSpec
from bitext_agent.tool_specs import ToolRuntimeContext


class SearchRowsArgs(BaseModel):
    """Arguments for searching rows."""

    category: str | None = Field(default=None, description="Optional exact category filter.")
    intent: str | None = Field(default=None, description="Optional exact intent filter.")
    query: str | None = Field(default=None, description="Optional phrase to search in instructions/responses.")
    fuzzy: bool = Field(default=True, description="Whether to use fuzzy text matching.")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum rows to return.")


def build_spec(context: ToolRuntimeContext) -> ToolSpec:
    """Build the search_rows tool spec."""

    def search_rows(
        category: str | None = None,
        intent: str | None = None,
        query: str | None = None,
        fuzzy: bool = True,
        limit: int = 20,
    ) -> SearchRowsResult:
        search_id, total, rows = context.repository.search_rows(category, intent, query, fuzzy, limit)
        return SearchRowsResult(
            search_id=search_id,
            total_matches=total,
            rows=[row.model_dump(mode="json") for row in rows],
        )

    return ToolSpec(
        name="search_rows",
        version="1.0.0",
        description="Search dataset records by category, intent, or natural phrase such as 'money back'.",
        args_schema=SearchRowsArgs,
        output_schema=SearchRowsResult,
        callable=search_rows,
        examples=[ToolExample(input={"query": "money back"}, output_summary="Rows about refunds.")],
        return_summary="Matching rows and a reusable search_id.",
    )
