"""Versioned `intent_distribution` tool spec and implementation."""

from pydantic import BaseModel, Field

from bitext_agent.schemas import IntentDistributionResult, ToolExample, ToolSpec
from bitext_agent.tool_specs import ToolRuntimeContext


class IntentDistributionArgs(BaseModel):
    """Arguments for intent distributions."""

    category: str | None = Field(default=None, description="Optional category filter.")


def build_spec(context: ToolRuntimeContext) -> ToolSpec:
    """Build the intent_distribution tool spec."""

    def intent_distribution(category: str | None = None) -> IntentDistributionResult:
        return IntentDistributionResult(distribution=context.repository.intent_distribution(category=category))

    return ToolSpec(
        name="intent_distribution",
        version="1.0.0",
        description="Count rows by intent for a category or for the full dataset.",
        args_schema=IntentDistributionArgs,
        output_schema=IntentDistributionResult,
        callable=intent_distribution,
        examples=[ToolExample(input={"category": "ACCOUNT"}, output_summary="Intent counts.")],
        return_summary="List of intent/count pairs.",
    )

