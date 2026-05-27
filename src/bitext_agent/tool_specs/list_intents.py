"""Versioned `list_intents` tool spec and implementation."""

from pydantic import BaseModel, Field

from bitext_agent.schemas import IntentList, ToolExample, ToolSpec
from bitext_agent.tool_specs import ToolRuntimeContext


class ListIntentsArgs(BaseModel):
    """Arguments for listing intents."""

    category: str | None = Field(default=None, description="Optional category filter.")


def build_spec(context: ToolRuntimeContext) -> ToolSpec:
    """Build the list_intents tool spec."""

    def list_intents(category: str | None = None) -> IntentList:
        return IntentList(intents=context.repository.list_intents(category=category))

    return ToolSpec(
        name="list_intents",
        version="1.0.0",
        description="List dataset intents, optionally filtered to a support category.",
        args_schema=ListIntentsArgs,
        output_schema=IntentList,
        callable=list_intents,
        examples=[ToolExample(input={"category": "ACCOUNT"}, output_summary="Returns ACCOUNT intents.")],
        return_summary="Sorted list of intent labels.",
    )

