"""Tool registry adapters for LangChain, FastMCP, and direct runner use."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from bitext_agent.data import DatasetRepository
from bitext_agent.schemas import ToolSpec
from bitext_agent.settings_store import SettingsStore
from bitext_agent.tool_specs import ToolRuntimeContext, all_tool_specs


class ToolRegistry:
    """Runtime registry around versioned Python tool specs."""

    def __init__(self, repository: DatasetRepository, store: SettingsStore, session_id: str, user_uuid: str):
        self.context = ToolRuntimeContext(
            repository=repository,
            store=store,
            session_id=session_id,
            user_uuid=user_uuid,
        )
        self.specs = {spec.name: spec for spec in all_tool_specs(self.context)}

    def call(self, name: str, **kwargs: Any) -> BaseModel:
        """Validate arguments and call a named tool."""

        spec = self.specs[name]
        args = spec.args_schema(**kwargs)
        return spec.callable(**args.model_dump())

    def langchain_tools(self) -> list[StructuredTool]:
        """Build LangChain StructuredTool objects from versioned Python specs."""

        return [self._to_structured_tool(spec) for spec in self.specs.values()]

    def public_specs(self) -> list[ToolSpec]:
        """Return all active specs for docs and diagnostics."""

        return list(self.specs.values())

    def _to_structured_tool(self, spec: ToolSpec) -> StructuredTool:
        def invoke(**kwargs: Any) -> str:
            result = self.call(spec.name, **kwargs)
            return result.model_dump_json(indent=2)

        return StructuredTool.from_function(
            func=invoke,
            name=spec.name,
            description=f"{spec.description}\nReturns: {spec.return_summary}",
            args_schema=spec.args_schema,
        )

