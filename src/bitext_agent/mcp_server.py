"""FastMCP server exposing selected dataset tools."""

from __future__ import annotations

from fastmcp import FastMCP

from bitext_agent.config import get_settings
from bitext_agent.data import DatasetRepository
from bitext_agent.settings_store import SettingsStore
from bitext_agent.tools import ToolRegistry


settings = get_settings()
repository = DatasetRepository(settings.dataset_path)
store = SettingsStore(settings.app_db_path)
registry = ToolRegistry(repository, store, session_id="mcp", user_uuid="mcp")
mcp = FastMCP("bitext-customer-support")


@mcp.tool
def list_categories() -> dict:
    """List all top-level customer support categories in the Bitext dataset."""

    return registry.call("list_categories").model_dump()


@mcp.tool
def count_rows(category: str | None = None, intent: str | None = None) -> dict:
    """Count dataset rows matching a category or intent."""

    return registry.call("count_rows", category=category, intent=intent).model_dump()


@mcp.tool
def category_distribution() -> dict:
    """Return row counts by top-level support category."""

    return registry.call("category_distribution").model_dump()


@mcp.tool
def show_examples(
    category: str | None = None,
    intent: str | None = None,
    n: int = 3,
    offset: int = 0,
) -> dict:
    """Show example customer instructions and agent responses."""

    return registry.call(
        "show_examples", category=category, intent=intent, n=n, offset=offset
    ).model_dump()


@mcp.tool
def intent_distribution(category: str | None = None) -> dict:
    """Return intent counts for a category or for the full dataset."""

    return registry.call("intent_distribution", category=category).model_dump()
