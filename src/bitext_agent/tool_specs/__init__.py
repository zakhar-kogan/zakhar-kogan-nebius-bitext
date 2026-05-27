"""Versioned Python tool specs and registry."""

from __future__ import annotations

from dataclasses import dataclass

from bitext_agent.data import DatasetRepository
from bitext_agent.schemas import ToolSpec
from bitext_agent.settings_store import SettingsStore


@dataclass(frozen=True)
class ToolRuntimeContext:
    """Runtime dependencies injected into tool callables."""

    repository: DatasetRepository
    store: SettingsStore
    session_id: str = "demo"
    user_uuid: str = "demo"


def all_tool_specs(context: ToolRuntimeContext) -> list[ToolSpec]:
    """Return all active versioned tool specs for the current runtime context."""

    from bitext_agent.tool_specs.count_rows import build_spec as count_rows
    from bitext_agent.tool_specs.get_user_profile import build_spec as get_user_profile
    from bitext_agent.tool_specs.intent_distribution import build_spec as intent_distribution
    from bitext_agent.tool_specs.list_categories import build_spec as list_categories
    from bitext_agent.tool_specs.list_intents import build_spec as list_intents
    from bitext_agent.tool_specs.recommend_next_query import build_spec as recommend_next_query
    from bitext_agent.tool_specs.search_rows import build_spec as search_rows
    from bitext_agent.tool_specs.show_examples import build_spec as show_examples
    from bitext_agent.tool_specs.summarize_records import build_spec as summarize_records

    specs = [
        list_categories(context),
        list_intents(context),
        search_rows(context),
        count_rows(context),
        show_examples(context),
        intent_distribution(context),
        summarize_records(context),
        get_user_profile(context),
        recommend_next_query(context),
    ]
    return [spec for spec in specs if spec.status == "active"]

