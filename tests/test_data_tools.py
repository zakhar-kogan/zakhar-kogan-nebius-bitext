"""Dataset repository and tool tests."""

from bitext_agent.data import DatasetRepository, normalize_label
from bitext_agent.settings_store import SettingsStore
from bitext_agent.tools import ToolRegistry


def test_dataset_schema_and_categories(sample_dataset) -> None:
    repo = DatasetRepository(sample_dataset)
    assert repo.load().height == 5
    assert repo.list_categories() == ["ACCOUNT", "COMPLAINT", "REFUND", "SHIPPING"]


def test_normalize_label() -> None:
    assert normalize_label("Money Back!") == "money_back"


def test_fuzzy_money_back_search(sample_dataset) -> None:
    repo = DatasetRepository(sample_dataset)
    search_id, total, rows = repo.search_rows(query="money back", limit=10)
    assert search_id
    assert total >= 1
    assert rows[0].category == "REFUND"


def test_tool_specs_and_registration(tmp_path, sample_dataset) -> None:
    repo = DatasetRepository(sample_dataset)
    store = SettingsStore(tmp_path / "app.sqlite")
    registry = ToolRegistry(repo, store, session_id="s", user_uuid="u")
    names = {spec.name for spec in registry.public_specs()}
    assert "count_rows" in names
    assert "show_examples" in names
    result = registry.call("count_rows", category="REFUND")
    assert result.count == 2

