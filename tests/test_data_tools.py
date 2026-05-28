"""Dataset repository and tool tests."""

from pydantic import BaseModel

from bitext_agent.data import DatasetRepository, normalize_label
from bitext_agent.settings_store import SettingsStore
from bitext_agent.tools import ToolRegistry
from bitext_agent.tool_specs import ToolRuntimeContext
from bitext_agent.tool_specs.search_rows import build_spec as build_search_rows_spec
from bitext_agent.tool_specs.show_examples import build_spec as build_show_examples_spec


class ForeignDatasetRow(BaseModel):
    row_id: int
    category: str
    intent: str
    instruction: str
    response: str


class ForeignRowRepository:
    row = ForeignDatasetRow(
        row_id=1,
        category="REFUND",
        intent="check_refund_policy",
        instruction="Where is the refund policy?",
        response="Open the refund policy page.",
    )

    def search_rows(self, category=None, intent=None, query=None, fuzzy=True, limit=20):
        return "search-1", 1, [self.row]

    def show_examples(self, category=None, intent=None, search_id=None, n=3, offset=0):
        return [self.row], None, 1


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


def test_row_tools_validate_serialized_rows_from_foreign_model(tmp_path) -> None:
    store = SettingsStore(tmp_path / "app.sqlite")
    context = ToolRuntimeContext(ForeignRowRepository(), store, session_id="s", user_uuid="u")

    search_result = build_search_rows_spec(context).callable(category="REFUND", limit=1)
    examples_result = build_show_examples_spec(context).callable(category="REFUND", n=1)

    assert search_result.rows[0].category == "REFUND"
    assert examples_result.rows[0].intent == "check_refund_policy"
