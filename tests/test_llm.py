"""LLM usage logging tests."""

from bitext_agent.llm import invoke_with_usage_log
from bitext_agent.settings_store import SettingsStore


class FakeNebiusResult:
    """Fake OpenAI-compatible response shape exposed through LangChain metadata."""

    response_metadata = {
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 5,
            "total_tokens": 16,
        }
    }


def test_nebius_style_usage_metadata_is_logged(tmp_path) -> None:
    store = SettingsStore(tmp_path / "app.sqlite")

    result = invoke_with_usage_log(
        store=store,
        model_name="nvidia/Nemotron-3-Nano-Omni",
        session_id="s",
        user_uuid="u",
        call=lambda: FakeNebiusResult(),
    )
    usage = store.recent_usage(session_id="s", user_uuid="u", limit=1)[0]

    assert isinstance(result, FakeNebiusResult)
    assert usage["prompt_tokens"] == 11
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 16
    assert "prompt_tokens" in usage["raw_usage_metadata_json"]
