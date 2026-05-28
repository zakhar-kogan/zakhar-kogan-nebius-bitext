"""LLM usage logging tests."""

from bitext_agent.config import Settings
from bitext_agent.llm import build_chat_model, invoke_with_usage_log
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


class FakeNestedNebiusResult:
    """Fake response where provider usage is nested under additional kwargs."""

    additional_kwargs = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "{}",
                },
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            }
        ]
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


def test_nested_nebius_usage_metadata_is_logged(tmp_path) -> None:
    store = SettingsStore(tmp_path / "app.sqlite")

    invoke_with_usage_log(
        store=store,
        model_name="nvidia/Nemotron-3-Nano-Omni",
        session_id="s",
        user_uuid="u",
        call=lambda: FakeNestedNebiusResult(),
    )
    usage = store.recent_usage(session_id="s", user_uuid="u", limit=1)[0]

    assert usage["prompt_tokens"] == 7
    assert usage["completion_tokens"] == 3
    assert usage["total_tokens"] == 10


def test_chat_model_requests_stream_usage() -> None:
    model = build_chat_model(Settings(nebius_api_key="key"), "nvidia/Nemotron-3-Nano-Omni")

    assert model.stream_usage is True
