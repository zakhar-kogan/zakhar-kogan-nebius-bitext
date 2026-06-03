from __future__ import annotations

import streamlit as st

from bitext_agent.graph import AgentService
from bitext_agent.schemas import AgentEvent, AgentResponse, ChartArtifact
from bitext_agent.streamlit_app import (
    _chart_rows,
    _distill_on_conversation_boundary,
    _load_visual_artifacts,
    _load_session_messages,
    _live_reasoning_status,
    _reasoning_label,
    _run_prompt,
    _select_recommendation,
)


def test_live_reasoning_status_stays_compact() -> None:
    status = _live_reasoning_status(
        [
            AgentEvent(kind="route", title="Router", detail="Classified as structured."),
            AgentEvent(kind="tool", title="Tool call: search_rows", detail="{'category': 'REFUND'}"),
            AgentEvent(kind="final", title="Final", detail="Done."),
        ]
    )

    assert status == "Reasoning: 🛠️ Tool call: search_rows - {'category': 'REFUND'}"


def test_reasoning_label_adds_kind_icon() -> None:
    assert _reasoning_label("route", "Router") == "🧭 Router"
    assert _reasoning_label("observation", "Observation") == "👁️ Observation"


def test_new_chat_boundary_distills_per_conversation_memory(test_settings) -> None:
    settings = test_settings.model_copy(update={"memory_distillation_mode": "per_conversation"})
    service = AgentService(settings)
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.add_turn("chat-1", user_uuid, "user", "My name is Alice.")

    saved = _distill_on_conversation_boundary(service, "chat-1", "demo")

    assert saved == ["User's name is Alice"]


def test_select_recommendation_records_rotation_and_clears_pending(test_settings) -> None:
    service = AgentService(test_settings)
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.set_pending_recommendation("chat-1", user_uuid, "query", "reason")

    _select_recommendation(service, "chat-1", "Query")

    assert service.store.get_pending_recommendation("chat-1") is None
    assert service.store.list_selected_recommendation_keys("chat-1") == {"query"}


def test_load_session_messages_hydrates_saved_turns(test_settings) -> None:
    service = AgentService(test_settings)
    user_uuid, _ = service.store.get_or_create_user("demo")
    other_uuid, _ = service.store.get_or_create_user("other")
    service.store.add_turn("chat-1", user_uuid, "user", "How many refund requests?")
    service.store.add_turn(
        "chat-1",
        user_uuid,
        "assistant",
        "REFUND: 2992 rows.",
        {"route": "structured", "suggested_query": None},
    )
    service.store.add_turn("chat-1", other_uuid, "user", "Do not show this.")

    messages = _load_session_messages(service, "chat-1", user_uuid)

    assert messages[0] == {"role": "user", "content": "How many refund requests?"}
    assert messages[1]["role"] == "assistant"
    assert messages[1]["response"].answer == "REFUND: 2992 rows."
    assert len(messages) == 2


def test_load_session_messages_hydrates_visual_artifacts(test_settings) -> None:
    service = AgentService(test_settings)
    user_uuid, _ = service.store.get_or_create_user("demo")
    artifact = ChartArtifact(
        title="Rows by category",
        x="category",
        y="count",
        rows=[{"category": "REFUND", "count": 2}],
    )
    service.store.add_turn(
        "chat-1",
        user_uuid,
        "assistant",
        "Here is the chart.",
        {"route": "structured", "visual_artifacts": [artifact.model_dump(mode="json")]},
    )

    messages = _load_session_messages(service, "chat-1", user_uuid)

    assert messages[0]["response"].visual_artifacts == [artifact]


def test_chart_rows_filters_malformed_rows() -> None:
    artifact = ChartArtifact(
        title="Rows by category",
        x="category",
        y="count",
        rows=[{"category": "REFUND", "count": 2}, {"category": "SHIPPING"}],
    )

    assert _chart_rows(artifact) == [{"category": "REFUND", "count": 2}]


def test_load_visual_artifacts_ignores_missing_metadata() -> None:
    assert _load_visual_artifacts(None) == []


def test_agent_response_accepts_stale_visual_artifact_model() -> None:
    class StaleArtifact:
        def model_dump(self, mode="python"):
            return {
                "title": "Rows by category",
                "chart_type": "bar",
                "x": "category",
                "y": "count",
                "rows": [{"category": "REFUND", "count": 2}],
            }

    current = ChartArtifact(
        title="Rows by category",
        x="category",
        y="count",
        rows=[{"category": "REFUND", "count": 2}],
    )

    assert AgentResponse(answer="ok", route="structured", visual_artifacts=[current]).visual_artifacts == [
        current
    ]
    response = AgentResponse(answer="ok", route="structured", visual_artifacts=[StaleArtifact()])
    assert response.visual_artifacts == [current]


def test_run_prompt_resets_request_state_on_error() -> None:
    class FailingService:
        def stream_turn(self, prompt, session_id, user_id):
            raise RuntimeError("boom")

    key = "messages:test-error:demo"
    st.session_state[key] = []
    try:
        _run_prompt(FailingService(), "fail", "test-error", "demo", key)
    except RuntimeError:
        pass

    assert st.session_state["request_running"] is False
    assert st.session_state["stop_requested"] is False
    assert "request_started_at" not in st.session_state


def test_new_chat_boundary_skips_non_conversation_memory_modes(test_settings) -> None:
    settings = test_settings.model_copy(update={"memory_distillation_mode": "every_n_turns"})
    service = AgentService(settings)
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.add_turn("chat-1", user_uuid, "user", "My name is Alice.")

    saved = _distill_on_conversation_boundary(service, "chat-1", "demo")

    assert saved == []
    assert service.store.list_profile_facts(user_uuid) == []
