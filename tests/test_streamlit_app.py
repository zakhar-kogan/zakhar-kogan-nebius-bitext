from __future__ import annotations

from bitext_agent.graph import AgentService
from bitext_agent.schemas import AgentEvent
from bitext_agent.streamlit_app import (
    _distill_on_conversation_boundary,
    _load_session_messages,
    _live_reasoning_status,
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

    assert status == "Reasoning: Tool call: search_rows - {'category': 'REFUND'}"


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


def test_new_chat_boundary_skips_non_conversation_memory_modes(test_settings) -> None:
    settings = test_settings.model_copy(update={"memory_distillation_mode": "every_n_turns"})
    service = AgentService(settings)
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.add_turn("chat-1", user_uuid, "user", "My name is Alice.")

    saved = _distill_on_conversation_boundary(service, "chat-1", "demo")

    assert saved == []
    assert service.store.list_profile_facts(user_uuid) == []
