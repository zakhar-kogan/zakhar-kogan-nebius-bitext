from __future__ import annotations

from bitext_agent.graph import AgentService
from bitext_agent.schemas import AgentEvent
from bitext_agent.streamlit_app import _distill_on_conversation_boundary, _live_reasoning_status


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


def test_new_chat_boundary_skips_non_conversation_memory_modes(test_settings) -> None:
    settings = test_settings.model_copy(update={"memory_distillation_mode": "every_n_turns"})
    service = AgentService(settings)
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.add_turn("chat-1", user_uuid, "user", "My name is Alice.")

    saved = _distill_on_conversation_boundary(service, "chat-1", "demo")

    assert saved == []
    assert service.store.list_profile_facts(user_uuid) == []
