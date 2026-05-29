"""Streamlit chat application for the Bitext agent."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import streamlit as st

from bitext_agent.graph import AgentService
from bitext_agent.schemas import AgentEvent, AgentResponse


IDLE_RECOMMENDATION_DELAY = timedelta(minutes=2)


@st.cache_resource
def get_service() -> AgentService:
    """Create one AgentService per Streamlit process."""

    return AgentService()


def new_session_id() -> str:
    """Return a readable unique session ID for a new chat."""

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"chat-{stamp}-{uuid.uuid4().hex[:6]}"


def main() -> None:
    """Render the Streamlit chat app."""

    st.set_page_config(page_title="Bitext Data Analyst", layout="wide")
    service = get_service()
    st.title("Bitext Data Analyst")
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = "demo"
    st.session_state.setdefault("last_user_activity_at", datetime.now(UTC))
    st.session_state.setdefault("request_running", False)
    st.session_state.setdefault("stop_requested", False)
    st.session_state.setdefault("idle_recommendations_visible", False)
    _idle_tick()

    with st.sidebar:
        user_id = st.text_input("User ID", value="demo")
        user_uuid, _ = service.store.get_or_create_user(user_id)
        st.caption(f"User UUID: `{user_uuid}`")
        sessions = service.store.list_user_sessions(user_uuid)
        if st.button("New chat"):
            saved = _distill_on_conversation_boundary(
                service, st.session_state["session_id"], user_id
            )
            if saved:
                st.session_state["memory_notice"] = (
                    f"Saved {len(saved)} profile fact(s) from the previous chat."
                )
            st.session_state["session_id"] = new_session_id()
            st.session_state["idle_recommendations_visible"] = False
            st.rerun()
        if notice := st.session_state.pop("memory_notice", None):
            st.success(notice)
        session_options = [item["session_id"] for item in sessions]
        active_session = st.session_state["session_id"]
        if session_options:
            if active_session not in session_options:
                session_options = [active_session, *session_options]
            selected_session = st.selectbox(
                "Saved sessions",
                options=session_options,
                index=session_options.index(active_session),
                format_func=lambda value: _session_label(value, sessions),
            )
            st.session_state["session_id"] = selected_session
        session_id = st.text_input("Session ID", value=st.session_state["session_id"])
        st.session_state["session_id"] = session_id
        if st.button("Save session memory"):
            saved = service.distill_session(session_id, user_id)
            st.success(f"Saved {len(saved)} profile facts.")
        summary = service.store.get_session_summary(session_id)
        if summary:
            with st.expander("Compacted session context"):
                st.caption(f"Cached from {summary['source_turn_count']} older turns.")
                st.write(summary["summary"])
        st.subheader("Dataset")
        st.json(service.repository.dataset_status())
        st.subheader("Usage")
        st.caption("Current session")
        st.json(service.store.usage_summary(session_id=session_id, user_uuid=user_uuid))
        st.json(service.store.tool_usage_summary(session_id=session_id, user_uuid=user_uuid))
        with st.expander("Usage by session"):
            st.dataframe(service.store.usage_by_session(user_uuid=user_uuid), hide_index=True)
        with st.expander("Recent LLM calls"):
            st.dataframe(service.store.recent_usage(session_id=session_id, user_uuid=user_uuid), hide_index=True)
        with st.expander("Tool calls by name"):
            st.dataframe(service.store.tool_calls_by_name(session_id=session_id, user_uuid=user_uuid), hide_index=True)
        st.subheader("Memory")
        facts = service.store.list_profile_facts(user_uuid)
        for fact in facts:
            cols = st.columns([4, 1])
            cols[0].caption(fact.fact)
            if cols[1].button(
                "🗑️", key=f"delete-{fact.id}", help=f"Delete memory: {fact.fact}"
            ):
                service.store.delete_profile_fact(fact.id or 0, user_uuid)
                st.rerun()

    key = f"messages:{session_id}:{user_id}"
    st.session_state["active_message_key"] = key
    if key not in st.session_state:
        st.session_state[key] = []

    for item in st.session_state[key]:
        if item["role"] == "user":
            with st.chat_message("user"):
                st.write(item["content"])
        else:
            response = item["response"]
            with st.chat_message("assistant"):
                st.write(response.answer)
                with st.expander("Reasoning"):
                    for step in response.reasoning:
                        st.markdown(f"**{step.title}**: {step.detail}")

    recommendation_label = "Recommended queries" if st.session_state[key] else "Starter queries"
    if _idle_ready(key):
        recommendation_label = "You might ask next"
    if not st.session_state["request_running"]:
        _render_recommendations(service, session_id, user_id, key, recommendation_label)

    prompt = st.chat_input(
        "Ask about Bitext support categories, intents, counts, examples, or summaries",
        disabled=st.session_state["request_running"],
    )
    if prompt:
        _run_prompt(service, prompt, session_id, user_id, key)
        st.rerun()


def _session_label(session_id: str, sessions: list[dict[str, object]]) -> str:
    for item in sessions:
        if item["session_id"] == session_id:
            return f"{session_id} ({item['user_turns']} user turns, last {item['last_turn']})"
    return f"{session_id} (new/manual)"


def _render_recommendations(
    service: AgentService, session_id: str, user_id: str, message_key: str, label: str
) -> None:
    if hasattr(service, "recommend_queries"):
        queries = service.recommend_queries(session_id, user_id, limit=2)
    else:
        st.cache_resource.clear()
        st.rerun()
        return
    if not queries:
        return
    st.subheader(label)
    cols = st.columns(len(queries))
    for index, query in enumerate(queries):
        if cols[index].button(query, key=f"recommendation:{message_key}:{index}:{query}"):
            _select_recommendation(service, session_id, query)
            _run_prompt(service, query, session_id, user_id, message_key)
            st.rerun()


def _select_recommendation(service: AgentService, session_id: str, query: str) -> None:
    service.store.record_selected_recommendation(session_id, query)
    service.store.clear_pending_recommendation(session_id)


def _distill_on_conversation_boundary(
    service: AgentService, session_id: str, user_id: str
) -> list[str]:
    if service.settings.memory_distillation_mode != "per_conversation":
        return []
    user_uuid, _ = service.store.get_or_create_user(user_id)
    if service.store.count_user_turns(session_id, user_uuid) == 0:
        return []
    return service.distill_session(session_id, user_id)


def _run_prompt(
    service: AgentService, prompt: str, session_id: str, user_id: str, message_key: str
) -> None:
    st.session_state["last_user_activity_at"] = datetime.now(UTC)
    st.session_state["idle_recommendations_visible"] = False
    st.session_state["request_running"] = True
    st.session_state["stop_requested"] = False
    st.session_state[message_key].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        response = _render_streaming_response(service, prompt, session_id, user_id)
    st.session_state[message_key].append({"role": "assistant", "response": response})
    st.session_state["request_running"] = False
    st.session_state["stop_requested"] = False


def _render_streaming_response(
    service: AgentService, prompt: str, session_id: str, user_id: str
) -> AgentResponse:
    events: list[AgentEvent] = []
    final_response: AgentResponse | None = None
    answer_box = st.empty()
    trace_box = st.empty()

    for event in service.stream_turn(prompt, session_id, user_id):
        events.append(event)
        if event.final_response:
            final_response = event.final_response
        if event.answer_delta:
            answer_box.write(event.answer_delta)
        if status := _live_reasoning_status(events):
            trace_box.caption(status)

    if final_response is None:
        final_response = AgentResponse(
            answer="I could not complete the request.",
            route="structured",
            reasoning=[],
        )
        answer_box.write(final_response.answer)
    return final_response


def _live_reasoning_status(events: list[AgentEvent]) -> str:
    for event in reversed(_dedupe_events(events)):
        if event.kind == "final":
            continue
        if event.detail:
            return f"Reasoning: {event.title} - {_shorten(event.detail)}"
        return f"Reasoning: {event.title}"
    return ""


def _shorten(value: str, limit: int = 160) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


@st.fragment(run_every="10s")
def _idle_tick() -> None:
    if _idle_ready() and not st.session_state.get("idle_recommendations_visible"):
        st.session_state["idle_recommendations_visible"] = True
        st.rerun()


def _idle_ready(message_key: str | None = None) -> bool:
    if message_key is None:
        message_key = st.session_state.get("active_message_key")
    has_messages = True if message_key is None else bool(st.session_state.get(message_key))
    return (
        datetime.now(UTC) - st.session_state["last_user_activity_at"] >= IDLE_RECOMMENDATION_DELAY
        and has_messages
        and not st.session_state["request_running"]
    )


def _dedupe_events(events: list[AgentEvent]) -> list[AgentEvent]:
    result: list[AgentEvent] = []
    seen: set[tuple[str, str, str]] = set()
    for event in events:
        key = (event.kind, event.title, event.detail)
        if key in seen:
            continue
        seen.add(key)
        result.append(event)
    return result


if __name__ == "__main__":
    main()
