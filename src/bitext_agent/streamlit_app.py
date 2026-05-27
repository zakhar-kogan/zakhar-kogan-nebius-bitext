"""Streamlit chat application for the Bitext agent."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import streamlit as st

from bitext_agent.graph import AgentService


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

    with st.sidebar:
        user_id = st.text_input("User ID", value="demo")
        user_uuid, _ = service.store.get_or_create_user(user_id)
        st.caption(f"User UUID: `{user_uuid}`")
        sessions = service.store.list_user_sessions(user_uuid)
        if st.button("New chat"):
            st.session_state["session_id"] = new_session_id()
            st.rerun()
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
            if cols[1].button("Delete", key=f"delete-{fact.id}"):
                service.store.delete_profile_fact(fact.id or 0, user_uuid)
                st.rerun()

    key = f"messages:{session_id}:{user_id}"
    if key not in st.session_state:
        st.session_state[key] = []
    if not st.session_state[key]:
        st.subheader("Starter queries")
        for query in service.starter_recommendations():
            if st.button(query, key=f"starter-{query}"):
                st.session_state[key].append({"role": "user", "content": query})
                response = service.run_turn(query, session_id, user_id)
                st.session_state[key].append({"role": "assistant", "response": response})
                st.rerun()

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

    prompt = st.chat_input("Ask about Bitext support categories, intents, counts, examples, or summaries")
    if prompt:
        st.session_state[key].append({"role": "user", "content": prompt})
        response = service.run_turn(prompt, session_id, user_id)
        st.session_state[key].append({"role": "assistant", "response": response})
        st.rerun()


def _session_label(session_id: str, sessions: list[dict[str, object]]) -> str:
    for item in sessions:
        if item["session_id"] == session_id:
            return f"{session_id} ({item['user_turns']} user turns, last {item['last_turn']})"
    return f"{session_id} (new/manual)"


if __name__ == "__main__":
    main()
