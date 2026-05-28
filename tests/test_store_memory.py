"""SQLite state and memory tests."""

from bitext_agent.memory import canonical_profile_key, distill_profile_memory, refresh_session_summary
from bitext_agent.settings_store import SettingsStore


def test_user_uuid_mapping(tmp_path) -> None:
    store = SettingsStore(tmp_path / "app.sqlite")
    first, ext = store.get_or_create_user("demo")
    second, _ = store.get_or_create_user("demo")
    assert first == second
    assert ext == "demo"


def test_profile_fact_crud(tmp_path) -> None:
    store = SettingsStore(tmp_path / "app.sqlite")
    user_uuid, _ = store.get_or_create_user("demo")
    fact_id = store.add_profile_fact(user_uuid, "topic", "User likes refund data", "test")
    assert len(store.list_profile_facts(user_uuid)) == 1
    store.delete_profile_fact(fact_id, user_uuid)
    assert store.list_profile_facts(user_uuid) == []


def test_profile_fact_upsert_dedupes_by_canonical_key(tmp_path) -> None:
    store = SettingsStore(tmp_path / "app.sqlite")
    user_uuid, _ = store.get_or_create_user("demo")
    key = canonical_profile_key("format_preference", "User prefers concise answers")

    first_id, created_first = store.upsert_profile_fact(
        user_uuid, "format_preference", "User prefers concise answers", key, "test", confidence=0.6
    )
    second_id, created_second = store.upsert_profile_fact(
        user_uuid, "format_preference", "User prefers concise answers", key, "test", confidence=0.9
    )

    facts = store.list_profile_facts(user_uuid)
    assert first_id == second_id
    assert created_first is True
    assert created_second is False
    assert len(facts) == 1
    assert facts[0].confidence == 0.9


def test_profile_fact_pruning_keeps_newer_stronger_facts(tmp_path) -> None:
    store = SettingsStore(tmp_path / "app.sqlite")
    user_uuid, _ = store.get_or_create_user("demo")
    for index in range(35):
        store.upsert_profile_fact(
            user_uuid,
            "topic_interest",
            f"User is interested in topic {index}",
            f"topic_interest:topic-{index}",
            "test",
            confidence=0.1 + (index / 100),
        )

    pruned = store.prune_profile_facts(user_uuid, max_active=30)
    facts = store.list_profile_facts(user_uuid)

    assert pruned == 5
    assert len(facts) == 30
    assert all("topic 0" not in fact.fact for fact in facts)


def test_prompt_override_resolution(tmp_path) -> None:
    store = SettingsStore(tmp_path / "app.sqlite")
    store.set_prompt_override("router", "override")
    assert store.get_prompt_override("router") == "override"


def test_recommendation_state_and_usage(tmp_path) -> None:
    store = SettingsStore(tmp_path / "app.sqlite")
    user_uuid, _ = store.get_or_create_user("demo")
    store.set_cached_recommendations("k", ["q1"])
    assert store.get_cached_recommendations("k") == ["q1"]
    store.set_pending_recommendation("s", "u", "query", "reason")
    assert store.get_pending_recommendation("s")["query"] == "query"
    store.log_usage(model="m", status="ok", session_id="s", user_uuid=user_uuid, total_tokens=3)
    store.log_usage(model="m", status="ok", session_id="other", user_uuid=user_uuid, total_tokens=7)
    assert store.usage_summary()["tokens"] == 10
    assert store.usage_summary(session_id="s")["tokens"] == 3
    assert store.usage_summary(session_id="other")["tokens"] == 7


def test_session_and_tool_diagnostics(tmp_path) -> None:
    store = SettingsStore(tmp_path / "app.sqlite")
    user_uuid, _ = store.get_or_create_user("demo")
    other_uuid, _ = store.get_or_create_user("other")
    store.add_turn("s1", user_uuid, "user", "hello")
    store.add_turn("s2", user_uuid, "user", "hello again")
    store.add_turn("s3", other_uuid, "user", "not mine")

    sessions = store.list_user_sessions(user_uuid)
    assert {item["session_id"] for item in sessions} == {"s1", "s2"}

    store.log_tool_call("s1", user_uuid, "count_rows", "ok", latency_ms=5)
    store.log_tool_call("s1", user_uuid, "show_examples", "error", latency_ms=2, error="bad")
    summary = store.tool_usage_summary(session_id="s1", user_uuid=user_uuid)
    assert summary["tool_calls"] == 2
    assert summary["tool_errors"] == 1
    assert store.tool_calls_by_name(session_id="s1")[0]["calls"] == 1


def test_distill_profile_memory(tmp_path) -> None:
    store = SettingsStore(tmp_path / "app.sqlite")
    user_uuid, _ = store.get_or_create_user("demo")
    store.add_turn("s", user_uuid, "user", "I am interested in refund data and prefer concise answers")
    saved = distill_profile_memory(store, "s", user_uuid)
    assert saved
    kinds = {fact.kind for fact in store.list_profile_facts(user_uuid)}
    assert "topic_interest" in kinds
    assert "format_preference" in kinds


def test_refresh_session_summary_preserves_full_turns(tmp_path) -> None:
    store = SettingsStore(tmp_path / "app.sqlite")
    user_uuid, _ = store.get_or_create_user("demo")
    for index in range(8):
        store.add_turn("long", user_uuid, "user", f"Question {index}")
        store.add_turn("long", user_uuid, "assistant", f"Answer {index}")

    summary = refresh_session_summary(store, "long", user_uuid, compact_after_turns=8, keep_recent_turns=4)

    assert summary is not None
    assert summary["source_turn_count"] == 12
    assert "Question" in summary["summary"]
    assert store.count_turns("long") == 16
