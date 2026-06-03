"""Agent integration-style tests over a sample dataset."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from bitext_agent.agent_runner import AgentRunner, LlmReActRunner, Router
from bitext_agent.graph import AgentService
from bitext_agent.prompts import PromptStore
from bitext_agent.schemas import (
    AgentResponse,
    ReasoningStep,
    RecommendationRefinementResult,
    RouteKind,
    RouterDecision,
)
from bitext_agent.tools import ToolRegistry


class FakeRouter(Router):
    """Router test double returning a fixed decision."""

    def __init__(self, route: RouteKind, reason: str = "test route") -> None:
        self.decision = RouterDecision(route=route, reason=reason)

    def route(self, message: str, session_id: str, user_uuid: str) -> RouterDecision:
        return self.decision


class FakeRunner(AgentRunner):
    """Runner test double for graph wiring tests."""

    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        message: str,
        route: RouteKind,
        checkpoint: dict[str, Any],
        route_reason: str = "",
    ) -> AgentResponse:
        self.calls += 1
        return AgentResponse(
            answer="runner answer",
            route=route,
            reasoning=[ReasoningStep(kind="route", title="Router", detail=route_reason)],
        )


class FakeRecommendationRefiner:
    """Recommendation refinement test double returning a fixed decision."""

    def __init__(self, result: RecommendationRefinementResult) -> None:
        self.result = result
        self.calls = 0

    def refine(
        self, message: str, pending: dict[str, str], session_id: str, user_uuid: str
    ) -> RecommendationRefinementResult:
        self.calls += 1
        return self.result


class FakeChatModel:
    """Chat model test double returning queued AI messages."""

    def __init__(self, messages: list[AIMessage]) -> None:
        self.messages = messages
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return self.messages.pop(0)


class FakeAngryModel:
    """Model double that searches before counting a vague natural-language concept."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if not tool_messages:
            return AIMessage(
                content="",
                tool_calls=[{"name": "search_rows", "args": {"query": "angry customers", "limit": 100}, "id": "s1"}],
            )
        first_observation = json.loads(tool_messages[0].content)
        if len(tool_messages) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "count_rows",
                        "args": {"search_id": first_observation["search_id"]},
                        "id": "c1",
                    }
                ],
            )
        count_observation = json.loads(tool_messages[1].content)
        return AIMessage(content=f"Found {count_observation['count']} rows matching angry customers.")


class FakeSearchExamplesModel:
    """Model double that searches by phrase before showing examples."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if not tool_messages:
            return AIMessage(
                content="",
                tool_calls=[{"name": "search_rows", "args": {"query": "money back", "limit": 10}, "id": "s1"}],
            )
        first_observation = json.loads(tool_messages[0].content)
        if len(tool_messages) == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "show_examples",
                        "args": {"search_id": first_observation["search_id"], "n": 1},
                        "id": "e1",
                    }
                ],
            )
        return AIMessage(content="Here is the first money-back example.")


def _registry(service: AgentService) -> ToolRegistry:
    user_uuid, _ = service.store.get_or_create_user("demo")
    return ToolRegistry(service.repository, service.store, "s", user_uuid)


def _runner(service: AgentService, model) -> LlmReActRunner:
    return LlmReActRunner(
        registry=_registry(service),
        settings=service.settings,
        store=service.store,
        prompt_store=PromptStore(service.store),
        model=model,
    )


def test_graph_uses_router_and_runner(test_settings) -> None:
    fake_runner = FakeRunner()
    service = AgentService(
        test_settings,
        router=FakeRouter("structured", "dataset count"),
        runner_factory=lambda registry, service, state: fake_runner,
    )

    response = service.run_turn("How many refund requests did we get?", "s", "demo")

    assert response.route == "structured"
    assert response.answer == "runner answer"
    assert fake_runner.calls == 1


def test_out_of_scope_declines_without_runner(test_settings) -> None:
    service = AgentService(
        test_settings,
        router=FakeRouter("out_of_scope", "not about the dataset"),
        runner_factory=lambda registry, service, state: (_ for _ in ()).throw(AssertionError("runner called")),
    )

    response = service.run_turn("Who is the president of France?", "s2", "demo")

    assert response.route == "out_of_scope"
    assert "Bitext customer service dataset" in response.answer


def test_recommendation_sets_pending_query(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("recommendation", "next query"))

    response = service.run_turn("What should I query next?", "rec", "demo")

    assert response.route == "recommendation"
    assert response.suggested_query
    assert "Should I go ahead?" in response.answer
    assert service.store.get_pending_recommendation("rec") is not None


def test_confirmed_recommendation_clears_pending_query(test_settings) -> None:
    service = AgentService(
        test_settings,
        router=FakeRouter("recommendation", "confirmed"),
        runner_factory=lambda registry, service, state: FakeRunner(),
    )
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.set_pending_recommendation(
        "rec-confirm", user_uuid, "How many refund requests did we get?", "test"
    )

    response = service.run_turn("yes", "rec-confirm", "demo")

    assert response.route == "structured"
    assert service.store.get_pending_recommendation("rec-confirm") is None


def test_recommendation_sure_confirms_pending_query(test_settings) -> None:
    fake_runner = FakeRunner()
    service = AgentService(
        test_settings,
        router=FakeRouter("recommendation", "confirmed"),
        runner_factory=lambda registry, service, state: fake_runner,
    )
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.set_pending_recommendation(
        "rec-sure", user_uuid, "How many refund requests did we get?", "test"
    )

    response = service.run_turn("sure", "rec-sure", "demo")

    assert response.route == "structured"
    assert fake_runner.calls == 1
    assert service.store.get_pending_recommendation("rec-sure") is None


def test_recommendation_yes_do_it_confirms_with_punctuation(test_settings) -> None:
    fake_runner = FakeRunner()
    service = AgentService(
        test_settings,
        router=FakeRouter("recommendation", "confirmed"),
        runner_factory=lambda registry, service, state: fake_runner,
    )
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.set_pending_recommendation(
        "rec-yes-do-it", user_uuid, "How many refund requests did we get?", "test"
    )

    response = service.run_turn('"Yes, do it. "', "rec-yes-do-it", "demo")

    assert response.route == "structured"
    assert fake_runner.calls == 1
    assert service.store.get_pending_recommendation("rec-yes-do-it") is None


def test_recommendation_no_clears_pending_without_runner(test_settings) -> None:
    service = AgentService(
        test_settings,
        router=FakeRouter("recommendation", "cancelled"),
        runner_factory=lambda registry, service, state: (_ for _ in ()).throw(AssertionError("runner called")),
    )
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.set_pending_recommendation(
        "rec-cancel", user_uuid, "How many refund requests did we get?", "test"
    )

    response = service.run_turn("no", "rec-cancel", "demo")

    assert response.route == "recommendation"
    assert "cancelled" in response.answer
    assert service.store.get_pending_recommendation("rec-cancel") is None


def test_stream_turn_emits_recommendation_refinement_event(test_settings) -> None:
    refiner = FakeRecommendationRefiner(
        RecommendationRefinementResult(
            refined_query="Show me 5 examples from the REFUND category.",
            reason="User wants examples instead.",
            unclear=False,
        )
    )
    service = AgentService(
        test_settings,
        router=FakeRouter("recommendation", "refine pending"),
        recommendation_refiner=refiner,
    )
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.set_pending_recommendation(
        "rec-refine", user_uuid, "What is the distribution of intents in the REFUND category?", "test"
    )

    events = list(service.stream_turn("I'd rather see examples instead.", "rec-refine", "demo"))

    assert any(event.kind == "recommendation" and event.title == "Refined" for event in events)
    assert events[-1].final_response is not None
    assert "Should I go ahead?" in events[-1].final_response.answer
    assert refiner.calls == 1


def test_unclear_recommendation_reply_does_not_create_new_suggestion(test_settings) -> None:
    refiner = FakeRecommendationRefiner(
        RecommendationRefinementResult(refined_query=None, reason="Request is unclear.", unclear=True)
    )
    service = AgentService(
        test_settings,
        router=FakeRouter("recommendation", "unclear pending"),
        recommendation_refiner=refiner,
    )
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.set_pending_recommendation(
        "rec-unclear", user_uuid, "Show me 5 examples from the REFUND category.", "test"
    )

    response = service.run_turn("hmm", "rec-unclear", "demo")

    assert "I still have this pending suggestion" in response.answer
    assert response.suggested_query == "Show me 5 examples from the REFUND category."
    assert service.store.get_pending_recommendation("rec-unclear")["query"] == response.suggested_query
    assert refiner.calls == 1


def test_recommend_queries_keep_slots_until_clicked(test_settings) -> None:
    service = AgentService(
        test_settings,
        router=FakeRouter("structured"),
        runner_factory=lambda registry, service, state: FakeRunner(),
    )

    before = service.recommend_queries("rec-refresh", "demo", limit=2)
    service.run_turn("How many refund requests did we get?", "rec-refresh", "demo")
    after = service.recommend_queries("rec-refresh", "demo", limit=2)

    assert before == after


def test_recommend_queries_excludes_selected_query_for_session(test_settings) -> None:
    service = AgentService(
        test_settings,
        router=FakeRouter("structured"),
        runner_factory=lambda registry, service, state: FakeRunner(),
    )
    session_id = "rec-rotate"
    selected = "Summarize how agents respond to complaint intents."

    service.run_turn(selected, session_id, "demo")
    service.store.record_selected_recommendation(session_id, selected.upper())
    recommendations = service.recommend_queries(session_id, "demo", limit=2)

    assert selected not in recommendations
    assert recommendations


def test_recommend_queries_uses_starters_without_profile(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))

    recommendations = service.recommend_queries("rec-ui", "demo", limit=2)

    assert recommendations == service.starter_recommendations()[:2]


def test_recommend_queries_include_visual_starter(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))

    recommendations = service.recommend_queries("rec-ui-visual", "demo", limit=3)

    assert "Show a bar chart of the category breakdown." in recommendations


def test_recommend_queries_uses_profile_facts(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.upsert_profile_fact(
        user_uuid,
        "topic_interest",
        "User is interested in refund data",
        "topic_interest:refund",
        "test",
    )

    recommendations = service.recommend_queries("rec-ui-profile", "demo", limit=2)

    assert "REFUND" in " ".join(recommendations)


def test_refund_recommendations_include_visual_chart(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.upsert_profile_fact(
        user_uuid,
        "topic_interest",
        "User is interested in refund data",
        "topic_interest:refund",
        "test",
    )

    recommendations = service.recommend_queries("rec-ui-profile-visual", "demo", limit=3)

    assert "Show a bar chart of intent distribution for REFUND." in recommendations


def test_recommendation_slots_start_with_visible_limit(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))

    slots = service.recommendation_slots("slot-start", "demo", limit=2)

    assert [slot["slot_index"] for slot in slots] == [0, 1]
    assert len(slots) == 2


def test_recommendation_slot_replacement_keeps_other_slot(test_settings) -> None:
    service = AgentService(
        test_settings,
        router=FakeRouter("structured"),
        runner_factory=lambda registry, service, state: FakeRunner(),
    )
    session_id = "slot-replace"
    slots = service.recommendation_slots(session_id, "demo", limit=2)
    clicked = str(slots[0]["query"])
    unchanged = str(slots[1]["query"])

    service.store.record_selected_recommendation(session_id, clicked)
    service.run_turn(clicked, session_id, "demo")
    replacement = service.replace_recommendation_slot(session_id, "demo", 0, limit=2)
    updated = service.recommendation_slots(session_id, "demo", limit=2)

    assert replacement
    assert str(updated[0]["query"]) != clicked
    assert str(updated[1]["query"]) == unchanged


def test_recommendation_replacement_rejects_fuzzy_duplicates(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))
    session_id = "slot-fuzzy"
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.upsert_profile_fact(
        user_uuid,
        "topic_interest",
        "User is interested in refund data",
        "topic_interest:refund",
        "test",
    )
    service.store.record_selected_recommendation(session_id, "Show me 5 examples from REFUND category")

    slots = service.recommendation_slots(session_id, "demo", limit=1)

    assert slots[0]["query"] == "What is the distribution of intents in the REFUND category?"


def test_recommendation_slots_empty_when_fresh_candidates_exhausted(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))
    session_id = "slot-exhausted"
    for query in service.starter_recommendations():
        service.store.record_selected_recommendation(session_id, query)

    slots = service.recommendation_slots(session_id, "demo", limit=2)

    assert slots == []


def test_stream_turn_emits_tool_events_and_final_response(test_settings) -> None:
    service = AgentService(
        test_settings,
        router=FakeRouter("structured"),
        runner_factory=lambda registry, service, state: LlmReActRunner(
            registry=registry,
            settings=service.settings,
            store=service.store,
            prompt_store=PromptStore(service.store),
            model=FakeChatModel(
                [
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "count_rows", "args": {"category": "REFUND"}, "id": "t1"}],
                    ),
                    AIMessage(content="REFUND: 2 rows."),
                ]
            ),
        ),
    )

    events = list(service.stream_turn("How many refund requests did we get?", "stream", "demo"))

    assert any(event.title == "Tool call: count_rows" for event in events)
    assert events[-1].final_response is not None
    assert events[-1].final_response.answer == "REFUND: 2 rows."


def test_stream_turn_can_cancel_before_runner(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured", "dataset count"))
    calls = 0

    def cancel_after_route() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    events = list(service.stream_turn("How many refund requests did we get?", "cancel", "demo", cancel_after_route))

    assert events[-1].kind == "cancelled"
    assert events[-1].final_response is not None
    assert events[-1].final_response.answer == "Stopped before completion."


def test_per_turn_memory_distillation_runs_during_graph(test_settings) -> None:
    settings = test_settings.model_copy(update={"memory_distillation_mode": "per_turn"})
    service = AgentService(
        settings,
        router=FakeRouter("structured", "profile update"),
        runner_factory=lambda registry, service, state: FakeRunner(),
    )

    response = service.run_turn("My name is Alice and I prefer concise answers.", "mem", "demo")
    user_uuid, _ = service.store.get_or_create_user("demo")
    facts = service.store.list_profile_facts(user_uuid)

    assert any(fact.fact == "User's name is Alice" for fact in facts)
    assert any(fact.kind == "format_preference" for fact in facts)
    assert any(step.title == "Memory" for step in response.reasoning)


def test_every_n_turns_memory_distillation_respects_interval(test_settings) -> None:
    settings = test_settings.model_copy(
        update={"memory_distillation_mode": "every_n_turns", "memory_distillation_turn_interval": 2}
    )
    service = AgentService(
        settings,
        router=FakeRouter("structured", "profile update"),
        runner_factory=lambda registry, service, state: FakeRunner(),
    )

    service.run_turn("My name is Alice.", "mem-interval", "demo")
    user_uuid, _ = service.store.get_or_create_user("demo")
    assert service.store.list_profile_facts(user_uuid) == []

    service.run_turn("I prefer concise answers.", "mem-interval", "demo")
    facts = service.store.list_profile_facts(user_uuid)

    assert any("Alice" in fact.fact for fact in facts)
    assert any(fact.kind == "format_preference" for fact in facts)


def test_per_conversation_memory_distillation_preserves_manual_boundary(test_settings) -> None:
    settings = test_settings.model_copy(update={"memory_distillation_mode": "per_conversation"})
    service = AgentService(
        settings,
        router=FakeRouter("structured", "profile update"),
        runner_factory=lambda registry, service, state: FakeRunner(),
    )

    service.run_turn("My name is Alice.", "mem-manual", "demo")
    user_uuid, _ = service.store.get_or_create_user("demo")
    assert service.store.list_profile_facts(user_uuid) == []

    service.distill_session("mem-manual", "demo")
    assert any("Alice" in fact.fact for fact in service.store.list_profile_facts(user_uuid))


def test_react_loop_executes_tool_and_final_answer(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))
    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "count_rows", "args": {"category": "REFUND"}, "id": "t1"}],
            ),
            AIMessage(content="REFUND: 2 rows."),
        ]
    )

    response = _runner(service, model).run("How many refund requests did we get?", "structured", {})

    assert response.answer == "REFUND: 2 rows."
    assert model.calls == 2
    assert any(step.title == "Tool call: count_rows" for step in response.reasoning)


def test_react_loop_attaches_chart_artifact(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))
    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "chart_summary",
                        "args": {"chart_kind": "category_distribution"},
                        "id": "t1",
                    }
                ],
            ),
            AIMessage(content="Here is the category breakdown."),
        ]
    )

    response = _runner(service, model).run("Show a category chart", "structured", {})

    assert response.answer == "Here is the category breakdown."
    assert response.visual_artifacts[0].title == "Rows by category"
    assert response.visual_artifacts[0].rows[0] == {"category": "REFUND", "count": 2}


def test_profile_memory_question_can_use_profile_tool(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))
    user_uuid, _ = service.store.get_or_create_user("demo")
    service.store.upsert_profile_fact(
        user_uuid,
        "format_preference",
        "User prefers concise answers",
        "format_preference:concise-answers",
        "test",
    )
    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_user_profile", "args": {"user_id": "current"}, "id": "p1"}],
            ),
            AIMessage(content="I remember that you prefer concise answers."),
        ]
    )

    response = _runner(service, model).run("What do you remember about me?", "structured", {})

    assert "concise" in response.answer
    assert any(step.title == "Tool call: get_user_profile" for step in response.reasoning)


def test_react_loop_executes_multiple_tool_calls(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))
    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "count_rows", "args": {"category": "REFUND"}, "id": "r"},
                    {"name": "count_rows", "args": {"category": "COMPLAINT"}, "id": "c"},
                ],
            ),
            AIMessage(content="REFUND has 2 rows and COMPLAINT has 1 row."),
        ]
    )

    response = _runner(service, model).run("Compare refunds and complaints.", "structured", {})

    assert "REFUND has 2" in response.answer
    assert len([step for step in response.reasoning if step.title == "Tool call: count_rows"]) == 2


def test_show_more_followup_uses_checkpoint_without_model_guessing(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))
    model = FakeChatModel([AIMessage(content="should not be used")])
    checkpoint = {
        "last_examples": {
            "category": "REFUND",
            "intent": None,
            "search_id": None,
            "n": 1,
            "next_offset": 1,
        }
    }

    response = _runner(service, model).run("Show me 1 more", "structured", checkpoint)

    assert "more examples" in response.answer
    assert checkpoint["last_examples"]["next_offset"] is None
    assert model.calls == 0
    assert any(step.title == "Tool call: show_examples" for step in response.reasoning)


def test_show_more_followup_replays_search_after_restart(test_settings) -> None:
    first_service = AgentService(test_settings, router=FakeRouter("structured"))
    first_model = FakeSearchExamplesModel()
    checkpoint: dict[str, Any] = {}

    first_response = _runner(first_service, first_model).run(
        "Show me examples of people wanting their money back.", "structured", checkpoint
    )

    assert first_response.answer == "Here is the first money-back example."
    assert first_model.calls == 3
    assert checkpoint["last_examples"]["query"] == "money back"
    assert checkpoint["last_examples"]["next_offset"] == 1

    restarted_service = AgentService(test_settings, router=FakeRouter("structured"))
    unused_model = FakeChatModel([AIMessage(content="should not be used")])
    response = _runner(restarted_service, unused_model).run("Show me 1 more", "structured", checkpoint)

    assert "Please refund my order" in response.answer
    assert checkpoint["last_examples"]["next_offset"] is None
    assert unused_model.calls == 0
    assert any(step.title == "Tool call: search_rows" for step in response.reasoning)
    assert any(step.title == "Tool call: show_examples" for step in response.reasoning)


def test_show_more_followup_handles_stale_search_id_without_query(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))
    model = FakeChatModel([AIMessage(content="should not be used")])
    checkpoint = {
        "last_examples": {
            "category": None,
            "intent": None,
            "search_id": "stale-search-id",
            "n": 1,
            "next_offset": 1,
        }
    }

    response = _runner(service, model).run("Show me 1 more", "structured", checkpoint)

    assert "cannot resume" in response.answer
    assert model.calls == 0


def test_count_followup_uses_react_loop_and_tools(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))
    checkpoint: dict[str, Any] = {}

    first_model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "count_rows", "args": {"category": "COMPLAINT"}, "id": "c1"}],
            ),
            AIMessage(content="COMPLAINT has 1 matching row."),
        ]
    )
    first_response = _runner(service, first_model).run(
        "How many complaints did we get?", "structured", checkpoint
    )

    assert first_response.answer == "COMPLAINT has 1 matching row."
    assert checkpoint["last_counts"] == [{"label": "COMPLAINT", "count": 1}]

    model = FakeChatModel(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": "count_rows", "args": {"category": "REFUND"}, "id": "r1"}],
            ),
            AIMessage(content="REFUND has 2 matching rows."),
        ]
    )

    response = _runner(service, model).run("What about refunds?", "structured", checkpoint)

    assert response.answer == "REFUND has 2 matching rows."
    assert checkpoint["last_counts"][-1] == {"label": "REFUND", "count": 2}
    assert model.calls == 2
    assert any(step.title == "Tool call: count_rows" for step in response.reasoning)
    assert any(step.title == "Observation" and "count=2" in step.detail for step in response.reasoning)


def test_last_two_count_total_uses_react_context(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))
    model = FakeChatModel([AIMessage(content="The total count of the last two is 3.")])
    checkpoint = {
        "last_counts": [
            {"label": "ACCOUNT", "count": 1},
            {"label": "COMPLAINT", "count": 1},
            {"label": "REFUND", "count": 2},
        ]
    }

    response = _runner(service, model).run(
        "What is the total count of the last two?", "structured", checkpoint
    )

    assert response.answer == "The total count of the last two is 3."
    assert checkpoint["last_counts"][-1] == {"label": "REFUND", "count": 2}
    assert model.calls == 1


def test_max_iteration_fallback(test_settings) -> None:
    settings = test_settings.model_copy(update={"max_agent_iterations": 1})
    service = AgentService(settings, router=FakeRouter("structured"))
    model = FakeChatModel(
        [AIMessage(content="", tool_calls=[{"name": "count_rows", "args": {"category": "REFUND"}, "id": "t1"}])]
    )

    response = _runner(service, model).run("How many refund requests did we get?", "structured", {})

    assert "maximum number" in response.answer


def test_vague_count_searches_before_counting(test_settings) -> None:
    service = AgentService(test_settings, router=FakeRouter("structured"))

    response = _runner(service, FakeAngryModel()).run(
        "how many angry customers there were?", "structured", {}
    )

    assert "5 rows" not in response.answer
    assert any(step.title == "Tool call: search_rows" for step in response.reasoning)
    assert any("search_id" in step.detail for step in response.reasoning if step.title == "Tool call: count_rows")
