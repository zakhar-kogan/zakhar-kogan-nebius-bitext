"""Agent integration-style tests over a sample dataset."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from bitext_agent.agent_runner import AgentRunner, LlmReActRunner, Router
from bitext_agent.graph import AgentService
from bitext_agent.prompts import PromptStore
from bitext_agent.schemas import AgentResponse, ReasoningStep, RouteKind, RouterDecision
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
