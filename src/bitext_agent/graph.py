"""LangGraph orchestration and high-level turn execution service."""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from bitext_agent.agent_runner import AgentRunner, LlmReActRunner, LlmRouter, Router
from bitext_agent.config import Settings, get_settings
from bitext_agent.data import DatasetRepository
from bitext_agent.llm import configure_langsmith
from bitext_agent.prompts import PromptStore
from bitext_agent.memory import (
    ConversationCheckpointStore,
    distill_profile_memory,
    refresh_session_summary,
)
from bitext_agent.schemas import AgentResponse, ReasoningStep, RouteKind
from bitext_agent.settings_store import SettingsStore
from bitext_agent.tools import ToolRegistry


class GraphState(TypedDict, total=False):
    """State passed through the LangGraph nodes."""

    message: str
    session_id: str
    user_uuid: str
    route: RouteKind
    route_reason: str
    checkpoint: dict[str, Any]
    response: AgentResponse


class AgentService:
    """Application service that wires settings, stores, data tools, and graph execution."""

    def __init__(
        self,
        settings: Settings | None = None,
        router: Router | None = None,
        runner_factory: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        configure_langsmith(self.settings)
        self.repository = DatasetRepository(self.settings.dataset_path)
        self.store = SettingsStore(self.settings.app_db_path)
        self.checkpoints = ConversationCheckpointStore(self.settings.checkpoint_db_path)
        self._langgraph_checkpoint_conn = sqlite3.connect(
            self.settings.checkpoint_db_path, check_same_thread=False
        )
        self.langgraph_checkpointer = SqliteSaver(self._langgraph_checkpoint_conn)
        self.prompt_store = PromptStore(self.store)
        self.router = router or LlmRouter(self.settings, self.store, self.prompt_store)
        self.runner_factory = runner_factory

    def run_turn(self, message: str, session_id: str, external_user_id: str | None) -> AgentResponse:
        """Run one conversation turn and persist history plus checkpoint state."""

        user_uuid, resolved_user_id = self.store.get_or_create_user(external_user_id)
        _ = resolved_user_id
        self.store.add_turn(session_id, user_uuid, "user", message)
        refresh_session_summary(
            self.store,
            session_id,
            user_uuid,
            compact_after_turns=self.settings.session_compaction_turn_threshold,
            keep_recent_turns=self.settings.session_recent_turn_limit,
        )
        state: GraphState = {
            "message": message,
            "session_id": session_id,
            "user_uuid": user_uuid,
        }
        result = build_graph(self).invoke(state, config={"configurable": {"thread_id": session_id}})
        response = result["response"]
        self.store.add_turn(
            session_id,
            user_uuid,
            "assistant",
            response.answer,
            {"route": response.route, "suggested_query": response.suggested_query},
        )
        refresh_session_summary(
            self.store,
            session_id,
            user_uuid,
            compact_after_turns=self.settings.session_compaction_turn_threshold,
            keep_recent_turns=self.settings.session_recent_turn_limit,
        )
        return response

    def starter_recommendations(self) -> list[str]:
        """Return cached starter questions based on dataset shape."""

        status = self.repository.dataset_status()
        if not status.get("exists"):
            return ["Run uv run python scripts/download_dataset.py, then ask what categories exist."]
        cache_key = f"starter:{status.get('rows')}:{status.get('categories')}:{status.get('intents')}"
        cached = self.store.get_cached_recommendations(cache_key)
        if cached:
            return cached
        categories = self.repository.list_categories()
        first = categories[0] if categories else "ACCOUNT"
        recommendations = [
            "What categories exist in the dataset?",
            "How many refund requests did we get?",
            f"What is the distribution of intents in the {first} category?",
            "Summarize how agents respond to complaint intents.",
        ]
        self.store.set_cached_recommendations(cache_key, recommendations)
        return recommendations

    def distill_session(self, session_id: str, external_user_id: str | None) -> list[str]:
        """Distill profile memory for a session on user-controlled boundaries."""

        user_uuid, _ = self.store.get_or_create_user(external_user_id)
        return distill_profile_memory(self.store, session_id, user_uuid)


def build_graph(service: AgentService | None = None):
    """Build the LangGraph StateGraph used by CLI, Streamlit, and Studio."""

    service = service or AgentService()
    builder = StateGraph(GraphState)
    builder.add_node("load_context", lambda state: _load_context(service, state))
    builder.add_node("route_query", lambda state: _route_query(service, state))
    builder.add_node("decline_out_of_scope", _decline_out_of_scope)
    builder.add_node("agent_runner", lambda state: _agent_runner(service, state))
    builder.add_node("query_recommendation", lambda state: _query_recommendation(service, state))
    builder.add_node("memory_distillation", lambda state: _memory_distillation(service, state))
    builder.add_node("finalize", lambda state: _finalize(service, state))
    builder.set_entry_point("load_context")
    builder.add_edge("load_context", "route_query")
    builder.add_conditional_edges(
        "route_query",
        _route_edge,
        {
            "decline_out_of_scope": "decline_out_of_scope",
            "query_recommendation": "query_recommendation",
            "agent_runner": "agent_runner",
        },
    )
    builder.add_edge("decline_out_of_scope", "finalize")
    builder.add_edge("query_recommendation", "memory_distillation")
    builder.add_edge("agent_runner", "memory_distillation")
    builder.add_edge("memory_distillation", "finalize")
    builder.add_edge("finalize", END)
    checkpointer = service.langgraph_checkpointer if service else None
    return builder.compile(checkpointer=checkpointer)


def _load_context(service: AgentService, state: GraphState) -> GraphState:
    checkpoint = service.checkpoints.load(state["session_id"])
    return {**state, "checkpoint": checkpoint}


def _route_query(service: AgentService, state: GraphState) -> GraphState:
    decision = service.router.route(state["message"], state["session_id"], state["user_uuid"])
    return {**state, "route": decision.route, "route_reason": decision.reason}


def _route_edge(state: GraphState) -> str:
    if state["route"] == "out_of_scope":
        return "decline_out_of_scope"
    if state["route"] == "recommendation":
        return "query_recommendation"
    return "agent_runner"


def _decline_out_of_scope(state: GraphState) -> GraphState:
    response = AgentResponse(
        answer=(
            "I can only answer questions about the Bitext customer service dataset. "
            "Ask about categories, intents, counts, examples, distributions, or summaries."
        ),
        route="out_of_scope",
        reasoning=[
            ReasoningStep(
                kind="route",
                title="Router",
                detail=state.get("route_reason", "Question is unrelated to the dataset."),
            )
        ],
    )
    return {**state, "response": response}


def _agent_runner(service: AgentService, state: GraphState) -> GraphState:
    registry = ToolRegistry(
        repository=service.repository,
        store=service.store,
        session_id=state["session_id"],
        user_uuid=state["user_uuid"],
    )
    if service.runner_factory:
        runner: AgentRunner = service.runner_factory(registry, service, state)
    else:
        runner = LlmReActRunner(
            registry=registry,
            settings=service.settings,
            store=service.store,
            prompt_store=service.prompt_store,
        )
    response = runner.run(
        state["message"],
        state["route"],
        state["checkpoint"],
        state.get("route_reason", ""),
    )
    return {**state, "response": response}


def _query_recommendation(service: AgentService, state: GraphState) -> GraphState:
    registry = ToolRegistry(
        repository=service.repository,
        store=service.store,
        session_id=state["session_id"],
        user_uuid=state["user_uuid"],
    )
    lower = state["message"].lower().strip()
    pending = service.store.get_pending_recommendation(state["session_id"])
    reasoning = [
        ReasoningStep(
            kind="route",
            title="Router",
            detail=f"Classified as recommendation. {state.get('route_reason', '')}".strip(),
        )
    ]
    if pending and re.fullmatch(r"(yes|y|do it|go ahead|please do|run it|execute it)[.! ]*", lower):
        service.store.clear_pending_recommendation(state["session_id"])
        reasoning.append(ReasoningStep(kind="recommendation", title="Confirmed", detail=pending["query"]))
        next_state = {**state, "message": pending["query"], "route": "structured", "route_reason": "Confirmed pending recommendation."}
        executed = _agent_runner(service, next_state)
        executed_response = executed["response"]
        executed_response.reasoning = reasoning + executed_response.reasoning
        return {**state, "checkpoint": executed.get("checkpoint", state.get("checkpoint", {})), "response": executed_response}
    if pending and any(term in lower for term in ["example", "summar", "count", "distribution"]):
        query = _refine_recommendation(lower, pending["query"])
        service.store.set_pending_recommendation(
            state["session_id"], state["user_uuid"], query, "Refined based on your preference."
        )
        reasoning.append(ReasoningStep(kind="recommendation", title="Refined", detail=query))
        return {
            **state,
            "response": AgentResponse(
                answer=f"Then I suggest: {query}\nShould I go ahead?",
                route="recommendation",
                reasoning=reasoning,
                suggested_query=query,
            ),
        }

    start = time.perf_counter()
    result = registry.call("recommend_next_query", session_id=state["session_id"], user_id=state["user_uuid"])
    service.store.log_tool_call(
        session_id=state["session_id"],
        user_uuid=state["user_uuid"],
        tool_name="recommend_next_query",
        status="ok",
        latency_ms=int((time.perf_counter() - start) * 1000),
    )
    reasoning.append(
        ReasoningStep(kind="tool", title="Tool call: recommend_next_query", detail=str({"session_id": state["session_id"]}))
    )
    return {
        **state,
        "response": AgentResponse(
            answer=f"Based on the conversation, I suggest: {result.query}\nReason: {result.reason}\nShould I go ahead?",
            route="recommendation",
            reasoning=reasoning,
            suggested_query=result.query,
        ),
    }


def _refine_recommendation(message: str, previous: str) -> str:
    if "example" in message:
        if "refund" in previous.lower():
            return "Show me 5 examples from the REFUND category."
        if "account" in previous.lower():
            return "Show me 5 examples from the ACCOUNT category."
        if "complaint" in previous.lower():
            return "Show me 5 examples from the FEEDBACK category."
        return "Show me 5 examples from the dataset."
    if "summar" in message:
        return "Summarize how agents respond to complaint intents."
    if "count" in message:
        return "How many refund requests did we get?"
    if "distribution" in message:
        return "What is the distribution of intents in the REFUND category?"
    return previous


def _memory_distillation(service: AgentService, state: GraphState) -> GraphState:
    mode = service.settings.memory_distillation_mode
    if mode == "per_conversation":
        return state
    if mode == "every_n_turns":
        user_turns = service.store.count_user_turns(state["session_id"], state["user_uuid"])
        if user_turns % service.settings.memory_distillation_turn_interval != 0:
            return state

    saved = distill_profile_memory(service.store, state["session_id"], state["user_uuid"])
    if not saved:
        return state
    response = state.get("response")
    if response:
        response.reasoning.append(
            ReasoningStep(
                kind="memory",
                title="Memory",
                detail=f"Saved {len(saved)} profile fact(s).",
            )
        )
    return state


def _finalize(service: AgentService, state: GraphState) -> GraphState:
    service.checkpoints.save(state["session_id"], state.get("checkpoint", {}))
    return state


graph = build_graph()
