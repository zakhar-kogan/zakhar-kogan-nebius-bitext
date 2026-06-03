"""LangGraph orchestration and high-level turn execution service."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from typing import Any, Protocol, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from rapidfuzz import fuzz

from bitext_agent.agent_runner import AgentRunner, LlmReActRunner, LlmRouter, Router
from bitext_agent.config import Settings, get_settings
from bitext_agent.data import DatasetRepository
from bitext_agent.llm import build_chat_model, configure_langsmith, invoke_with_usage_log
from bitext_agent.prompts import PromptStore
from bitext_agent.memory import (
    ConversationCheckpointStore,
    distill_profile_memory,
    refresh_session_summary,
)
from bitext_agent.schemas import (
    AgentEvent,
    AgentResponse,
    ReasoningStep,
    RecommendationRefinementResult,
    RouteKind,
)
from bitext_agent.settings_store import SettingsStore, normalize_recommendation_query
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


class RecommendationRefiner(Protocol):
    """Interface for interpreting semantic changes to pending recommendations."""

    def refine(
        self, message: str, pending: dict[str, str], session_id: str, user_uuid: str
    ) -> RecommendationRefinementResult:
        """Return a refined query or mark the request unclear."""


class LlmRecommendationRefiner:
    """Nebius-backed structured refiner for pending recommendation changes."""

    def __init__(
        self,
        settings: Settings,
        store: SettingsStore,
        prompt_store: PromptStore,
        repository: DatasetRepository,
    ) -> None:
        self.settings = settings
        self.store = store
        self.prompt_store = prompt_store
        self.repository = repository
        model = build_chat_model(settings, settings.active_recommender_model, temperature=0.0)
        self.chain = model.with_structured_output(RecommendationRefinementResult, method="json_mode")

    def refine(
        self, message: str, pending: dict[str, str], session_id: str, user_uuid: str
    ) -> RecommendationRefinementResult:
        """Interpret a user's requested change to the pending recommendation."""

        if not self.settings.nebius_api_key:
            raise RuntimeError("NEBIUS_API_KEY is required for recommendation refinement.")
        context = {
            "pending_recommendation": pending,
            "recent_turns": self.store.list_turns(session_id, limit=8),
            "profile_facts": [
                fact.model_dump(mode="json") for fact in self.store.list_profile_facts(user_uuid)
            ],
            "dataset_status": self.repository.dataset_status(),
        }
        messages = [
            SystemMessage(content=self.prompt_store.load("recommendation_refinement")),
            SystemMessage(content="Runtime context JSON:\n" + json.dumps(context, default=str)),
            HumanMessage(content=message),
        ]
        return invoke_with_usage_log(
            self.store,
            self.settings.active_recommender_model,
            session_id,
            user_uuid,
            lambda: self.chain.invoke(messages),
        )


class AgentService:
    """Application service that wires settings, stores, data tools, and graph execution."""

    def __init__(
        self,
        settings: Settings | None = None,
        router: Router | None = None,
        runner_factory: Any | None = None,
        recommendation_refiner: RecommendationRefiner | None = None,
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
        self.recommendation_refiner = recommendation_refiner or LlmRecommendationRefiner(
            self.settings, self.store, self.prompt_store, self.repository
        )

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
            {
                "route": response.route,
                "suggested_query": response.suggested_query,
                "visual_artifacts": [
                    artifact.model_dump(mode="json") for artifact in response.visual_artifacts
                ],
            },
        )
        refresh_session_summary(
            self.store,
            session_id,
            user_uuid,
            compact_after_turns=self.settings.session_compaction_turn_threshold,
            keep_recent_turns=self.settings.session_recent_turn_limit,
        )
        return response

    def stream_turn(
        self,
        message: str,
        session_id: str,
        external_user_id: str | None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> Iterator[AgentEvent]:
        """Run one conversation turn and yield progress events."""

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
        state: GraphState = {"message": message, "session_id": session_id, "user_uuid": user_uuid}
        final_response: AgentResponse | None = None
        cancel_check = cancel_check or (lambda: False)

        try:
            state = _load_context(self, state)
            if cancel_check():
                final_response = _cancelled_response("structured", [])
                yield AgentEvent(kind="cancelled", title="Stopped", detail=final_response.answer, final_response=final_response)
                return
            state = _route_query(self, state)
            route_step = ReasoningStep(
                kind="route",
                title="Router",
                detail=f"Classified as {state['route']}. {state.get('route_reason', '')}".strip(),
            )
            if cancel_check():
                final_response = _cancelled_response(state["route"], [route_step])
                yield AgentEvent(kind="cancelled", title="Stopped", detail=final_response.answer, final_response=final_response)
                return

            if state["route"] == "out_of_scope":
                state = _decline_out_of_scope(state)
                final_response = state["response"]
                yield AgentEvent(kind="route", title=route_step.title, detail=route_step.detail)
                yield AgentEvent(kind="final", title="Final", detail=final_response.answer, answer_delta=final_response.answer, final_response=final_response)
            elif state["route"] == "recommendation":
                state = _query_recommendation(self, state)
                final_response = state["response"]
                for step in final_response.reasoning:
                    yield AgentEvent(kind=step.kind, title=step.title, detail=step.detail)
                yield AgentEvent(kind="final", title="Final", detail=final_response.answer, answer_delta=final_response.answer, final_response=final_response)
            else:
                async_state = _agent_runner_stream(self, state, cancel_check)
                for event in async_state:
                    if event.final_response:
                        final_response = event.final_response
                    yield event
                state["response"] = final_response or _cancelled_response(state["route"], [])

            state = _memory_distillation(self, state)
            if final_response and state.get("response") is final_response:
                memory_steps = [step for step in final_response.reasoning if step.kind == "memory"]
                for step in memory_steps[-1:]:
                    yield AgentEvent(kind="memory", title=step.title, detail=step.detail)
            _finalize(self, state)
        finally:
            if final_response is not None:
                self.store.add_turn(
                    session_id,
                    user_uuid,
                    "assistant",
                    final_response.answer,
                    {
                        "route": final_response.route,
                        "suggested_query": final_response.suggested_query,
                        "visual_artifacts": [
                            artifact.model_dump(mode="json")
                            for artifact in final_response.visual_artifacts
                        ],
                    },
                )
                refresh_session_summary(
                    self.store,
                    session_id,
                    user_uuid,
                    compact_after_turns=self.settings.session_compaction_turn_threshold,
                    keep_recent_turns=self.settings.session_recent_turn_limit,
                )

    def starter_recommendations(self) -> list[str]:
        """Return cached starter questions based on dataset shape."""

        status = self.repository.dataset_status()
        if not status.get("exists"):
            return ["Run uv run python scripts/download_dataset.py, then ask what categories exist."]
        cache_key = f"starter:v2:{status.get('rows')}:{status.get('categories')}:{status.get('intents')}"
        cached = self.store.get_cached_recommendations(cache_key)
        if cached:
            return cached
        categories = self.repository.list_categories()
        first = categories[0] if categories else "ACCOUNT"
        recommendations = [
            "What categories exist in the dataset?",
            "How many refund requests did we get?",
            "Show a bar chart of the category breakdown.",
            "Show a bar chart of the top intents.",
            f"What is the distribution of intents in the {first} category?",
            "Summarize how agents respond to complaint intents.",
        ]
        self.store.set_cached_recommendations(cache_key, recommendations)
        return recommendations

    def recommend_queries(
        self, session_id: str, external_user_id: str | None, limit: int = 2
    ) -> list[str]:
        """Return starter or profile-aware query recommendations for UI buttons."""

        return [str(slot["query"]) for slot in self.recommendation_slots(session_id, external_user_id, limit)]

    def recommendation_slots(
        self, session_id: str, external_user_id: str | None, limit: int = 2
    ) -> list[dict[str, object]]:
        """Return visible recommendation slots, creating missing slots only."""

        user_uuid, _ = self.store.get_or_create_user(external_user_id)
        slots = self.store.list_recommendation_slots(session_id)
        for index in range(limit):
            if any(int(slot["slot_index"]) == index for slot in slots):
                continue
            replacement = self._next_recommendation(session_id, user_uuid, index, limit)
            self.store.set_recommendation_slot(session_id, index, replacement)
            if replacement:
                slots.append({"slot_index": index, "query": replacement})
        return [slot for slot in sorted(slots, key=lambda item: int(item["slot_index"])) if int(slot["slot_index"]) < limit]

    def replace_recommendation_slot(
        self, session_id: str, external_user_id: str | None, slot_index: int, limit: int = 2
    ) -> str | None:
        """Replace one consumed recommendation slot using current session context."""

        user_uuid, _ = self.store.get_or_create_user(external_user_id)
        replacement = self._next_recommendation(session_id, user_uuid, slot_index, limit)
        self.store.set_recommendation_slot(session_id, slot_index, replacement)
        return replacement

    def _next_recommendation(
        self, session_id: str, user_uuid: str, slot_index: int, limit: int
    ) -> str | None:
        candidates = _recommendation_candidates(self, session_id, user_uuid)
        visible = [
            str(slot["query"])
            for slot in self.store.list_recommendation_slots(session_id)
            if int(slot["slot_index"]) != slot_index and int(slot["slot_index"]) < limit
        ]
        used = self.store.list_selected_recommendation_queries(session_id)
        blocked = visible + used
        for candidate in candidates:
            if not _is_similar_recommendation(candidate, blocked):
                return candidate
        return None

    def distill_session(self, session_id: str, external_user_id: str | None) -> list[str]:
        """Distill profile memory for a session on user-controlled boundaries."""

        user_uuid, _ = self.store.get_or_create_user(external_user_id)
        return distill_profile_memory(self.store, session_id, user_uuid, settings=self.settings)


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


def _agent_runner_stream(
    service: AgentService, state: GraphState, cancel_check: Callable[[], bool]
) -> Iterator[AgentEvent]:
    registry = ToolRegistry(
        repository=service.repository,
        store=service.store,
        session_id=state["session_id"],
        user_uuid=state["user_uuid"],
    )
    if service.runner_factory:
        runner: AgentRunner = service.runner_factory(registry, service, state)
        response = runner.run(
            state["message"], state["route"], state["checkpoint"], state.get("route_reason", "")
        )
        for step in response.reasoning:
            yield AgentEvent(kind=step.kind, title=step.title, detail=step.detail)
        yield AgentEvent(kind="final", title="Final", detail=response.answer, answer_delta=response.answer, final_response=response)
        return
    runner = LlmReActRunner(
        registry=registry,
        settings=service.settings,
        store=service.store,
        prompt_store=service.prompt_store,
    )
    yield from runner.stream(
        state["message"], state["route"], state["checkpoint"], state.get("route_reason", ""), cancel_check
    )


def _query_recommendation(service: AgentService, state: GraphState) -> GraphState:
    registry = ToolRegistry(
        repository=service.repository,
        store=service.store,
        session_id=state["session_id"],
        user_uuid=state["user_uuid"],
    )
    normalized_reply = _normalize_recommendation_reply(state["message"])
    pending = service.store.get_pending_recommendation(state["session_id"])
    reasoning = [
        ReasoningStep(
            kind="route",
            title="Router",
            detail=f"Classified as recommendation. {state.get('route_reason', '')}".strip(),
        )
    ]
    if pending and _is_recommendation_confirmation(normalized_reply):
        service.store.clear_pending_recommendation(state["session_id"])
        reasoning.append(ReasoningStep(kind="recommendation", title="Confirmed", detail=pending["query"]))
        next_state = {**state, "message": pending["query"], "route": "structured", "route_reason": "Confirmed pending recommendation."}
        executed = _agent_runner(service, next_state)
        executed_response = executed["response"]
        executed_response.reasoning = reasoning + executed_response.reasoning
        return {**state, "checkpoint": executed.get("checkpoint", state.get("checkpoint", {})), "response": executed_response}
    if pending and _is_recommendation_cancellation(normalized_reply):
        service.store.clear_pending_recommendation(state["session_id"])
        answer = "Okay, I cancelled that suggestion. What would you like to explore instead?"
        reasoning.append(ReasoningStep(kind="recommendation", title="Cancelled", detail=pending["query"]))
        return {
            **state,
            "response": AgentResponse(answer=answer, route="recommendation", reasoning=reasoning),
        }
    if pending:
        refinement = service.recommendation_refiner.refine(
            state["message"], pending, state["session_id"], state["user_uuid"]
        )
        query = refinement.refined_query.strip() if refinement.refined_query else ""
        if refinement.unclear or not query:
            answer = (
                f"I still have this pending suggestion: {pending['query']}\n"
                "Should I run it, change it, or cancel it?"
            )
            reasoning.append(
                ReasoningStep(kind="recommendation", title="Needs clarification", detail=refinement.reason)
            )
            return {
                **state,
                "response": AgentResponse(
                    answer=answer,
                    route="recommendation",
                    reasoning=reasoning,
                    suggested_query=pending["query"],
                ),
            }
        service.store.set_pending_recommendation(state["session_id"], state["user_uuid"], query, refinement.reason)
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


def _normalize_recommendation_reply(message: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", message.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _is_recommendation_confirmation(normalized: str) -> bool:
    return normalized in {
        "yes",
        "y",
        "yep",
        "yeah",
        "sure",
        "ok",
        "okay",
        "do it",
        "yes do it",
        "sure do it",
        "go ahead",
        "please do",
        "run it",
        "execute it",
    }


def _is_recommendation_cancellation(normalized: str) -> bool:
    return normalized in {"no", "nope", "cancel", "never mind", "dont", "don t", "do not"}


def _memory_distillation(service: AgentService, state: GraphState) -> GraphState:
    mode = service.settings.memory_distillation_mode
    if mode == "per_conversation":
        return state
    if mode == "every_n_turns":
        user_turns = service.store.count_user_turns(state["session_id"], state["user_uuid"])
        if user_turns % service.settings.memory_distillation_turn_interval != 0:
            return state

    saved = distill_profile_memory(
        service.store,
        state["session_id"],
        state["user_uuid"],
        settings=service.settings,
    )
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


def _recommendation_candidates(service: AgentService, session_id: str, user_uuid: str) -> list[str]:
    facts = service.store.list_profile_facts(user_uuid)
    recent_text = " ".join(
        turn["content"].lower() for turn in service.store.list_turns(session_id, limit=8)
    )
    candidates = _profile_recommendations([fact.fact for fact in facts]) if facts else []
    return _dedupe(_recent_recommendations(recent_text) + candidates + service.starter_recommendations())


def _profile_recommendations(facts: list[str]) -> list[str]:
    text = " ".join(facts).lower()
    recommendations: list[str] = []
    if any(term in text for term in ["refund", "money back", "billing"]):
        recommendations.extend([
            "Show me 5 examples from the REFUND category.",
            "What is the distribution of intents in the REFUND category?",
            "Show a bar chart of intent distribution for REFUND.",
        ])
    if any(term in text for term in ["shipping", "shipment", "delivery"]):
        recommendations.extend([
            "Show me 5 examples from the SHIPPING category.",
            "Show a bar chart of intent distribution for SHIPPING.",
        ])
    if any(term in text for term in ["complaint", "angry", "frustrated", "escalation"]):
        recommendations.extend([
            "Summarize how agents respond to complaint intents.",
            "Show me 5 examples from the FEEDBACK category.",
        ])
    if "concise" in text or "brief" in text:
        recommendations.append("What categories exist in the dataset?")
    if "example" in text:
        recommendations.append("Show me 5 examples from the dataset.")
    if "count" in text or "number" in text:
        recommendations.append("How many refund requests did we get?")
    return recommendations or [
        "What categories exist in the dataset?",
        "Summarize how agents respond to complaint intents.",
    ]


def _recent_recommendations(recent_text: str) -> list[str]:
    recommendations: list[str] = []
    if "refund" in recent_text or "money back" in recent_text:
        recommendations.append("Show me 5 examples from the REFUND category.")
        recommendations.append("Show a bar chart of intent distribution for REFUND.")
    if "shipping" in recent_text or "shipment" in recent_text:
        recommendations.append("Show a bar chart of intent distribution for SHIPPING.")
    if "complaint" in recent_text:
        recommendations.append("Summarize how agents respond to complaint intents.")
    return recommendations


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _exclude_selected(values: list[str], selected: set[str]) -> list[str]:
    return [value for value in values if normalize_recommendation_query(value) not in selected]


def _is_similar_recommendation(candidate: str, blocked: list[str]) -> bool:
    candidate_key = normalize_recommendation_query(candidate)
    for value in blocked:
        value_key = normalize_recommendation_query(value)
        if not value_key:
            continue
        if candidate_key == value_key or fuzz.token_set_ratio(candidate_key, value_key) >= 90:
            return True
    return False


def _cancelled_response(route: RouteKind, reasoning: list[ReasoningStep]) -> AgentResponse:
    answer = "Stopped before completion."
    return AgentResponse(
        answer=answer,
        route=route,
        reasoning=[*reasoning, ReasoningStep(kind="cancelled", title="Stopped", detail=answer)],
    )


graph = build_graph()
