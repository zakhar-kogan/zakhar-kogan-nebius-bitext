"""LLM-backed routing and ReAct agent execution."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from bitext_agent.config import Settings
from bitext_agent.llm import build_chat_model, invoke_with_usage_log
from bitext_agent.prompts import PromptStore
from bitext_agent.schemas import (
    AgentEvent,
    AgentResponse,
    ChartArtifact,
    ReasoningStep,
    RouteKind,
    RouterDecision,
)
from bitext_agent.settings_store import SettingsStore
from bitext_agent.tools import ToolRegistry


class Router(ABC):
    """Interface for query routing before tool selection."""

    @abstractmethod
    def route(self, message: str, session_id: str, user_uuid: str) -> RouterDecision:
        """Classify a user message."""


class AgentRunner(ABC):
    """Interface for interchangeable agent execution strategies."""

    @abstractmethod
    def run(
        self,
        message: str,
        route: RouteKind,
        checkpoint: dict[str, Any],
        route_reason: str = "",
    ) -> AgentResponse:
        """Run one user turn and return a final response plus trace."""


CancelCheck = Callable[[], bool]


class ChatModelLike(Protocol):
    """Small protocol used by tests to inject fake chat models."""

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Invoke a chat model."""


class LlmRouter(Router):
    """Nebius-backed structured-output router."""

    def __init__(self, settings: Settings, store: SettingsStore, prompt_store: PromptStore) -> None:
        self.settings = settings
        self.store = store
        self.prompt_store = prompt_store
        model = build_chat_model(settings, settings.router_model, temperature=0.0)
        self.chain = model.with_structured_output(RouterDecision, method="json_mode")

    def route(self, message: str, session_id: str, user_uuid: str) -> RouterDecision:
        """Classify a user message with the router model."""

        if not self.settings.nebius_api_key:
            raise RuntimeError("NEBIUS_API_KEY is required for LLM routing.")
        system = self.prompt_store.load("router")
        context = {"pending_recommendation": self.store.get_pending_recommendation(session_id)}
        pending = context["pending_recommendation"]
        user_content = (
            f"Pending recommendation JSON: {json.dumps(pending, default=str)}\nUser message: {message}"
            if pending
            else message
        )
        messages = [
            SystemMessage(content=system),
            SystemMessage(content="Runtime context JSON:\n" + json.dumps(context, default=str)),
            HumanMessage(content=user_content),
        ]
        return invoke_with_usage_log(
            self.store,
            self.settings.router_model,
            session_id,
            user_uuid,
            lambda: self.chain.invoke(messages),
        )


class LlmReActRunner(AgentRunner):
    """Bounded ReAct tool loop driven by the main LLM."""

    def __init__(
        self,
        registry: ToolRegistry,
        settings: Settings,
        store: SettingsStore,
        prompt_store: PromptStore,
        model: ChatModelLike | None = None,
    ) -> None:
        self.registry = registry
        self.settings = settings
        self.store = store
        self.prompt_store = prompt_store
        self.model = model or build_chat_model(settings, settings.main_model, temperature=0.0).bind_tools(
            registry.langchain_tools()
        )

    def run(
        self,
        message: str,
        route: RouteKind,
        checkpoint: dict[str, Any],
        route_reason: str = "",
    ) -> AgentResponse:
        """Execute a bounded ReAct loop for a dataset query."""

        final_response: AgentResponse | None = None
        for event in self.stream(message, route, checkpoint, route_reason):
            if event.final_response:
                final_response = event.final_response
        if final_response is None:
            return AgentResponse(
                answer="I could not complete the request.",
                route=route,
                reasoning=[ReasoningStep(kind="error", title="Error", detail="No final response was produced.")],
            )
        return final_response

    def stream(
        self,
        message: str,
        route: RouteKind,
        checkpoint: dict[str, Any],
        route_reason: str = "",
        cancel_check: CancelCheck | None = None,
    ) -> Iterator[AgentEvent]:
        """Execute a bounded ReAct loop and yield visible progress events."""

        if not self.settings.nebius_api_key and not _is_fake_model(self.model):
            raise RuntimeError("NEBIUS_API_KEY is required for LLM agent execution.")

        cancel_check = cancel_check or (lambda: False)
        reasoning = [
            ReasoningStep(
                kind="route",
                title="Router",
                detail=f"Classified as {route}." + (f" {route_reason}" if route_reason else ""),
            )
        ]
        visual_artifacts: list[ChartArtifact] = []
        yield _event_from_step(reasoning[-1])
        if cancel_check():
            yield _cancelled_event(route, reasoning)
            return
        show_more_response = self._try_show_more_followup(message, checkpoint, reasoning, route)
        if show_more_response:
            for step in show_more_response.reasoning[1:]:
                yield _event_from_step(step)
            yield AgentEvent(
                kind="final",
                title="Final",
                detail=_shorten(show_more_response.answer),
                answer_delta=show_more_response.answer,
                final_response=show_more_response,
            )
            return

        messages = self._initial_messages(message, route, checkpoint)

        for _ in range(self.settings.max_agent_iterations):
            if cancel_check():
                yield _cancelled_event(route, reasoning)
                return
            try:
                ai_message = self._invoke_model(messages)
            except Exception as exc:
                response = AgentResponse(
                    answer=f"I could not complete the model call: {exc}",
                    route=route,
                    reasoning=reasoning,
                )
                yield AgentEvent(kind="error", title="Model call failed", detail=str(exc), final_response=response)
                return
            messages.append(ai_message)
            tool_calls = getattr(ai_message, "tool_calls", None) or []
            if not tool_calls:
                answer = _message_text(ai_message)
                reasoning.append(ReasoningStep(kind="final", title="Final", detail=_shorten(answer)))
                response = AgentResponse(
                    answer=answer,
                    route=route,
                    reasoning=reasoning,
                    visual_artifacts=visual_artifacts,
                )
                yield AgentEvent(
                    kind="final",
                    title="Final",
                    detail=_shorten(answer),
                    answer_delta=answer,
                    final_response=response,
                )
                return
            for tool_call in tool_calls:
                if cancel_check():
                    yield _cancelled_event(route, reasoning)
                    return
                observation = self._execute_tool_call(tool_call, checkpoint, reasoning, visual_artifacts)
                yield _event_from_step(reasoning[-2])
                yield _event_from_step(reasoning[-1])
                messages.append(
                    ToolMessage(
                        content=observation,
                        tool_call_id=tool_call.get("id") or tool_call.get("tool_call_id") or "tool_call",
                    )
                )

        fallback = (
            "I reached the maximum number of reasoning steps before producing a final answer. "
            "Please narrow the question or ask for a specific count, examples, distribution, or summary."
        )
        reasoning.append(ReasoningStep(kind="fallback", title="Max iterations", detail=fallback))
        response = AgentResponse(
            answer=fallback,
            route=route,
            reasoning=reasoning,
            visual_artifacts=visual_artifacts,
        )
        yield AgentEvent(kind="fallback", title="Max iterations", detail=fallback, final_response=response)

    def _initial_messages(
        self, message: str, route: RouteKind, checkpoint: dict[str, Any]
    ) -> list[BaseMessage]:
        context = {
            "route": route,
            "session_id": self.registry.context.session_id,
            "user_uuid": self.registry.context.user_uuid,
            "checkpoint": checkpoint,
            "session_summary": self.registry.context.store.get_session_summary(
                self.registry.context.session_id
            ),
            "dataset_status": self.registry.context.repository.dataset_status(),
            "recent_turns": self._recent_turns_for_prompt(message),
            "profile_facts": [
                fact.model_dump(mode="json")
                for fact in self.registry.context.store.list_profile_facts(self.registry.context.user_uuid)
            ],
            "pending_recommendation": self.registry.context.store.get_pending_recommendation(
                self.registry.context.session_id
            ),
        }
        system = self.prompt_store.load("react_system")
        return [
            SystemMessage(content=system),
            SystemMessage(content="Runtime context JSON:\n" + json.dumps(context, default=str)),
            HumanMessage(content=message),
        ]

    def _invoke_model(self, messages: list[BaseMessage]) -> AIMessage:
        return invoke_with_usage_log(
            self.store,
            self.settings.main_model,
            self.registry.context.session_id,
            self.registry.context.user_uuid,
            lambda: self.model.invoke(messages),
        )

    def _execute_tool_call(
        self,
        tool_call: dict[str, Any],
        checkpoint: dict[str, Any],
        reasoning: list[ReasoningStep],
        visual_artifacts: list[ChartArtifact],
    ) -> str:
        name = tool_call.get("name", "")
        args = tool_call.get("args") or {}
        reasoning.append(ReasoningStep(kind="tool", title=f"Tool call: {name}", detail=str(args)))
        start = time.perf_counter()
        try:
            result = self.registry.call(name, **args)
        except Exception as exc:
            detail = f"Tool error: {exc}"
            self.store.log_tool_call(
                session_id=self.registry.context.session_id,
                user_uuid=self.registry.context.user_uuid,
                tool_name=name or "unknown",
                status="error",
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=str(exc),
            )
            reasoning.append(ReasoningStep(kind="observation", title="Observation", detail=detail))
            return json.dumps({"error": detail})

        self.store.log_tool_call(
            session_id=self.registry.context.session_id,
            user_uuid=self.registry.context.user_uuid,
            tool_name=name or "unknown",
            status="ok",
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
        self._update_checkpoint(name, args, result, checkpoint)
        artifact = _visual_artifact_from_result(name, result)
        if artifact:
            visual_artifacts.append(artifact)
        payload = result.model_dump(mode="json")
        observation = json.dumps(payload, ensure_ascii=False)
        reasoning.append(ReasoningStep(kind="observation", title="Observation", detail=_summarize_result(payload)))
        return observation

    def _try_show_more_followup(
        self,
        message: str,
        checkpoint: dict[str, Any],
        reasoning: list[ReasoningStep],
        route: RouteKind,
    ) -> AgentResponse | None:
        previous = checkpoint.get("last_examples")
        if not previous or not _is_show_more_request(message):
            return None

        n = _requested_more_count(message) or previous.get("n", 3)
        offset = previous.get("next_offset") or 0
        category = previous.get("category")
        intent = previous.get("intent")
        args = {
            "category": category,
            "intent": intent,
            "search_id": None if category or intent else previous.get("search_id"),
            "n": n,
            "offset": offset,
        }
        if not category and not intent and previous.get("query"):
            search_args = {
                "category": category,
                "intent": intent,
                "query": previous.get("query"),
                "fuzzy": previous.get("fuzzy", True),
                "limit": max(offset + n, n),
            }
            reasoning.append(ReasoningStep(kind="tool", title="Tool call: search_rows", detail=str(search_args)))
            start = time.perf_counter()
            try:
                search_result = self.registry.call("search_rows", **search_args)
            except Exception as exc:
                self.store.log_tool_call(
                    session_id=self.registry.context.session_id,
                    user_uuid=self.registry.context.user_uuid,
                    tool_name="search_rows",
                    status="error",
                    latency_ms=int((time.perf_counter() - start) * 1000),
                    error=str(exc),
                )
                detail = f"Tool error: {exc}"
                reasoning.append(ReasoningStep(kind="observation", title="Observation", detail=detail))
                return AgentResponse(answer=detail, route=route, reasoning=reasoning)
            self.store.log_tool_call(
                session_id=self.registry.context.session_id,
                user_uuid=self.registry.context.user_uuid,
                tool_name="search_rows",
                status="ok",
                latency_ms=int((time.perf_counter() - start) * 1000),
            )
            self._update_checkpoint("search_rows", search_args, search_result, checkpoint)
            payload = search_result.model_dump(mode="json")
            reasoning.append(ReasoningStep(kind="observation", title="Observation", detail=_summarize_result(payload)))
            args["search_id"] = search_result.search_id
        elif not category and not intent and previous.get("search_id"):
            answer = (
                "I cannot resume that example search after restart. "
                "Please repeat the category, intent, or search phrase."
            )
            reasoning.append(ReasoningStep(kind="fallback", title="Search expired", detail=answer))
            return AgentResponse(answer=answer, route=route, reasoning=reasoning)

        reasoning.append(ReasoningStep(kind="tool", title="Tool call: show_examples", detail=str(args)))
        start = time.perf_counter()
        try:
            result = self.registry.call("show_examples", **args)
        except Exception as exc:
            self.store.log_tool_call(
                session_id=self.registry.context.session_id,
                user_uuid=self.registry.context.user_uuid,
                tool_name="show_examples",
                status="error",
                latency_ms=int((time.perf_counter() - start) * 1000),
                error=str(exc),
            )
            detail = f"Tool error: {exc}"
            reasoning.append(ReasoningStep(kind="observation", title="Observation", detail=detail))
            return AgentResponse(answer=detail, route=route, reasoning=reasoning)

        self.store.log_tool_call(
            session_id=self.registry.context.session_id,
            user_uuid=self.registry.context.user_uuid,
            tool_name="show_examples",
            status="ok",
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
        self._update_checkpoint("show_examples", args, result, checkpoint)
        payload = result.model_dump(mode="json")
        reasoning.append(ReasoningStep(kind="observation", title="Observation", detail=_summarize_result(payload)))
        answer = _format_examples(result.rows, result.offset, result.next_offset)
        reasoning.append(ReasoningStep(kind="final", title="Final", detail=_shorten(answer)))
        return AgentResponse(answer=answer, route=route, reasoning=reasoning)

    def _recent_turns_for_prompt(self, message: str) -> list[dict[str, Any]]:
        turns = self.registry.context.store.list_turns(
            self.registry.context.session_id,
            limit=self.settings.session_recent_turn_limit + 1,
        )
        if turns and turns[-1]["role"] == "user" and turns[-1]["content"] == message:
            turns = turns[:-1]
        return turns[-self.settings.session_recent_turn_limit :]

    def _update_checkpoint(
        self, name: str, args: dict[str, Any], result: Any, checkpoint: dict[str, Any]
    ) -> None:
        if name == "count_rows":
            label = args.get("intent") or args.get("category") or args.get("search_id") or "matching rows"
            checkpoint.setdefault("last_counts", []).append({"label": label, "count": result.count})
            checkpoint["last_counts"] = checkpoint["last_counts"][-5:]
        if name == "search_rows":
            checkpoint["last_search"] = {
                "search_id": result.search_id,
                "query": args.get("query"),
                "category": args.get("category"),
                "intent": args.get("intent"),
                "fuzzy": args.get("fuzzy", True),
                "total_matches": result.total_matches,
            }
        if name == "show_examples":
            last_search = checkpoint.get("last_search") or {}
            search_query = None
            search_fuzzy = None
            if args.get("search_id") and args.get("search_id") == last_search.get("search_id"):
                search_query = last_search.get("query")
                search_fuzzy = last_search.get("fuzzy", True)
            checkpoint["last_examples"] = {
                "category": args.get("category"),
                "intent": args.get("intent"),
                "search_id": args.get("search_id"),
                "query": search_query,
                "fuzzy": search_fuzzy,
                "n": args.get("n", 3),
                "next_offset": result.next_offset,
            }


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(content).strip()


def _is_show_more_request(message: str) -> bool:
    return bool(re.search(r"\b(show|give|list|display)?\s*(me\s*)?(\d+\s*)?more\b", message, re.I))


def _requested_more_count(message: str) -> int | None:
    match = re.search(r"\b(\d+)\s+more\b", message, re.I)
    if not match:
        return None
    return max(1, min(int(match.group(1)), 25))


def _format_examples(rows: list[Any], offset: int, next_offset: int | None) -> str:
    if not rows:
        return "There are no more examples for the previous request."
    lines = [f"Here are {len(rows)} more examples starting at offset {offset}:", ""]
    for row in rows:
        lines.append(
            f"- Row {row.row_id} ({row.category} / {row.intent}): {row.instruction} -> {row.response}"
        )
    if next_offset is None:
        lines.append("\nNo further examples are available for that filter.")
    return "\n".join(lines)


def _summarize_result(payload: dict[str, Any]) -> str:
    if "artifact" in payload:
        artifact = payload["artifact"]
        return f"chart={artifact.get('title')}, rows={len(artifact.get('rows', []))}"
    if "count" in payload:
        return f"count={payload['count']}, filters={payload.get('filters')}"
    if "total_matches" in payload and "rows" in payload:
        return f"total_matches={payload['total_matches']}, rows_returned={len(payload['rows'])}"
    if "distribution" in payload:
        return f"distribution_rows={len(payload['distribution'])}"
    if "summary" in payload:
        return f"record_count={payload.get('record_count')}: {_shorten(payload['summary'])}"
    if "categories" in payload:
        return f"categories={payload['categories']}"
    if "intents" in payload:
        return f"intents={payload['intents'][:12]}"
    return _shorten(json.dumps(payload, ensure_ascii=False))


def _visual_artifact_from_result(name: str, result: Any) -> ChartArtifact | None:
    if hasattr(result, "artifact"):
        return result.artifact
    if name == "category_distribution" and hasattr(result, "distribution"):
        return ChartArtifact(
            title="Rows by category",
            x="category",
            y="count",
            rows=result.distribution,
        )
    return None


def _shorten(value: str, limit: int = 240) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _is_fake_model(model: Any) -> bool:
    return model.__class__.__module__.startswith("tests.") or model.__class__.__name__.startswith("Fake")


def _event_from_step(step: ReasoningStep) -> AgentEvent:
    return AgentEvent(kind=step.kind, title=step.title, detail=step.detail)


def _cancelled_event(route: RouteKind, reasoning: list[ReasoningStep]) -> AgentEvent:
    answer = "Stopped before completion."
    reasoning.append(ReasoningStep(kind="cancelled", title="Stopped", detail=answer))
    response = AgentResponse(answer=answer, route=route, reasoning=reasoning)
    return AgentEvent(kind="cancelled", title="Stopped", detail=answer, final_response=response)
