"""Conversation checkpoint persistence and profile distillation helpers."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from rapidfuzz import fuzz

from bitext_agent.config import Settings
from bitext_agent.llm import build_chat_model, invoke_with_usage_log
from bitext_agent.schemas import ProfileFact
from bitext_agent.settings_store import SettingsStore, utc_now


PROFILE_MEMORY_LIMIT = 30
AUTO_DEDUPE_SIMILARITY = 95.0
LLM_DEDUPE_SIMILARITY = 85.0

DedupeReviewer = Callable[[str, str], bool | None]


class MemoryDedupeDecision(BaseModel):
    """Small-model decision for gray-band profile memory duplicates."""

    duplicate: bool


class ConversationCheckpointStore:
    """SQLite checkpoint store keyed by LangGraph thread/session ID."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        """Open a checkpoint database connection."""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create the lightweight checkpoint table."""

        with self.connect() as conn:
            conn.execute(
                """
                create table if not exists session_checkpoints (
                    session_id text primary key,
                    state_json text not null,
                    updated_at text not null
                )
                """
            )

    def load(self, session_id: str) -> dict[str, Any]:
        """Load checkpointed state for a session."""

        with self.connect() as conn:
            row = conn.execute(
                "select state_json from session_checkpoints where session_id = ?", (session_id,)
            ).fetchone()
        return dict(json.loads(row["state_json"])) if row else {"last_counts": [], "last_examples": None}

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist checkpointed state for a session."""

        with self.connect() as conn:
            conn.execute(
                """
                insert into session_checkpoints(session_id, state_json, updated_at) values (?, ?, ?)
                on conflict(session_id) do update set state_json = excluded.state_json,
                updated_at = excluded.updated_at
                """,
                (session_id, json.dumps(state), utc_now()),
            )


def distill_profile_memory(
    store: SettingsStore,
    session_id: str,
    user_uuid: str,
    settings: Settings | None = None,
    dedupe_reviewer: DedupeReviewer | None = None,
) -> list[str]:
    """Distill harmless profile facts from a conversation using deterministic rules."""

    cleanup_profile_memory(store, user_uuid)
    turns = store.list_turns(session_id, limit=50)
    saved: list[str] = []
    for turn in turns:
        if turn["role"] != "user":
            continue
        content = turn["content"].strip()
        candidates = _dedupe_candidates(_candidate_facts(content))
        for kind, fact, confidence in candidates:
            canonical_key = canonical_profile_key(kind, fact)
            match = _find_duplicate_fact(
                store=store,
                user_uuid=user_uuid,
                kind=kind,
                fact=fact,
                canonical_key=canonical_key,
                settings=settings,
                session_id=session_id,
                dedupe_reviewer=dedupe_reviewer,
            )
            if match:
                _merge_candidate_with_matches(
                    store=store,
                    matches=match,
                    kind=kind,
                    fact=fact,
                    canonical_key=canonical_key,
                    source=f"session:{session_id}",
                    confidence=confidence,
                )
                continue
            _, created = store.upsert_profile_fact(
                user_uuid=user_uuid,
                kind=kind,
                fact=fact,
                canonical_key=canonical_key,
                source=f"session:{session_id}",
                confidence=confidence,
            )
            if created:
                saved.append(fact)
    store.prune_profile_facts(user_uuid, max_active=PROFILE_MEMORY_LIMIT)
    return saved


def cleanup_profile_memory(store: SettingsStore, user_uuid: str) -> int:
    """Backfill canonical keys and collapse active exact duplicate memories."""

    changed = 0
    facts = store.list_profile_facts(user_uuid)
    for fact in facts:
        if fact.id is None:
            continue
        canonical_key = canonical_profile_key(fact.kind, fact.fact)
        if not fact.canonical_key:
            store.update_profile_fact_canonical_key(fact.id, canonical_key)
            fact.canonical_key = canonical_key
            changed += 1

    groups: dict[str, list[ProfileFact]] = {}
    for fact in store.list_profile_facts(user_uuid):
        if fact.canonical_key:
            groups.setdefault(fact.canonical_key, []).append(fact)

    for group in groups.values():
        if len(group) < 2:
            continue
        keeper = _best_existing_fact(group)
        duplicate_ids = [fact.id for fact in group if fact.id is not None and fact.id != keeper.id]
        changed += store.mark_profile_facts_duplicate(duplicate_ids)
    return changed


def canonical_profile_key(kind: str, fact: str) -> str:
    """Return a stable key for deduplicating equivalent profile facts."""

    normalized = re.sub(r"[^a-z0-9]+", " ", fact.lower()).strip()
    normalized = re.sub(r"\b(user|prefers|usually|often|explores|support|data)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f"{kind}:{normalized}"


def _dedupe_candidates(candidates: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    unique: dict[str, tuple[str, str, float]] = {}
    for kind, fact, confidence in candidates:
        key = canonical_profile_key(kind, fact)
        current = unique.get(key)
        if current is None or confidence > current[2]:
            unique[key] = (kind, fact, confidence)
    return list(unique.values())


def _find_duplicate_fact(
    store: SettingsStore,
    user_uuid: str,
    kind: str,
    fact: str,
    canonical_key: str,
    settings: Settings | None,
    session_id: str,
    dedupe_reviewer: DedupeReviewer | None,
) -> list[ProfileFact]:
    active_same_kind = [item for item in store.list_profile_facts(user_uuid) if item.kind == kind]
    exact = [item for item in active_same_kind if item.canonical_key == canonical_key]
    if exact:
        return exact

    scored = [
        (item, _profile_fact_similarity(fact, item.fact))
        for item in active_same_kind
    ]
    auto_matches = [item for item, score in scored if score >= AUTO_DEDUPE_SIMILARITY]
    if auto_matches:
        return auto_matches

    gray_matches = sorted(
        [(item, score) for item, score in scored if score >= LLM_DEDUPE_SIMILARITY],
        key=lambda pair: pair[1],
        reverse=True,
    )
    if not gray_matches:
        return []

    reviewer = dedupe_reviewer or _build_llm_dedupe_reviewer(store, settings, session_id, user_uuid)
    if reviewer is None:
        return []
    for existing, _ in gray_matches:
        try:
            decision = reviewer(fact, existing.fact)
        except Exception:
            return []
        if decision is True:
            return [existing]
        if decision is None:
            return []
    return []


def _merge_candidate_with_matches(
    store: SettingsStore,
    matches: list[ProfileFact],
    kind: str,
    fact: str,
    canonical_key: str,
    source: str,
    confidence: float,
) -> None:
    keeper = _best_existing_fact(matches)
    if keeper.id is None:
        return
    duplicate_ids = [item.id for item in matches if item.id is not None and item.id != keeper.id]
    store.mark_profile_facts_duplicate(duplicate_ids)
    if _candidate_beats_existing(confidence, keeper):
        store.update_profile_fact(keeper.id, kind, fact, canonical_key, source, confidence)


def _best_existing_fact(facts: list[ProfileFact]) -> ProfileFact:
    return max(
        facts,
        key=lambda fact: (
            fact.confidence,
            fact.updated_at.isoformat() if fact.updated_at else "",
            fact.id or 0,
        ),
    )


def _candidate_beats_existing(confidence: float, existing: ProfileFact) -> bool:
    return confidence > existing.confidence


def _profile_fact_similarity(first: str, second: str) -> float:
    return float(
        fuzz.token_set_ratio(_normalize_for_similarity(first), _normalize_for_similarity(second))
    )


def _normalize_for_similarity(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", normalized)


def _build_llm_dedupe_reviewer(
    store: SettingsStore,
    settings: Settings | None,
    session_id: str,
    user_uuid: str,
) -> DedupeReviewer | None:
    if settings is None or not settings.nebius_api_key:
        return None

    model_name = settings.active_memory_dedupe_model
    chain = build_chat_model(settings, model_name, temperature=0.0).with_structured_output(
        MemoryDedupeDecision, method="json_mode"
    )

    def review(candidate: str, existing: str) -> bool | None:
        result = invoke_with_usage_log(
            store,
            model_name,
            session_id,
            user_uuid,
            lambda: chain.invoke([
                SystemMessage(
                    content=(
                        "Decide whether two profile memory facts mean the same durable user "
                        "memory. "
                        "Return JSON with a single boolean field named duplicate. Be conservative."
                    )
                ),
                HumanMessage(content=f"Candidate fact: {candidate}\nExisting fact: {existing}"),
            ]),
        )
        if isinstance(result, MemoryDedupeDecision):
            return result.duplicate
        if isinstance(result, dict) and isinstance(result.get("duplicate"), bool):
            return bool(result["duplicate"])
        return None

    return review


def refresh_session_summary(
    store: SettingsStore,
    session_id: str,
    user_uuid: str,
    compact_after_turns: int = 16,
    keep_recent_turns: int = 12,
) -> dict[str, Any] | None:
    """Cache a compact session summary while preserving full turn history."""

    total_turns = store.count_turns(session_id)
    if total_turns <= compact_after_turns:
        return store.get_session_summary(session_id)

    older_turn_count = max(total_turns - keep_recent_turns, 0)
    current = store.get_session_summary(session_id)
    if current and int(current["source_turn_count"]) >= older_turn_count:
        return current

    turns = store.list_turns(session_id, limit=total_turns)
    older_turns = turns[:older_turn_count]
    summary = _compact_turns(older_turns)
    store.set_session_summary(session_id, user_uuid, summary, older_turn_count)
    return store.get_session_summary(session_id)


def _candidate_facts(content: str) -> list[tuple[str, str, float]]:
    """Extract non-sensitive memory candidates from a single user utterance."""

    if _contains_sensitive_content(content):
        return []

    candidates: list[tuple[str, str, float]] = []
    name_match = re.search(r"\bmy name is ([A-Za-z][A-Za-z .'-]{1,60})", content, flags=re.I)
    if name_match:
        name = _clean_name_fragment(name_match.group(1))
        if name:
            candidates.append(("identity", f"User's name is {name}", 0.9))
    interest_match = re.search(
        r"\b(?:i am interested in|i care about|i usually ask about) ([^.?!]{3,80})",
        content,
        flags=re.I,
    )
    if interest_match:
        topic = _clean_memory_fragment(interest_match.group(1))
        if topic:
            candidates.append(("topic_interest", f"User is interested in {topic}", 0.8))
    if re.search(r"\b(?:be concise|concise answers|short answers|keep it short)\b", content, flags=re.I):
        candidates.append(("format_preference", "User prefers concise answers", 0.85))
    if re.search(r"\b(?:bullet points|bullets|list format)\b", content, flags=re.I):
        candidates.append(("format_preference", "User prefers bullet-list answers", 0.8))
    if re.search(r"\b(?:examples first|show examples first)\b", content, flags=re.I):
        candidates.append(("format_preference", "User prefers examples before explanation", 0.8))
    if re.search(r"\b(?:count before examples|counts before examples|count first)\b", content, flags=re.I):
        candidates.append(("workflow_pattern", "User usually asks for counts before examples", 0.8))
    if re.search(r"\b(refund|refunds|money back)\b", content, flags=re.I):
        candidates.append(("topic_interest", "User often explores refund-related support data", 0.75))
    if re.search(r"\b(complaint|complaints)\b", content, flags=re.I):
        candidates.append(("topic_interest", "User often explores complaint-related support data", 0.75))
    if re.search(r"\b(shipping|delivery|package)\b", content, flags=re.I):
        candidates.append(("topic_interest", "User often explores shipping and delivery support data", 0.7))
    return candidates


def _contains_sensitive_content(content: str) -> bool:
    """Avoid storing secrets, credentials, and contact details as profile memory."""

    return bool(
        re.search(r"\b(password|api key|secret|token|credential|ssn|social security)\b", content, re.I)
        or re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", content, re.I)
        or re.search(r"\b(?:\+?\d[\d .()-]{7,}\d)\b", content)
    )


def _clean_memory_fragment(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" .?!")
    cleaned = re.split(r"\b(?:and i prefer|and prefer|but i prefer|but prefer)\b", cleaned, maxsplit=1, flags=re.I)[0]
    return cleaned[:80]


def _clean_name_fragment(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip(" .?!")
    cleaned = re.split(r"\b(?:and i|and my|but i|but my|i prefer)\b", cleaned, maxsplit=1, flags=re.I)[0]
    return cleaned.strip(" .?!")[:60]


def _compact_turns(turns: list[dict[str, Any]], max_items: int = 12) -> str:
    """Build a deterministic compact summary of older session turns."""

    if not turns:
        return "No older turns have been compacted."

    user_requests: list[str] = []
    assistant_answers: list[str] = []
    for turn in turns:
        text = _clean_summary_text(turn["content"])
        if not text:
            continue
        if turn["role"] == "user":
            user_requests.append(text)
        elif turn["role"] == "assistant":
            assistant_answers.append(text)

    lines = [f"Older compacted context covers {len(turns)} persisted turns."]
    if user_requests:
        lines.append("Earlier user requests: " + "; ".join(user_requests[-max_items:]))
    if assistant_answers:
        lines.append("Earlier assistant answers: " + "; ".join(assistant_answers[-max_items:]))
    return "\n".join(lines)


def _clean_summary_text(text: str, max_chars: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."
