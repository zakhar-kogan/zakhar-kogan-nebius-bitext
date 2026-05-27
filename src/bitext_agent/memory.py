"""Conversation checkpoint persistence and profile distillation helpers."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from bitext_agent.settings_store import SettingsStore, utc_now


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


def distill_profile_memory(store: SettingsStore, session_id: str, user_uuid: str) -> list[str]:
    """Distill harmless profile facts from a conversation using deterministic rules."""

    turns = store.list_turns(session_id, limit=50)
    saved: list[str] = []
    existing = {fact.fact.lower() for fact in store.list_profile_facts(user_uuid)}
    for turn in turns:
        if turn["role"] != "user":
            continue
        content = turn["content"].strip()
        candidates = _candidate_facts(content)
        for kind, fact in candidates:
            if fact.lower() in existing:
                continue
            store.add_profile_fact(
                user_uuid=user_uuid,
                kind=kind,
                fact=fact,
                source=f"session:{session_id}",
                confidence=0.75,
            )
            existing.add(fact.lower())
            saved.append(fact)
    return saved


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


def _candidate_facts(content: str) -> list[tuple[str, str]]:
    """Extract non-sensitive memory candidates from a single user utterance."""

    candidates: list[tuple[str, str]] = []
    name_match = re.search(r"\bmy name is ([A-Za-z][A-Za-z .'-]{1,60})", content, flags=re.I)
    if name_match:
        candidates.append(("name", f"User's name is {name_match.group(1).strip()}"))
    interest_match = re.search(
        r"\b(?:i am interested in|i care about|i usually ask about) ([^.?!]{3,80})",
        content,
        flags=re.I,
    )
    if interest_match:
        candidates.append(("preference", f"User is interested in {interest_match.group(1).strip()}"))
    if re.search(r"\b(refund|refunds|money back)\b", content, flags=re.I):
        candidates.append(("topic", "User often explores refund-related support data"))
    if re.search(r"\b(complaint|complaints)\b", content, flags=re.I):
        candidates.append(("topic", "User often explores complaint-related support data"))
    return candidates


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
