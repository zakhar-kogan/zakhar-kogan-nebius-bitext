"""SQLite-backed app state for users, profile facts, prompts, logs, and recommendations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bitext_agent.schemas import ProfileFact


def utc_now() -> str:
    """Return a stable UTC timestamp string for SQLite rows."""

    return datetime.now(UTC).isoformat()


def normalize_recommendation_query(query: str) -> str:
    """Normalize recommendation text for case-insensitive session rotation."""

    return " ".join(query.strip().lower().split())


class SettingsStore:
    """Durable SQLite store for app state outside LangGraph checkpoints."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with row dictionaries enabled."""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create all app-state tables if they do not exist."""

        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists users (
                    user_uuid text primary key,
                    external_user_id text unique,
                    created_at text not null,
                    updated_at text not null
                );
                create table if not exists profile_facts (
                    id integer primary key autoincrement,
                    user_uuid text not null,
                    kind text not null,
                    fact text not null,
                    canonical_key text,
                    source text not null,
                    confidence real not null,
                    status text not null,
                    created_at text not null,
                    updated_at text not null,
                    foreign key(user_uuid) references users(user_uuid)
                );
                create table if not exists prompt_overrides (
                    name text primary key,
                    content text not null,
                    active integer not null default 1,
                    updated_at text not null
                );
                create table if not exists llm_usage (
                    id integer primary key autoincrement,
                    session_id text,
                    user_uuid text,
                    model text not null,
                    prompt_tokens integer,
                    completion_tokens integer,
                    total_tokens integer,
                    latency_ms integer,
                    status text not null,
                    estimated_cost real,
                    raw_usage_metadata_json text,
                    created_at text not null
                );
                create table if not exists cached_recommendations (
                    cache_key text primary key,
                    recommendations_json text not null,
                    updated_at text not null
                );
                create table if not exists pending_recommendations (
                    session_id text primary key,
                    user_uuid text not null,
                    query text not null,
                    reason text not null,
                    updated_at text not null
                );
                create table if not exists selected_recommendations (
                    session_id text not null,
                    normalized_query text not null,
                    query text not null,
                    selected_at text not null,
                    primary key(session_id, normalized_query)
                );
                create table if not exists recommendation_slots (
                    session_id text not null,
                    slot_index integer not null,
                    query text not null,
                    updated_at text not null,
                    primary key(session_id, slot_index)
                );
                create table if not exists conversation_turns (
                    id integer primary key autoincrement,
                    session_id text not null,
                    user_uuid text not null,
                    role text not null,
                    content text not null,
                    metadata_json text not null default '{}',
                    created_at text not null
                );
                create table if not exists session_summaries (
                    session_id text primary key,
                    user_uuid text not null,
                    summary text not null,
                    source_turn_count integer not null,
                    updated_at text not null
                );
                create table if not exists tool_calls (
                    id integer primary key autoincrement,
                    session_id text not null,
                    user_uuid text not null,
                    tool_name text not null,
                    status text not null,
                    latency_ms integer,
                    error text,
                    created_at text not null
                );
                """
            )
            self._ensure_column(conn, "profile_facts", "canonical_key", "text")
            self._ensure_column(conn, "llm_usage", "raw_usage_metadata_json", "text")
            conn.execute(
                """
                create index if not exists idx_profile_facts_user_canonical
                on profile_facts(user_uuid, canonical_key, status)
                """
            )

    def get_or_create_user(self, external_user_id: str | None) -> tuple[str, str]:
        """Resolve an external user ID to an internal UUID, creating it when needed."""

        ext_id = external_user_id or f"local-{uuid.uuid4()}"
        with self.connect() as conn:
            row = conn.execute(
                "select user_uuid, external_user_id from users where external_user_id = ?", (ext_id,)
            ).fetchone()
            if row:
                return row["user_uuid"], row["external_user_id"]
            user_uuid = str(uuid.uuid4())
            now = utc_now()
            conn.execute(
                "insert into users(user_uuid, external_user_id, created_at, updated_at) values (?, ?, ?, ?)",
                (user_uuid, ext_id, now, now),
            )
            return user_uuid, ext_id

    def add_profile_fact(
        self,
        user_uuid: str,
        kind: str,
        fact: str,
        source: str,
        confidence: float = 0.5,
        status: str = "active",
        canonical_key: str | None = None,
    ) -> int:
        """Add a distilled profile fact for a user."""

        now = utc_now()
        with self.connect() as conn:
            cursor = conn.execute(
                """
                insert into profile_facts(user_uuid, kind, fact, canonical_key, source, confidence,
                status, created_at, updated_at)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_uuid, kind, fact, canonical_key, source, confidence, status, now, now),
            )
            return int(cursor.lastrowid)

    def upsert_profile_fact(
        self,
        user_uuid: str,
        kind: str,
        fact: str,
        canonical_key: str,
        source: str,
        confidence: float = 0.75,
    ) -> tuple[int, bool]:
        """Create or refresh one active canonical profile fact."""

        now = utc_now()
        with self.connect() as conn:
            existing = conn.execute(
                """
                select id, confidence from profile_facts
                where user_uuid = ? and canonical_key = ? and status = 'active'
                order by confidence desc, updated_at desc, id desc
                """,
                (user_uuid, canonical_key),
            ).fetchall()
            if existing:
                keeper = existing[0]
                conn.execute(
                    """
                    update profile_facts
                    set kind = ?, fact = ?, source = ?, confidence = max(confidence, ?), updated_at = ?
                    where id = ?
                    """,
                    (kind, fact, source, confidence, now, int(keeper["id"])),
                )
                duplicate_ids = [int(row["id"]) for row in existing[1:]]
                if duplicate_ids:
                    placeholders = ",".join("?" for _ in duplicate_ids)
                    conn.execute(
                        f"""
                        update profile_facts
                        set status = 'duplicate', updated_at = ?
                        where id in ({placeholders})
                        """,
                        (now, *duplicate_ids),
                    )
                return int(keeper["id"]), False

            cursor = conn.execute(
                """
                insert into profile_facts(user_uuid, kind, fact, canonical_key, source, confidence,
                status, created_at, updated_at)
                values (?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """,
                (user_uuid, kind, fact, canonical_key, source, confidence, now, now),
            )
            return int(cursor.lastrowid), True

    def update_profile_fact(
        self,
        fact_id: int,
        kind: str,
        fact: str,
        canonical_key: str,
        source: str,
        confidence: float,
    ) -> None:
        """Replace one active profile fact with a stronger equivalent fact."""

        with self.connect() as conn:
            conn.execute(
                """
                update profile_facts
                set kind = ?, fact = ?, canonical_key = ?, source = ?, confidence = ?, updated_at = ?
                where id = ? and status = 'active'
                """,
                (kind, fact, canonical_key, source, confidence, utc_now(), fact_id),
            )

    def update_profile_fact_canonical_key(self, fact_id: int, canonical_key: str) -> None:
        """Backfill a missing or stale canonical key for one profile fact."""

        with self.connect() as conn:
            conn.execute(
                """
                update profile_facts
                set canonical_key = ?, updated_at = ?
                where id = ?
                """,
                (canonical_key, utc_now(), fact_id),
            )

    def mark_profile_facts_duplicate(self, fact_ids: list[int]) -> int:
        """Soft-mark profile facts as duplicates."""

        if not fact_ids:
            return 0
        placeholders = ",".join("?" for _ in fact_ids)
        with self.connect() as conn:
            cursor = conn.execute(
                f"""
                update profile_facts
                set status = 'duplicate', updated_at = ?
                where id in ({placeholders})
                """,
                (utc_now(), *fact_ids),
            )
            return int(cursor.rowcount)

    def prune_profile_facts(self, user_uuid: str, max_active: int = 30) -> int:
        """Soft-delete old, low-confidence facts beyond the active cap."""

        with self.connect() as conn:
            rows = conn.execute(
                """
                select id from profile_facts
                where user_uuid = ? and status = 'active'
                order by confidence desc, updated_at desc, id desc
                """,
                (user_uuid,),
            ).fetchall()
            stale_ids = [int(row["id"]) for row in rows[max_active:]]
            if not stale_ids:
                return 0
            placeholders = ",".join("?" for _ in stale_ids)
            conn.execute(
                f"""
                update profile_facts
                set status = 'pruned', updated_at = ?
                where id in ({placeholders})
                """,
                (utc_now(), *stale_ids),
            )
            return len(stale_ids)

    def list_profile_facts(self, user_uuid: str, include_inactive: bool = False) -> list[ProfileFact]:
        """Return profile facts for a user."""

        sql = "select * from profile_facts where user_uuid = ?"
        params: tuple[Any, ...] = (user_uuid,)
        if not include_inactive:
            sql += " and status = 'active'"
        sql += " order by updated_at desc, id desc"
        with self.connect() as conn:
            return [self._fact_from_row(row) for row in conn.execute(sql, params).fetchall()]

    def delete_profile_fact(self, fact_id: int, user_uuid: str | None = None) -> None:
        """Soft-delete a profile fact."""

        now = utc_now()
        with self.connect() as conn:
            if user_uuid:
                conn.execute(
                    "update profile_facts set status = 'deleted', updated_at = ? where id = ? and user_uuid = ?",
                    (now, fact_id, user_uuid),
                )
            else:
                conn.execute(
                    "update profile_facts set status = 'deleted', updated_at = ? where id = ?",
                    (now, fact_id),
                )

    def get_prompt_override(self, name: str) -> str | None:
        """Return an active prompt override if one exists."""

        with self.connect() as conn:
            row = conn.execute(
                "select content from prompt_overrides where name = ? and active = 1", (name,)
            ).fetchone()
            return str(row["content"]) if row else None

    def set_prompt_override(self, name: str, content: str, active: bool = True) -> None:
        """Create or replace a prompt override."""

        with self.connect() as conn:
            conn.execute(
                """
                insert into prompt_overrides(name, content, active, updated_at) values (?, ?, ?, ?)
                on conflict(name) do update set content = excluded.content, active = excluded.active,
                updated_at = excluded.updated_at
                """,
                (name, content, int(active), utc_now()),
            )

    def log_usage(
        self,
        model: str,
        status: str,
        session_id: str | None = None,
        user_uuid: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: int | None = None,
        estimated_cost: float | None = None,
        raw_usage_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write best-effort LLM usage metadata."""

        with self.connect() as conn:
            conn.execute(
                """
                insert into llm_usage(session_id, user_uuid, model, prompt_tokens, completion_tokens,
                total_tokens, latency_ms, status, estimated_cost, raw_usage_metadata_json, created_at)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_uuid,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    latency_ms,
                    status,
                    estimated_cost,
                    json.dumps(raw_usage_metadata) if raw_usage_metadata else None,
                    utc_now(),
                ),
            )

    def usage_summary(
        self, session_id: str | None = None, user_uuid: str | None = None
    ) -> dict[str, Any]:
        """Return compact LLM usage diagnostics, optionally scoped."""

        with self.connect() as conn:
            where, params = self._scope_clause(session_id=session_id, user_uuid=user_uuid)
            row = conn.execute(
                f"""
                select count(*) calls,
                       coalesce(sum(total_tokens), 0) tokens,
                       coalesce(sum(prompt_tokens), 0) prompt_tokens,
                       coalesce(sum(completion_tokens), 0) completion_tokens,
                       count(total_tokens) rows_with_tokens,
                       coalesce(avg(total_tokens), 0) avg_tokens,
                       coalesce(avg(latency_ms), 0) avg_latency_ms,
                       coalesce(sum(latency_ms), 0) total_time_ms
                from llm_usage
                {where}
                """,
                params,
            ).fetchone()
            return {
                "calls": int(row["calls"]),
                "llm_calls": int(row["calls"]),
                "tokens": int(row["tokens"]),
                "prompt_tokens": int(row["prompt_tokens"]),
                "completion_tokens": int(row["completion_tokens"]),
                "rows_with_tokens": int(row["rows_with_tokens"]),
                "avg_tokens": round(float(row["avg_tokens"] or 0), 1),
                "avg_latency_ms": round(float(row["avg_latency_ms"] or 0), 1),
                "total_time_ms": int(row["total_time_ms"] or 0),
                "ttft": "UNAVAILABLE: calls are non-streaming",
            }

    def usage_by_session(self, user_uuid: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Return LLM usage grouped by session."""

        with self.connect() as conn:
            where, params = self._scope_clause(user_uuid=user_uuid)
            rows = conn.execute(
                f"""
                select session_id,
                       count(*) llm_calls,
                       coalesce(sum(total_tokens), 0) tokens,
                       coalesce(avg(total_tokens), 0) avg_tokens,
                       coalesce(avg(latency_ms), 0) avg_latency_ms,
                       coalesce(sum(latency_ms), 0) total_time_ms,
                       min(created_at) first_call,
                       max(created_at) last_call
                from llm_usage
                {where}
                group by session_id
                order by tokens desc, last_call desc
                limit ?
                """,
                (*params, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def recent_usage(
        self, session_id: str | None = None, user_uuid: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return recent LLM call diagnostics."""

        with self.connect() as conn:
            where, params = self._scope_clause(session_id=session_id, user_uuid=user_uuid)
            rows = conn.execute(
                f"""
                select session_id, model, status, prompt_tokens, completion_tokens,
                       total_tokens, latency_ms, raw_usage_metadata_json, created_at
                from llm_usage
                {where}
                order by id desc
                limit ?
                """,
                (*params, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def log_tool_call(
        self,
        session_id: str,
        user_uuid: str,
        tool_name: str,
        status: str,
        latency_ms: int | None = None,
        error: str | None = None,
    ) -> None:
        """Write tool-call diagnostics without storing tool payloads."""

        with self.connect() as conn:
            conn.execute(
                """
                insert into tool_calls(session_id, user_uuid, tool_name, status, latency_ms, error, created_at)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_uuid, tool_name, status, latency_ms, error, utc_now()),
            )

    def tool_usage_summary(
        self, session_id: str | None = None, user_uuid: str | None = None
    ) -> dict[str, Any]:
        """Return compact tool-call diagnostics, optionally scoped."""

        with self.connect() as conn:
            where, params = self._scope_clause(session_id=session_id, user_uuid=user_uuid)
            row = conn.execute(
                f"""
                select count(*) tool_calls,
                       coalesce(sum(case when status = 'error' then 1 else 0 end), 0) tool_errors,
                       coalesce(avg(latency_ms), 0) avg_latency_ms,
                       coalesce(sum(latency_ms), 0) total_time_ms
                from tool_calls
                {where}
                """,
                params,
            ).fetchone()
            return {
                "tool_calls": int(row["tool_calls"]),
                "tool_errors": int(row["tool_errors"]),
                "avg_latency_ms": round(float(row["avg_latency_ms"] or 0), 1),
                "total_time_ms": int(row["total_time_ms"] or 0),
            }

    def tool_calls_by_name(
        self, session_id: str | None = None, user_uuid: str | None = None
    ) -> list[dict[str, Any]]:
        """Return tool-call counts grouped by tool name."""

        with self.connect() as conn:
            where, params = self._scope_clause(session_id=session_id, user_uuid=user_uuid)
            rows = conn.execute(
                f"""
                select tool_name,
                       count(*) calls,
                       coalesce(sum(case when status = 'error' then 1 else 0 end), 0) errors,
                       coalesce(avg(latency_ms), 0) avg_latency_ms
                from tool_calls
                {where}
                group by tool_name
                order by calls desc, tool_name asc
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_cached_recommendations(self, cache_key: str) -> list[str] | None:
        """Return cached starter recommendations for a dataset fingerprint."""

        with self.connect() as conn:
            row = conn.execute(
                "select recommendations_json from cached_recommendations where cache_key = ?", (cache_key,)
            ).fetchone()
            if not row:
                return None
            return list(json.loads(row["recommendations_json"]))

    def set_cached_recommendations(self, cache_key: str, recommendations: list[str]) -> None:
        """Persist starter recommendations for reuse."""

        with self.connect() as conn:
            conn.execute(
                """
                insert into cached_recommendations(cache_key, recommendations_json, updated_at)
                values (?, ?, ?)
                on conflict(cache_key) do update set recommendations_json = excluded.recommendations_json,
                updated_at = excluded.updated_at
                """,
                (cache_key, json.dumps(recommendations), utc_now()),
            )

    def get_pending_recommendation(self, session_id: str) -> dict[str, str] | None:
        """Return a pending recommendation for a session if present."""

        with self.connect() as conn:
            row = conn.execute(
                "select query, reason from pending_recommendations where session_id = ?", (session_id,)
            ).fetchone()
            return {"query": row["query"], "reason": row["reason"]} if row else None

    def set_pending_recommendation(
        self, session_id: str, user_uuid: str, query: str, reason: str
    ) -> None:
        """Create or replace the pending query recommendation for a session."""

        with self.connect() as conn:
            conn.execute(
                """
                insert into pending_recommendations(session_id, user_uuid, query, reason, updated_at)
                values (?, ?, ?, ?, ?)
                on conflict(session_id) do update set user_uuid = excluded.user_uuid,
                query = excluded.query, reason = excluded.reason, updated_at = excluded.updated_at
                """,
                (session_id, user_uuid, query, reason, utc_now()),
            )

    def clear_pending_recommendation(self, session_id: str) -> None:
        """Clear a pending recommendation after execution or cancellation."""

        with self.connect() as conn:
            conn.execute("delete from pending_recommendations where session_id = ?", (session_id,))

    def record_selected_recommendation(self, session_id: str, query: str) -> None:
        """Record a recommendation button selected during the current session."""

        normalized_query = normalize_recommendation_query(query)
        if not normalized_query:
            return
        with self.connect() as conn:
            conn.execute(
                """
                insert into selected_recommendations(session_id, normalized_query, query, selected_at)
                values (?, ?, ?, ?)
                on conflict(session_id, normalized_query) do update set
                query = excluded.query, selected_at = excluded.selected_at
                """,
                (session_id, normalized_query, query, utc_now()),
            )

    def list_selected_recommendation_keys(self, session_id: str) -> set[str]:
        """Return normalized recommendation queries already selected in a session."""

        with self.connect() as conn:
            rows = conn.execute(
                "select normalized_query from selected_recommendations where session_id = ?",
                (session_id,),
            ).fetchall()
        return {row["normalized_query"] for row in rows}

    def list_selected_recommendation_queries(self, session_id: str) -> list[str]:
        """Return selected recommendation query text for fuzzy duplicate checks."""

        with self.connect() as conn:
            rows = conn.execute(
                "select query from selected_recommendations where session_id = ? order by selected_at asc",
                (session_id,),
            ).fetchall()
        return [str(row["query"]) for row in rows]

    def list_recommendation_slots(self, session_id: str) -> list[dict[str, object]]:
        """Return currently visible recommendation slots for a session."""

        with self.connect() as conn:
            rows = conn.execute(
                """
                select slot_index, query, updated_at from recommendation_slots
                where session_id = ?
                order by slot_index asc
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_recommendation_slot(self, session_id: str, slot_index: int, query: str | None) -> None:
        """Set or clear one visible recommendation slot."""

        with self.connect() as conn:
            if query is None:
                conn.execute(
                    "delete from recommendation_slots where session_id = ? and slot_index = ?",
                    (session_id, slot_index),
                )
                return
            conn.execute(
                """
                insert into recommendation_slots(session_id, slot_index, query, updated_at)
                values (?, ?, ?, ?)
                on conflict(session_id, slot_index) do update set
                query = excluded.query, updated_at = excluded.updated_at
                """,
                (session_id, slot_index, query, utc_now()),
            )

    def add_turn(
        self,
        session_id: str,
        user_uuid: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Append a conversation turn for recommendations and fallback history."""

        with self.connect() as conn:
            conn.execute(
                """
                insert into conversation_turns(session_id, user_uuid, role, content, metadata_json, created_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_uuid, role, content, json.dumps(metadata or {}), utc_now()),
            )

    def list_user_sessions(self, user_uuid: str) -> list[dict[str, Any]]:
        """Return conversation sessions that have turns for a user."""

        with self.connect() as conn:
            rows = conn.execute(
                """
                select session_id,
                       count(*) turns,
                       sum(case when role = 'user' then 1 else 0 end) user_turns,
                       min(created_at) first_turn,
                       max(created_at) last_turn
                from conversation_turns
                where user_uuid = ?
                group by session_id
                order by last_turn desc, session_id asc
                """,
                (user_uuid,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_turns(
        self, session_id: str, limit: int = 20, user_uuid: str | None = None
    ) -> list[dict[str, Any]]:
        """Return recent conversation turns in chronological order."""

        sql = """
                select id, user_uuid, role, content, metadata_json, created_at from conversation_turns
                where session_id = ?
                """
        params: tuple[Any, ...] = (session_id,)
        if user_uuid is not None:
            sql += " and user_uuid = ?"
            params = (session_id, user_uuid)
        sql += " order by id desc limit ?"
        with self.connect() as conn:
            rows = conn.execute(
                sql,
                (*params, limit),
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "user_uuid": row["user_uuid"],
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]

    def count_turns(self, session_id: str) -> int:
        """Return the number of persisted turns for a session."""

        with self.connect() as conn:
            row = conn.execute(
                "select count(*) turns from conversation_turns where session_id = ?", (session_id,)
            ).fetchone()
        return int(row["turns"])

    def count_user_turns(self, session_id: str, user_uuid: str | None = None) -> int:
        """Return the number of persisted user turns for a session."""

        sql = "select count(*) turns from conversation_turns where session_id = ? and role = 'user'"
        params: tuple[Any, ...] = (session_id,)
        if user_uuid is not None:
            sql += " and user_uuid = ?"
            params = (session_id, user_uuid)
        with self.connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return int(row["turns"])

    def get_session_summary(self, session_id: str) -> dict[str, Any] | None:
        """Return a cached compact summary for a session if one exists."""

        with self.connect() as conn:
            row = conn.execute(
                """
                select session_id, user_uuid, summary, source_turn_count, updated_at
                from session_summaries where session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def set_session_summary(
        self, session_id: str, user_uuid: str, summary: str, source_turn_count: int
    ) -> None:
        """Create or replace the cached compact summary for a session."""

        with self.connect() as conn:
            conn.execute(
                """
                insert into session_summaries(session_id, user_uuid, summary, source_turn_count, updated_at)
                values (?, ?, ?, ?, ?)
                on conflict(session_id) do update set user_uuid = excluded.user_uuid,
                summary = excluded.summary, source_turn_count = excluded.source_turn_count,
                updated_at = excluded.updated_at
                """,
                (session_id, user_uuid, summary, source_turn_count, utc_now()),
            )

    def _scope_clause(
        self, session_id: str | None = None, user_uuid: str | None = None
    ) -> tuple[str, tuple[Any, ...]]:
        clauses: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if user_uuid is not None:
            clauses.append("user_uuid = ?")
            params.append(user_uuid)
        return ("where " + " and ".join(clauses) if clauses else "", tuple(params))

    def _ensure_column(
        self, conn: sqlite3.Connection, table: str, column: str, declaration: str
    ) -> None:
        columns = {row["name"] for row in conn.execute(f"pragma table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"alter table {table} add column {column} {declaration}")

    def _fact_from_row(self, row: sqlite3.Row) -> ProfileFact:
        return ProfileFact(
            id=int(row["id"]),
            user_uuid=row["user_uuid"],
            kind=row["kind"],
            fact=row["fact"],
            canonical_key=row["canonical_key"],
            source=row["source"],
            confidence=float(row["confidence"]),
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
