"""Local usage logging facade."""

from __future__ import annotations

from bitext_agent.settings_store import SettingsStore


class UsageLogger:
    """Small wrapper around SQLite usage logging."""

    def __init__(self, store: SettingsStore) -> None:
        self.store = store

    def record_success(
        self,
        model: str,
        session_id: str | None = None,
        user_uuid: str | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Record a successful model call when token metadata is available."""

        self.store.log_usage(
            model=model,
            status="ok",
            session_id=session_id,
            user_uuid=user_uuid,
            total_tokens=total_tokens,
        )

    def record_error(
        self,
        model: str,
        session_id: str | None = None,
        user_uuid: str | None = None,
    ) -> None:
        """Record a failed model call without interrupting caller cleanup."""

        self.store.log_usage(model=model, status="error", session_id=session_id, user_uuid=user_uuid)

