"""Prompt loading with SQLite overrides and versioned file defaults."""

from __future__ import annotations

from pathlib import Path

from bitext_agent.settings_store import SettingsStore


PROMPT_DIR = Path(__file__).parent / "prompts"


class PromptStore:
    """Resolve prompts from active SQLite overrides or packaged markdown files."""

    def __init__(self, settings_store: SettingsStore) -> None:
        self.settings_store = settings_store

    def load(self, name: str) -> str:
        """Load a prompt by stem name, using override first and markdown default second."""

        override = self.settings_store.get_prompt_override(name)
        if override is not None:
            return override
        path = PROMPT_DIR / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Unknown prompt: {name}")
        return path.read_text(encoding="utf-8")

