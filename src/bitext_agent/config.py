"""Application configuration loaded from environment variables."""

from pathlib import Path
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


MemoryMode = Literal["per_conversation", "every_n_turns", "per_turn"]


class Settings(BaseSettings):
    """Runtime settings for the Bitext data analyst agent."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    nebius_api_key: str = Field(default="", alias="NEBIUS_API_KEY")
    nebius_base_url: str = Field(
        default="https://api.studio.nebius.com/v1/", alias="NEBIUS_BASE_URL"
    )
    router_model: str = Field(default="nvidia/Nemotron-3-Nano-Omni", alias="ROUTER_MODEL")
    main_model: str = Field(default="MiniMaxAI/MiniMax-M2.5-fast", alias="MAIN_MODEL")
    recommender_model: str = Field(default="", alias="RECOMMENDER_MODEL")

    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_project: str = Field(
        default="bitext-data-analyst-agent", alias="LANGSMITH_PROJECT"
    )

    dataset_path: Path = Field(
        default=Path("data/raw/bitext_customer_support.csv"), alias="DATASET_PATH"
    )
    checkpoint_db_path: Path = Field(
        default=Path("data/state/checkpoints.sqlite"), alias="CHECKPOINT_DB_PATH"
    )
    app_db_path: Path = Field(default=Path("data/state/app.sqlite"), alias="APP_DB_PATH")

    max_agent_iterations: int = Field(default=12, ge=1, le=50, alias="MAX_AGENT_ITERATIONS")
    memory_distillation_mode: MemoryMode = Field(
        default="per_conversation", alias="MEMORY_DISTILLATION_MODE"
    )
    memory_distillation_turn_interval: int = Field(
        default=3, ge=1, alias="MEMORY_DISTILLATION_TURN_INTERVAL"
    )
    session_compaction_turn_threshold: int = Field(
        default=16, ge=4, alias="SESSION_COMPACTION_TURN_THRESHOLD"
    )
    session_recent_turn_limit: int = Field(default=12, ge=2, alias="SESSION_RECENT_TURN_LIMIT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @computed_field
    @property
    def active_recommender_model(self) -> str:
        """Return the configured recommender model, falling back to the router model."""

        return self.recommender_model or self.router_model


def get_settings() -> Settings:
    """Load settings from environment variables and `.env` if present."""

    settings = Settings()
    settings.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    settings.app_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
