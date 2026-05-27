"""Config tests."""

from bitext_agent.config import Settings


def test_recommender_defaults_to_router() -> None:
    settings = Settings(ROUTER_MODEL="router", MAIN_MODEL="main", RECOMMENDER_MODEL="")
    assert settings.active_recommender_model == "router"

