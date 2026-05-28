"""Config tests."""

from bitext_agent.config import Settings


def test_recommender_defaults_to_router() -> None:
    settings = Settings(ROUTER_MODEL="router", MAIN_MODEL="main", RECOMMENDER_MODEL="")
    assert settings.active_recommender_model == "router"


def test_low_token_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.max_agent_iterations == 10
    assert settings.memory_distillation_mode == "every_n_turns"
    assert settings.session_recent_turn_limit == 6
