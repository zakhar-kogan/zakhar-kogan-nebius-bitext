"""Shared pytest fixtures for the Bitext agent tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from bitext_agent.config import Settings


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Create a compact Bitext-shaped CSV for deterministic tests."""

    path = tmp_path / "bitext.csv"
    path.write_text(
        "flags,instruction,category,intent,response\n"
        'B,"I want my money back",REFUND,get_refund,"I can help process your refund."\n'
        'B,"Please refund my order",REFUND,get_refund,"Please provide your order number."\n'
        'B,"My package is late",SHIPPING,track_order,"I can help track your shipment."\n'
        'B,"I want to complain",COMPLAINT,file_complaint,"I apologize and will escalate this complaint."\n'
        'B,"I cannot access my account",ACCOUNT,recover_account,"I can help recover your account."\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def test_settings(tmp_path: Path, sample_dataset: Path) -> Settings:
    """Return isolated settings for tests."""

    return Settings(
        DATASET_PATH=sample_dataset,
        APP_DB_PATH=tmp_path / "app.sqlite",
        CHECKPOINT_DB_PATH=tmp_path / "checkpoints.sqlite",
    )

