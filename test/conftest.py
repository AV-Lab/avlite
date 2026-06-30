"""Shared pytest fixtures for the AVlite test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def isolated_config_dir(tmp_path, monkeypatch):
    """Redirect config paths to a temporary directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return config_dir


@pytest.fixture
def minimal_corridor_map_path() -> Path:
    """Synthetic race corridor JSON used by default planner tests."""
    return FIXTURES_DIR / "minimal_corridor.map.json"


@pytest.fixture
def minimal_opendrive_path() -> Path:
    """Minimal OpenDRIVE map for HDMap parser tests."""
    return FIXTURES_DIR / "minimal_opendrive.xodr"


@pytest.fixture(autouse=True)
def restore_execution_settings():
    """Restore ExecutionSettings after each test to prevent singleton leaks."""
    from avlite.c40_execution.c49_settings import ExecutionSettings

    snapshot = ExecutionSettings.model_dump()
    yield
    for key, value in snapshot.items():
        setattr(ExecutionSettings, key, value)
