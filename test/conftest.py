"""Shared pytest fixtures for the AVlite test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

_STACK_SINGLETONS = (
    "avlite.c10_perception.c19_settings:PerceptionSettings",
    "avlite.c20_planning.c29_settings:PlanningSettings",
    "avlite.c30_control.c39_settings:ControlSettings",
    "avlite.c40_execution.c49_settings:ExecutionSettings",
    "avlite.c60_apps.c69_settings:AppSettings",
)


def _import_singleton(module_path: str, class_name: str):
    from importlib import import_module

    return getattr(import_module(module_path), class_name)


@pytest.fixture
def isolated_config_dir(tmp_path, monkeypatch):
    """Redirect config paths to a temporary directory."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
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
def restore_stack_settings():
    """Restore all stack settings singletons after each test."""
    snapshots: list[tuple[object, dict]] = []
    for spec in _STACK_SINGLETONS:
        module_path, class_name = spec.split(":")
        cls = _import_singleton(module_path, class_name)
        snapshots.append((cls, cls.model_dump()))
    yield
    for cls, snapshot in snapshots:
        for key, value in snapshot.items():
            setattr(cls, key, value)


@pytest.fixture(autouse=True)
def isolated_config_environment(tmp_path, monkeypatch):
    """Keep config-path state off the developer machine and out of repo configs/."""
    from avlite.c60_apps.c68_paths import ConfigPaths

    repo_root = Path(__file__).resolve().parent.parent
    real_bundled = repo_root / "avlite" / "configs"
    xdg = tmp_path / "pytest_xdg"
    config_dir = tmp_path / "pytest_config"
    bundled = tmp_path / "pytest_bundled"
    bundled.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    for src in ConfigPaths.iter_profile_paths(real_bundled):
        dst = bundled / src.name
        if not dst.is_file():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(ConfigPaths, "bundled_dir", staticmethod(lambda: bundled))
    ConfigPaths.set_repo_target(False)
    yield
    ConfigPaths.set_repo_target(False)
