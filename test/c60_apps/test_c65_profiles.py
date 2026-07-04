"""Tests for profile file operations and built-in plugin removal guards."""

from __future__ import annotations

import yaml

from avlite.c60_apps.c68_paths import ConfigPaths
from avlite.c60_apps.c65_setting_utils import (
    can_remove_builtin_plugin,
    delete_profile,
)


def test_delete_profile_default_is_protected(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    ConfigPaths.set_repo_target(False)
    (tmp_path / "default.yaml").write_text(yaml.dump({"c69_apps": {}}))

    assert delete_profile("default") is False
    assert (tmp_path / "default.yaml").is_file()


def test_delete_profile_custom_removes_file(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    ConfigPaths.set_repo_target(False)
    (tmp_path / "robot.yaml").write_text(yaml.dump({"c69_apps": {}}))

    assert delete_profile("robot") is True
    assert not (tmp_path / "robot.yaml").exists()


def test_delete_profile_missing_returns_false(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    ConfigPaths.set_repo_target(False)
    assert delete_profile("missing") is False


def test_can_remove_builtin_plugin_blocks_hosting_plugin():
    msg = can_remove_builtin_plugin("p60_visualizer_tk", "p60_visualizer_tk")
    assert msg is not None
    assert "p60_visualizer_tk" in msg


def test_can_remove_builtin_plugin_allows_other_plugins():
    assert can_remove_builtin_plugin("p60_headless_mode", "p60_visualizer_tk") is None
    assert can_remove_builtin_plugin("p60_setting_cli", "p60_visualizer_tk") is None
