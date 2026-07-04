"""Tests for community plugin settings paths and import hook."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pytest
import yaml

from avlite.c50_apps.c53_plugins import (
    find_community_plugin_dir,
    import_plugin_modules,
    load_community_plugin_setting,
    plugin_module_prefix,
    register_community_plugin_import_hook,
)
from avlite.c50_apps.c58_paths import ConfigPaths, PluginPaths
from avlite.c50_apps.c55_setting_utils import load_setting, stored_filepath

_PLUGIN_NAME = "avlite-executer-ROS2"
_SETTINGS_BODY = (
    "class PluginSettings:\n"
    "    filepath = ''\n"
    "    replan_dt = 0.1\n"
    "    control_dt = 0.02\n"
)


def _clear_plugin_modules(name: str) -> None:
    prefix = plugin_module_prefix(name)
    for mod_name in list(sys.modules):
        if mod_name == prefix or mod_name.startswith(prefix + "."):
            del sys.modules[mod_name]


@pytest.fixture
def dashed_plugin(tmp_path):
    """Minimal community plugin with dashes in the directory name."""
    plugin_dir = tmp_path / "plugins" / _PLUGIN_NAME
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text('"""test plugin"""\n', encoding="utf-8")
    (plugin_dir / "settings.py").write_text(_SETTINGS_BODY, encoding="utf-8")
    _clear_plugin_modules(_PLUGIN_NAME)
    yield plugin_dir
    _clear_plugin_modules(_PLUGIN_NAME)


def test_import_plugin_modules_patches_dashed_settings_filepath(dashed_plugin, monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    import_plugin_modules(str(dashed_plugin), pkg_name=_PLUGIN_NAME)

    from avlite.plugins.avlite_executer_ROS2.settings import PluginSettings

    expected = PluginPaths.settings_filepath(_PLUGIN_NAME)
    assert stored_filepath(PluginSettings) == expected
    assert expected == "configs/plugin_avlite-executer-ROS2.yaml"


def test_load_setting_uses_dashed_yaml_after_import(dashed_plugin, monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    user_yaml = tmp_path / "plugin_avlite-executer-ROS2.yaml"
    user_yaml.write_text(
        yaml.dump({"default": {"replan_dt": 0.42, "control_dt": 0.07}}),
        encoding="utf-8",
    )

    import_plugin_modules(str(dashed_plugin), pkg_name=_PLUGIN_NAME)
    from avlite.plugins.avlite_executer_ROS2.settings import PluginSettings

    assert load_setting(PluginSettings, profile="default") is True
    assert PluginSettings.replan_dt == pytest.approx(0.42)
    assert PluginSettings.control_dt == pytest.approx(0.07)


def test_load_community_plugin_setting_reuses_imported_singleton(dashed_plugin, monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    import_plugin_modules(str(dashed_plugin), pkg_name=_PLUGIN_NAME)
    from avlite.plugins.avlite_executer_ROS2.settings import PluginSettings as module_ps

    cls = load_community_plugin_setting(_PLUGIN_NAME, str(dashed_plugin), profile="default")
    assert cls is module_ps


def test_load_setting_missing_file_is_non_fatal(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))

    class _Settings:
        filepath = "configs/plugin_missing_xyz_test.yaml"

    with caplog.at_level(logging.DEBUG):
        assert load_setting(_Settings, profile="default") is False

    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_community_plugin_finder_imports_from_install_dir(monkeypatch, tmp_path):
    plugins_dir = tmp_path / "plugins"
    plugin_dir = plugins_dir / _PLUGIN_NAME
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text('"""test plugin"""\n', encoding="utf-8")
    (plugin_dir / "settings.py").write_text(
        "class PluginSettings:\n    filepath = ''\n    value = 1\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins_dir))
    _clear_plugin_modules(_PLUGIN_NAME)
    register_community_plugin_import_hook()

    mod = importlib.import_module(f"{plugin_module_prefix(_PLUGIN_NAME)}.settings")
    assert mod.PluginSettings.value == 1
    _clear_plugin_modules(_PLUGIN_NAME)


def test_find_community_plugin_dir_from_install_dir(monkeypatch, tmp_path):
    install = tmp_path / "plugins" / _PLUGIN_NAME
    install.mkdir(parents=True)
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "plugins"))

    found = find_community_plugin_dir("avlite_executer_ROS2")
    assert found == install.resolve()


def test_find_community_plugin_dir_from_community_dev(monkeypatch, tmp_path):
    dev = tmp_path / "avlite-community-plugins" / _PLUGIN_NAME
    dev.mkdir(parents=True)
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "empty_plugins"))
    (tmp_path / "empty_plugins").mkdir()

    repo_root = tmp_path
    monkeypatch.setattr(PluginPaths, "repo_root", staticmethod(lambda: repo_root))

    found = find_community_plugin_dir("avlite_executer_ROS2")
    assert found == dev.resolve()


def test_load_stack_settings_imports_community_plugins(dashed_plugin, monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "c59_apps.yaml").write_text(
        yaml.dump(
            {
                "default": {
                    "c52_community_plugins": {
                        _PLUGIN_NAME: str(dashed_plugin),
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    from avlite.c50_apps.c52_factory import load_stack_settings

    _clear_plugin_modules(_PLUGIN_NAME)
    load_stack_settings(profile="default", load_plugins=True)

    prefix = plugin_module_prefix(_PLUGIN_NAME)
    assert f"{prefix}.settings" in sys.modules
