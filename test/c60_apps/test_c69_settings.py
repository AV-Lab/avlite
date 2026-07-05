"""Tests for app-layer settings (c69_apps section of per-profile YAML)."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from avlite.c60_apps.c69_settings import AppSettings
from avlite.c60_apps.c62_factory import get_stack_settings_classes, load_stack_settings
from avlite.c60_apps.c68_paths import ConfigPaths
from avlite.c60_apps.c65_setting_utils import load_setting


def test_c69_settings_has_no_tkinter():
    path = Path(__file__).resolve().parents[2] / "avlite/c60_apps/c69_settings.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "tkinter"
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "tkinter"


def test_app_settings_in_stack_export():
    classes = get_stack_settings_classes()
    assert AppSettings in classes


def test_load_stack_settings_reads_app_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "default.yaml").write_text(
        yaml.dump(
            {
                "c69_apps": {
                    "c62_load_plugins": False,
                    "c62_default_plugins": ["p60_headless_mode"],
                    "c62_community_plugins": {},
                }
            }
        )
    )
    load_stack_settings(profile="default")
    assert AppSettings.c62_load_plugins is False
    assert AppSettings.c62_default_plugins == ["p60_headless_mode"]


def test_load_setting_app_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "hdmap.yaml").write_text(
        yaml.dump({"c69_apps": {"c60_selected_profile": "hdmap"}})
    )
    assert load_setting(AppSettings, profile="hdmap")
    assert AppSettings.c60_selected_profile == "hdmap"


def test_load_setting_empty_plugin_section_uses_schema_defaults(monkeypatch, tmp_path):
    from avlite.plugins.p60_visualizer_tk.settings import PluginSettings, PluginSettingsSchema

    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    ConfigPaths.set_repo_target(False)
    (tmp_path / "carla.yaml").write_text(
        yaml.dump({"plugins": {"p60_visualizer_tk": {}}})
    )
    expected = PluginSettingsSchema.model_validate({})
    PluginSettings.p67_global_plan_view = not expected.p67_global_plan_view

    assert load_setting(PluginSettings, profile="carla") is True
    assert PluginSettings.p67_global_plan_view == expected.p67_global_plan_view
    assert PluginSettings.p67_local_plan_view == expected.p67_local_plan_view
    assert PluginSettings.p67_global_plan_view is True
    assert PluginSettings.p67_local_plan_view is True
    assert PluginSettings.p66_show_global_plan == expected.p66_show_global_plan
