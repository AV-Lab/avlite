"""Tests for app-layer settings (c59_apps.yaml)."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from avlite.c50_apps.c59_settings import AppSettings
from avlite.c50_apps.c52_factory import get_stack_settings_classes, load_stack_settings
from avlite.c50_apps.c55_setting_utils import load_setting


def test_c59_settings_has_no_tkinter():
    path = Path(__file__).resolve().parents[2] / "avlite/c50_apps/c59_settings.py"
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
    (tmp_path / "c59_apps.yaml").write_text(
        yaml.dump(
            {
                "default": {
                    "c50_load_plugins": False,
                    "c50_default_plugins": ["p50_headless_mode"],
                    "c50_community_plugins": {},
                }
            }
        )
    )
    load_stack_settings(profile="default")
    assert AppSettings.c50_load_plugins is False
    assert AppSettings.c50_default_plugins == ["p50_headless_mode"]


def test_load_setting_app_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "c59_apps.yaml").write_text(
        yaml.dump({"hdmap": {"c50_selected_profile": "hdmap", "c50_next_profile": "default"}})
    )
    assert load_setting(AppSettings, profile="hdmap")
    assert AppSettings.c50_selected_profile == "hdmap"
    assert AppSettings.c50_next_profile == "default"
