"""Tests for syncing stack singletons into visualizer Tk settings."""

from __future__ import annotations

import tkinter as tk

import yaml

from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_apps.c62_factory import load_stack_settings
from avlite.plugins.p60_visualizer_tk.settings import VisualizationSettings, sync_stack_settings_to_ui


def test_sync_stack_settings_to_ui_reads_controller_from_profile(monkeypatch, tmp_path):
    root = tk.Tk()
    root.withdraw()
    try:
        monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
        (tmp_path / "default.yaml").write_text(
            yaml.dump(
                {
                    "c69_apps": {
                        "c62_load_plugins": False,
                        "c62_default_plugins": [],
                        "c62_community_plugins": {},
                    },
                    "c40_execution": {
                        "c40_controller": "StanleyController",
                        "c40_bridge": "BasicSim",
                    },
                }
            )
        )

        load_stack_settings(profile="default", load_plugins=False)
        assert ExecutionSettings.c40_controller == "StanleyController"

        setting = VisualizationSettings()
        registry_default = list(ControlStrategy.registry.keys())[0]
        assert setting.controller_type.get() == registry_default or setting.controller_type.get() in ControlStrategy.registry

        sync_stack_settings_to_ui(setting)
        assert setting.controller_type.get() == "StanleyController"
    finally:
        root.destroy()


def test_sync_stack_settings_preserves_empty_modules(monkeypatch, tmp_path):
    root = tk.Tk()
    root.withdraw()
    try:
        monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
        (tmp_path / "default.yaml").write_text(
            yaml.dump(
                {
                    "c69_apps": {
                        "c62_load_plugins": False,
                        "c62_default_plugins": [],
                        "c62_community_plugins": {},
                    },
                    "c40_execution": {
                        "c40_perception": "",
                        "c40_localization": "",
                        "c40_global_planner": "",
                        "c40_local_planner": "",
                        "c40_controller": "",
                        "c40_bridge": "BasicSim",
                    },
                }
            )
        )

        load_stack_settings(profile="default", load_plugins=False)
        setting = VisualizationSettings()
        sync_stack_settings_to_ui(setting)

        assert setting.perception_type.get() == ""
        assert setting.localization_type.get() == ""
        assert setting.global_planner_type.get() == ""
        assert setting.local_planner_type.get() == ""
        assert setting.controller_type.get() == ""
    finally:
        root.destroy()
