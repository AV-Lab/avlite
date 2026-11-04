"""VisualizationSettings sync helpers (no Qt required for model itself)."""

from __future__ import annotations

import yaml

from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_apps.c62_factory import load_stack_settings


def test_sync_stack_from_singletons(monkeypatch, tmp_path):
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
                    "c40_perception": "",
                },
            }
        )
    )
    load_stack_settings(profile="default", load_plugins=False)
    assert ExecutionSettings.c40_controller == "StanleyController"

    # Import via community package prefix if loaded, else path bootstrap.
    from avlite.c60_apps.c61_app_strategy import bootstrap_apps

    bootstrap_apps()
    from avlite.plugins.avlite_app_visualizer_qt.setting_model import VisualizationSettings

    setting = VisualizationSettings()
    setting.sync_stack_from_singletons()
    assert setting.controller_type.get() == "StanleyController"
    assert setting.execution_bridge.get() == "BasicSim"
    assert setting.perception_type.get() == ""


def test_var_bind_notifies():
    from avlite.plugins.avlite_app_visualizer_qt.setting_model import Var

    v = Var(1)
    seen = []
    v.bind(lambda: seen.append(v.get()))
    v.set(2)
    v.set(2)  # no-op
    assert seen == [2]
