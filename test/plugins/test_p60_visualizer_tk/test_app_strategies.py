"""Tests for AppStrategy registration in merged p50_visualizer_tk."""

from avlite.c50_apps.c51_app_strategy import AppStrategy, bootstrap_apps


def test_bootstrap_registers_three_tk_apps():
    bootstrap_apps()
    assert AppStrategy.registry[None].__name__ == "VisualizationApp"
    assert AppStrategy.registry["setting"].__name__ == "SettingApp"
    assert AppStrategy.registry["plugins"].__name__ == "PluginsApp"
