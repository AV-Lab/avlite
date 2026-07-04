"""Tests for AppStrategy registration in merged p60_visualizer_tk."""

from avlite.c60_apps.c61_app_strategy import AppStrategy, bootstrap_apps


def test_visualizer_settings_module_imports():
    import avlite.plugins.p60_visualizer_tk.settings  # noqa: F401


def test_bootstrap_registers_three_tk_apps():
    bootstrap_apps()
    assert AppStrategy.registry[None].__name__ == "VisualizationApp"
    assert AppStrategy.registry["setting"].__name__ == "SettingApp"
    assert AppStrategy.registry["plugins"].__name__ == "PluginsApp"
