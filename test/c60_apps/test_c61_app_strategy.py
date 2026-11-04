"""Tests for AppStrategy registration in c61_app_strategy."""

import logging
import sys

from avlite.c60_apps.c61_app_strategy import AppStrategy, bootstrap_apps
from avlite.c60_apps.c68_paths import DataPaths


def test_bootstrap_registers_three_tk_apps():
    bootstrap_apps()
    assert AppStrategy.registry[None].__name__ == "VisualizationApp"
    assert AppStrategy.registry["setting"].__name__ == "SettingApp"
    assert AppStrategy.registry["plugins"].__name__ == "PluginsApp"


def test_apps_yaml_registers_plugin_without_overriding(tmp_path, caplog):
    plugin = tmp_path / "demo-qt5"
    plugin.mkdir()
    (plugin / "__init__.py").write_text("", encoding="utf-8")
    (plugin / "app.py").write_text(
        "from avlite.c60_apps.c61_app_strategy import AppStrategy\n"
        "class DemoQtApp(AppStrategy):\n"
        "    cli_name = 'qt5'\n    help = 'Qt5 demo'\n"
        "    def run(self, args, unknown):\n        return 0\n"
        "class Hijack(AppStrategy):\n"
        "    cli_name = 'headless'\n    help = 'nope'\n"
        "    def run(self, args, unknown):\n        return 1\n",
        encoding="utf-8",
    )
    registry = DataPaths.user_dir() / "apps.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(f"qt5: {plugin}\nsetting: {plugin}\nbroken: not_a_real_module_xyz\n", encoding="utf-8")
    try:
        with caplog.at_level(logging.WARNING, logger="avlite.c60_apps.c61_app_strategy"):
            bootstrap_apps()
        assert AppStrategy.registry["qt5"].__name__ == "DemoQtApp"
        assert AppStrategy.registry["setting"].__name__ == "SettingApp"
        assert AppStrategy.registry["headless"].__name__ == "HeadlessApp"
        assert "Skipping app 'setting'" in caplog.text
        assert "Hijack" in caplog.text
        assert "Could not load app 'broken'" in caplog.text
    finally:
        AppStrategy.registry.pop("qt5", None)
        for mod in list(sys.modules):
            if mod.startswith("avlite.plugins.demo_qt5"):
                del sys.modules[mod]
