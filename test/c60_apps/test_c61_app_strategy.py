"""Tests for AppStrategy registration in c61_app_strategy."""

import logging
import sys

from avlite.c60_apps.c61_app_strategy import AppStrategy, bootstrap_apps
from avlite.c60_apps.c65_setting_utils import list_profiles
from avlite.c60_apps.c68_paths import ConfigPaths


def test_bootstrap_registers_three_tk_apps():
    bootstrap_apps()
    assert AppStrategy.registry[None].__name__ == "VisualizationApp"
    assert AppStrategy.registry["setting"].__name__ == "SettingApp"
    assert AppStrategy.registry["plugins"].__name__ == "PluginsApp"


def test_apps_yaml_registers_plugin_without_overriding(tmp_path, caplog, capsys):
    plugin = tmp_path / "demo-qt5"
    plugin.mkdir()
    (plugin / "app.py").write_text(
        "from avlite.c60_apps.c61_app_strategy import AppStrategy\n"
        "class DemoQtApp(AppStrategy):\n"
        "    cli_name = 'qt5'\n    help = 'Qt5 demo'\n    def run(self, a, u): return 0\n"
        "class Hijack(AppStrategy):\n"
        "    cli_name = 'headless'\n    help = 'nope'\n    def run(self, a, u): return 1\n",
        encoding="utf-8",
    )
    reg = ConfigPaths.user_dir() / "apps.yaml"
    reg.write_text(f"qt5: {plugin}\nsetting: {plugin}\nbroken: not_a_real_module_xyz\n", encoding="utf-8")
    try:
        with caplog.at_level(logging.WARNING, logger="avlite.c60_apps.c61_app_strategy"):
            bootstrap_apps()
        assert AppStrategy.registry["qt5"].__name__ == "DemoQtApp"
        assert AppStrategy.registry["setting"].__name__ == "SettingApp"
        assert AppStrategy.registry["headless"].__name__ == "HeadlessApp"
        assert "Hijack" in caplog.text and "broken" in caplog.text
        assert "apps" not in list_profiles()
        assert AppStrategy.registry["apps"]().run(None, None) == 0
        listed = capsys.readouterr().out
        assert "qt5" in listed and "headless" in listed and "apps" in listed
        reg.write_text(":\n[", encoding="utf-8")
        bootstrap_apps()
        assert AppStrategy.registry["plugins"].__name__ == "PluginsApp"
    finally:
        AppStrategy.registry.pop("qt5", None)
        for mod in list(sys.modules):
            if mod.startswith("avlite.plugins.demo_qt5"):
                del sys.modules[mod]
