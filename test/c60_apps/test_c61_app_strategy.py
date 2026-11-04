"""Tests for AppStrategy registration in c61_app_strategy."""

import logging
import os
import sys
from pathlib import Path

from avlite.c60_apps.c61_app_strategy import AppStrategy, bootstrap_apps
from avlite.c60_apps.c65_setting_utils import list_profiles


def test_bootstrap_registers_three_tk_apps():
    bootstrap_apps()
    assert AppStrategy.registry[None].__name__ == "VisualizationApp"
    assert AppStrategy.registry["setting"].__name__ == "SettingApp"
    assert AppStrategy.registry["plugins"].__name__ == "PluginsApp"


def test_apps_yaml_is_not_a_profile():
    path = Path(os.environ["AVLITE_CONFIG_DIR"]) / "apps.yaml"
    path.write_text("qt5: /tmp/not-installed\n", encoding="utf-8")
    assert "apps" not in list_profiles()


def test_apps_yaml_registers_plugin_without_profile_or_override(tmp_path, caplog):
    plugin = tmp_path / "demo-qt5"
    plugin.mkdir()
    (plugin / "__init__.py").write_text("", encoding="utf-8")
    (plugin / "app.py").write_text(
        "from avlite.c60_apps.c61_app_strategy import AppStrategy\n"
        "class DemoQtApp(AppStrategy):\n"
        "    cli_name = 'qt5'\n"
        "    help = 'Qt5 demo app'\n"
        "    def run(self, args, unknown):\n"
        "        return 0\n",
        encoding="utf-8",
    )
    evil = tmp_path / "evil"
    evil.mkdir()
    (evil / "__init__.py").write_text("", encoding="utf-8")
    (evil / "app.py").write_text(
        "from avlite.c60_apps.c61_app_strategy import AppStrategy\n"
        "class HijackHeadless(AppStrategy):\n"
        "    cli_name = 'headless'\n"
        "    help = 'hijacked headless'\n"
        "    def run(self, args, unknown):\n"
        "        return 1\n",
        encoding="utf-8",
    )
    registry = Path(os.environ["AVLITE_CONFIG_DIR"]) / "apps.yaml"
    registry.write_text(
        f"qt5: {plugin}\nsetting: {plugin}\nextra: {evil}\nbroken: not_a_real_module_xyz\n",
        encoding="utf-8",
    )
    try:
        with caplog.at_level(logging.WARNING, logger="avlite.c60_apps.c61_app_strategy"):
            bootstrap_apps()
        assert AppStrategy.registry["qt5"].__name__ == "DemoQtApp"
        assert AppStrategy.registry["setting"].__name__ == "SettingApp"
        assert AppStrategy.registry["headless"].__name__ == "HeadlessApp"
        assert "Skipping app 'setting'" in caplog.text
        assert "HijackHeadless" in caplog.text
        assert "Could not load app 'broken'" in caplog.text
    finally:
        AppStrategy.registry.pop("qt5", None)
        for mod in list(sys.modules):
            if mod.startswith("avlite.plugins.demo_qt5") or mod.startswith("avlite.plugins.evil"):
                del sys.modules[mod]
