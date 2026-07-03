"""Tests for community plugin import path filtering."""

import sys

from avlite.c30_control.c32_control_strategy import ControlStrategy
from avlite.c50_apps.c53_plugins import (
    import_plugin_modules,
    plugin_module_prefix,
    sync_community_plugins,
)


def test_import_plugin_modules_skips_venv(tmp_path):
    plugin_dir = tmp_path / "sample_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "settings.py").write_text("X = 1\n")

    venv_setup = plugin_dir / ".venv" / "lib" / "site-packages" / "setup.py"
    venv_setup.parent.mkdir(parents=True)
    venv_setup.write_text("import sys; sys.exit('venv setup must not run')\n")

    prefix = plugin_module_prefix("sample_plugin")
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]

    import_plugin_modules(str(plugin_dir), pkg_name="sample_plugin")

    assert f"{prefix}.settings" in sys.modules
    assert not any(
        name.startswith(prefix + ".") and "venv" in name
        for name in sys.modules
    )


def test_sync_community_plugins_unloads_removed(tmp_path):
    plugin_dir = tmp_path / "sample_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "strategy.py").write_text(
        "from avlite.c30_control.c32_control_strategy import ControlStrategy\n"
        "class SamplePluginController(ControlStrategy):\n"
        "    def control(self, ego_state, local_plan):\n"
        "        return self.cmd\n"
    )
    (plugin_dir / "__init__.py").write_text("from .strategy import SamplePluginController\n")

    prefix = plugin_module_prefix("sample_plugin")
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]
    ControlStrategy.registry.pop("SamplePluginController", None)

    sync_community_plugins({"sample_plugin": str(plugin_dir)})
    assert "SamplePluginController" in ControlStrategy.registry

    sync_community_plugins({})
    assert "SamplePluginController" not in ControlStrategy.registry
    assert prefix not in sys.modules
