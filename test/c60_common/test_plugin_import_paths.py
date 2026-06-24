"""Tests for community plugin import path filtering."""

import sys

from avlite.c60_common.c60_plugins import import_plugin_modules, plugin_module_prefix


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
