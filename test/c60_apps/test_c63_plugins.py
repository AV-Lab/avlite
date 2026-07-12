"""Tests for community plugin settings paths and import hook."""

from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path

import pytest
import yaml

from avlite.c60_apps.c63_plugins import (
    find_community_plugin_dir,
    import_plugin_modules,
    load_community_plugin_setting,
    plugin_module_prefix,
    register_community_plugin_import_hook,
)
from avlite.c60_apps.c68_paths import ConfigPaths, PluginPaths
from avlite.c60_apps.c65_setting_utils import load_setting, setting_section

_PLUGIN_NAME = "avlite-executer-ROS2"
_SETTINGS_BODY = (
    "class PluginSettings:\n"
    "    filepath = ''\n"
    "    replan_dt = 0.1\n"
    "    control_dt = 0.02\n"
)


def _clear_plugin_modules(name: str) -> None:
    prefix = plugin_module_prefix(name)
    for mod_name in list(sys.modules):
        if mod_name == prefix or mod_name.startswith(prefix + "."):
            del sys.modules[mod_name]


@pytest.fixture
def dashed_plugin(tmp_path):
    """Minimal community plugin with dashes in the directory name."""
    plugin_dir = tmp_path / "plugins" / _PLUGIN_NAME
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text('"""test plugin"""\n', encoding="utf-8")
    (plugin_dir / "settings.py").write_text(_SETTINGS_BODY, encoding="utf-8")
    _clear_plugin_modules(_PLUGIN_NAME)
    yield plugin_dir
    _clear_plugin_modules(_PLUGIN_NAME)


def test_import_plugin_modules_maps_dashed_plugin_section(dashed_plugin, monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    import_plugin_modules(str(dashed_plugin), pkg_name=_PLUGIN_NAME)

    from avlite.plugins.avlite_executer_ROS2.settings import PluginSettings

    assert setting_section(PluginSettings) == ("plugins", "avlite_executer_ROS2")


def test_load_setting_uses_plugin_section_after_import(dashed_plugin, monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "default.yaml").write_text(
        yaml.dump(
            {"plugins": {"avlite_executer_ROS2": {"replan_dt": 0.42, "control_dt": 0.07}}}
        ),
        encoding="utf-8",
    )

    import_plugin_modules(str(dashed_plugin), pkg_name=_PLUGIN_NAME)
    from avlite.plugins.avlite_executer_ROS2.settings import PluginSettings

    assert load_setting(PluginSettings, profile="default") is True
    assert PluginSettings.replan_dt == pytest.approx(0.42)
    assert PluginSettings.control_dt == pytest.approx(0.07)


def test_load_community_plugin_setting_reuses_imported_singleton(dashed_plugin, monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    import_plugin_modules(str(dashed_plugin), pkg_name=_PLUGIN_NAME)
    from avlite.plugins.avlite_executer_ROS2.settings import PluginSettings as module_ps

    cls = load_community_plugin_setting(_PLUGIN_NAME, str(dashed_plugin), profile="default")
    assert cls is module_ps


def test_load_setting_missing_file_is_non_fatal(monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))

    class _Settings:
        filepath = "configs/plugin_missing_xyz_test.yaml"

    with caplog.at_level(logging.DEBUG):
        assert load_setting(_Settings, profile="default") is False

    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_community_plugin_finder_imports_from_install_dir(monkeypatch, tmp_path):
    plugins_dir = tmp_path / "plugins"
    plugin_dir = plugins_dir / _PLUGIN_NAME
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "__init__.py").write_text('"""test plugin"""\n', encoding="utf-8")
    (plugin_dir / "settings.py").write_text(
        "class PluginSettings:\n    filepath = ''\n    value = 1\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins_dir))
    _clear_plugin_modules(_PLUGIN_NAME)
    register_community_plugin_import_hook()

    mod = importlib.import_module(f"{plugin_module_prefix(_PLUGIN_NAME)}.settings")
    assert mod.PluginSettings.value == 1
    _clear_plugin_modules(_PLUGIN_NAME)


def test_find_community_plugin_dir_from_install_dir(monkeypatch, tmp_path):
    install = tmp_path / "plugins" / _PLUGIN_NAME
    install.mkdir(parents=True)
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "plugins"))

    found = find_community_plugin_dir("avlite_executer_ROS2")
    assert found == install.resolve()


def test_find_community_plugin_dir_from_community_dev(monkeypatch, tmp_path):
    dev = tmp_path / "avlite-community-plugins" / _PLUGIN_NAME
    dev.mkdir(parents=True)
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "empty_plugins"))
    (tmp_path / "empty_plugins").mkdir()

    repo_root = tmp_path
    monkeypatch.setattr(PluginPaths, "repo_root", staticmethod(lambda: repo_root))

    found = find_community_plugin_dir("avlite_executer_ROS2")
    assert found == dev.resolve()


def test_load_stack_settings_imports_community_plugins(dashed_plugin, monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    (tmp_path / "default.yaml").write_text(
        yaml.dump(
            {
                "c69_apps": {
                    "c62_community_plugins": {
                        _PLUGIN_NAME: str(dashed_plugin),
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    from avlite.c60_apps.c62_factory import load_stack_settings

    _clear_plugin_modules(_PLUGIN_NAME)
    load_stack_settings(profile="default", load_plugins=True)

    prefix = plugin_module_prefix(_PLUGIN_NAME)
    assert f"{prefix}.settings" in sys.modules


def test_reload_lib_reloads_core_before_plugins(monkeypatch):
    """Core ABCs reload before plugin packages (public-API cache cleared between)."""
    from avlite.c60_apps.c63_plugins import reload_lib

    reloaded: list[str] = []
    real_reload = importlib.reload

    def _track(module):
        name = getattr(module, "__name__", "")
        reloaded.append(name)
        # Skip reloading c63 itself so this test's monkeypatch / caller stay intact.
        if name == "avlite.c60_apps.c63_plugins":
            return module
        return real_reload(module)

    monkeypatch.setattr(importlib, "reload", _track)

    fake_plugin = types.ModuleType("avlite.plugins._test_reload_order")
    # Give importlib.reload a trivial spec so tracking can record the name.
    fake_plugin.__spec__ = importlib.machinery.ModuleSpec(
        "avlite.plugins._test_reload_order", loader=None
    )
    sys.modules["avlite.plugins._test_reload_order"] = fake_plugin
    try:
        reload_lib(reload_plugins=True, exclude_settings=True)
    finally:
        sys.modules.pop("avlite.plugins._test_reload_order", None)

    assert reloaded and reloaded[0] == "avlite"
    core_idxs = [i for i, n in enumerate(reloaded) if n.startswith("avlite.c")]
    plugin_idxs = [i for i, n in enumerate(reloaded) if n.startswith("avlite.plugins.")]
    assert core_idxs, "expected core modules to reload"
    assert plugin_idxs, "expected plugin modules to reload"
    assert max(core_idxs) < min(plugin_idxs)


def test_avlite_public_api_survives_package_reload():
    """New _LAZY exports become importable after importlib.reload(avlite)."""
    import avlite as av

    assert "SingleTrajectory" in av._LAZY
    av._LAZY.pop("SingleTrajectory", None)
    av.__dict__.pop("SingleTrajectory", None)
    with pytest.raises(AttributeError):
        getattr(av, "SingleTrajectory")

    importlib.reload(av)
    from avlite import SingleTrajectory

    assert SingleTrajectory is not None
    assert "SingleTrajectory" in av._LAZY


def test_reload_lib_public_api_prediction_strategy_matches_submodule():
    """After reload_lib, ``from avlite import PredictionStrategy`` is the live ABC."""
    from avlite.c60_apps.c63_plugins import reload_lib

    # Prime the lazy cache with whatever is current.
    import avlite as av
    from avlite import PredictionStrategy as _primed  # noqa: F401

    assert "PredictionStrategy" in av.__dict__

    reload_lib(reload_plugins=False, exclude_settings=True)

    import avlite.c10_perception.c12_perception_strategy as c12
    from avlite import PredictionStrategy as api_ps

    assert api_ps is c12.PredictionStrategy
    assert "ConstantVelocityPrediction" in api_ps.registry


def test_reload_lib_plugin_via_public_api_registers_on_live_abc(tmp_path):
    """Plugin subclassing PredictionStrategy via ``avlite`` lands in the live registry."""
    from avlite.c60_apps.c63_plugins import reload_lib

    plugin_dir = tmp_path / "api_pred_plugin"
    plugin_dir.mkdir()
    (plugin_dir / "__init__.py").write_text("", encoding="utf-8")
    (plugin_dir / "predictor.py").write_text(
        "from avlite import PredictionStrategy, StackCapability\n"
        "\n"
        "class PluginApiReloadPredictor(PredictionStrategy):\n"
        "    world_requirements = frozenset()\n"
        "    stack_requirements = frozenset()\n"
        "    stack_capabilities = frozenset({StackCapability.PREDICTION})\n"
        "\n"
        "    def predict(self, perception_model=None, sensors=None):\n"
        "        return perception_model\n",
        encoding="utf-8",
    )

    pkg = "avlite.plugins.api_pred_plugin"
    # Clear any prior load.
    for name in list(sys.modules):
        if name == pkg or name.startswith(pkg + "."):
            del sys.modules[name]

    import_plugin_modules(str(plugin_dir), pkg_name="api_pred_plugin")
    from avlite import PredictionStrategy as before_ps

    assert "PluginApiReloadPredictor" in before_ps.registry

    # Cache a stale public-API binding, then reload — plugin must re-bind to new ABC.
    import avlite as av

    _ = av.PredictionStrategy  # ensure cached
    reload_lib(reload_plugins=True, exclude_settings=True)

    import avlite.c10_perception.c12_perception_strategy as c12
    from avlite import PredictionStrategy as after_ps

    assert after_ps is c12.PredictionStrategy
    assert "PluginApiReloadPredictor" in after_ps.registry
    assert after_ps.registry["PluginApiReloadPredictor"].__module__.startswith(pkg)
