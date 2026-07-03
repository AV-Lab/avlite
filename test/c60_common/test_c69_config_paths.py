"""Tests for user config directory resolution."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pytest
import yaml

from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c50_apps.c52_factory import StackSettingsSync, load_stack_settings
from avlite.plugins.p50_visualizer_tk.settings import get_stack_settings_classes
from avlite.plugins.p50_visualizer_tk.p53_ui_lib import DataPicker, UiAssets
from avlite.c50_apps.c58_paths import (
    COMMUNITY_DEV_SUBDIR,
    PRIVATE_DEV_SUBDIR,
    ConfigPaths,
    PluginPaths,
)
from avlite.c50_apps.c58_paths import DataPaths
from avlite.c50_apps.c55_setting_utils import (
    dev_mode_export_warning,
    dev_mode_uninstall_warning,
    export_profile,
    import_profile,
    order_profiles_for_dropdown,
)
from avlite.plugins.p50_config_cli.p51_config_cli import cmd_export_profile, cmd_import_profile

REPO_EXEC = Path(__file__).resolve().parents[2] / "configs" / "c40_execution.yaml"
REPO_DATA = Path(__file__).resolve().parents[2] / "data"
SAMPLE_MAP = "data/san_campus.xodr"


def test_get_config_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    assert ConfigPaths.user_dir() == tmp_path.resolve()
    assert ConfigPaths.user_configs_dir() == tmp_path.resolve()


def test_effective_config_path_read_falls_back_to_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    path = ConfigPaths.effective_path("configs/c40_execution.yaml", for_write=False)
    assert Path(path) == REPO_EXEC


def test_effective_config_path_read_prefers_user(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "meta"))
    ConfigPaths.set_repo_target(False)
    user_file = ConfigPaths.user_dir() / "c40_execution.yaml"
    user_file.write_text("custom: {}\n")
    path = ConfigPaths.effective_path("configs/c40_execution.yaml", for_write=False)
    assert Path(path) == user_file


def test_effective_config_path_write_targets_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "meta"))
    ConfigPaths.set_repo_target(False)
    path = ConfigPaths.effective_path("configs/c40_execution.yaml", for_write=True)
    assert Path(path) == ConfigPaths.user_dir() / "c40_execution.yaml"
    assert ConfigPaths.user_dir().is_dir()


def test_clear_user_configs_removes_flat_files(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    user_file = ConfigPaths.user_dir() / "c40_execution.yaml"
    user_file.write_text("user: {}\n")
    viz_file = ConfigPaths.user_dir() / "plugin_p50_visualizer_tk.yaml"
    viz_file.write_text("viz: {}\n")
    deleted = ConfigPaths.clear_user_profiles()
    assert not user_file.is_file()
    assert not viz_file.is_file()
    assert len(deleted) >= 2


def test_resolve_plugin_path_name_only(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path))
    assert PluginPaths.resolve("my_plugin", "my_plugin") == tmp_path / "my_plugin"
    assert PluginPaths.resolve("my_plugin", "") == tmp_path / "my_plugin"


def test_installed_community_plugins_map(monkeypatch, tmp_path):
    install_dir = tmp_path / "plugins"
    (install_dir / "foo").mkdir(parents=True)
    (install_dir / ".hidden").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(install_dir))
    assert PluginPaths.installed_map() == {"foo": "~/plugins/foo"}


def test_installed_map_ignores_repo_dev_checkouts_when_off(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    plugin_dir = repo_root / "avlite-community-plugins" / "foo"
    plugin_dir.mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "meta"))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "empty_plugins"))
    (tmp_path / "empty_plugins").mkdir()
    monkeypatch.setattr(PluginPaths, "repo_root", staticmethod(lambda: repo_root))
    PluginPaths.set_dev_mode(False)
    assert PluginPaths.installed_map() == {}


def test_installed_map_includes_repo_dev_checkouts_when_dev_mode(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    plugin_dir = repo_root / "avlite-community-plugins" / "foo"
    plugin_dir.mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "meta"))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "empty_plugins"))
    (tmp_path / "empty_plugins").mkdir()
    monkeypatch.setattr(PluginPaths, "repo_root", staticmethod(lambda: repo_root))
    PluginPaths.set_dev_mode(True)
    assert PluginPaths.installed_map() == {"foo": "avlite-community-plugins/foo"}


def test_installed_map_install_dir_takes_priority(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    install_dir = tmp_path / "plugins"
    (install_dir / "foo").mkdir(parents=True)
    (repo_root / "avlite-community-plugins" / "foo").mkdir(parents=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "meta"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(install_dir))
    monkeypatch.setattr(PluginPaths, "repo_root", staticmethod(lambda: repo_root))
    PluginPaths.set_dev_mode(True)
    assert PluginPaths.installed_map() == {"foo": "~/plugins/foo"}


def test_resolve_plugin_path_legacy_absolute(tmp_path):
    legacy = tmp_path / "dev_plugin"
    legacy.mkdir()
    assert PluginPaths.resolve("dev_plugin", str(legacy)) == legacy


def test_resolve_plugin_path_repo_relative(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    plugin_dir = repo_root / "avlite-community-plugins" / "avlite-executer-ROS2"
    plugin_dir.mkdir(parents=True)
    monkeypatch.setattr(PluginPaths, "repo_root", staticmethod(lambda: repo_root))
    resolved = PluginPaths.resolve(
        "avlite-executer-ROS2", "avlite-community-plugins/avlite-executer-ROS2"
    )
    assert resolved.is_dir()
    assert resolved.name == "avlite-executer-ROS2"


def test_normalize_stored_repo_relative(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    plugin_dir = repo_root / "avlite-community-plugins" / "avlite-executer-ROS2"
    plugin_dir.mkdir(parents=True)
    monkeypatch.setattr(PluginPaths, "repo_root", staticmethod(lambda: repo_root))
    stored = PluginPaths.normalize_stored(
        "avlite-executer-ROS2", str(plugin_dir)
    )
    assert stored == "avlite-community-plugins/avlite-executer-ROS2"


def test_dev_mode_preference_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    assert not PluginPaths.is_dev_mode()
    PluginPaths.set_dev_mode(True)
    assert PluginPaths.is_dev_mode()
    PluginPaths.set_dev_mode(False)
    assert not PluginPaths.is_dev_mode()


def test_dev_mode_export_warning_none_when_off(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    PluginPaths.set_dev_mode(False)
    assert dev_mode_export_warning({"foo": "foo"}) is None


def test_dev_mode_export_warning_when_on(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    PluginPaths.set_dev_mode(True)
    warning = dev_mode_export_warning({})
    assert warning is not None
    assert COMMUNITY_DEV_SUBDIR in warning
    assert PRIVATE_DEV_SUBDIR in warning
    assert "target machine" in warning.lower()


def test_dev_mode_export_warning_lists_missing_plugins(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "plugins"))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(PluginPaths, "repo_root", staticmethod(lambda: repo_root))
    PluginPaths.set_dev_mode(True)
    warning = dev_mode_export_warning(
        {
            "installed": "installed",
            "missing": "avlite-community-plugins/missing",
        }
    )
    assert warning is not None
    assert "missing" in warning
    (tmp_path / "plugins" / "installed").mkdir(parents=True)
    warning2 = dev_mode_export_warning(
        {
            "installed": "installed",
            "missing": "avlite-community-plugins/missing",
        }
    )
    assert "missing" in warning2
    assert "installed" not in warning2.split("Plugins not found locally:")[-1]


def test_dev_mode_uninstall_warning_none_when_off(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    PluginPaths.set_dev_mode(False)
    install_dir = PluginPaths.install_dir()
    assert dev_mode_uninstall_warning(install_dir, "foo") is None


def test_dev_mode_uninstall_warning_none_for_other_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "plugins"))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    PluginPaths.set_dev_mode(True)
    assert dev_mode_uninstall_warning(tmp_path / "other", "foo") is None


def test_dev_mode_uninstall_warning_none_for_install_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "plugins"))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    PluginPaths.set_dev_mode(True)
    install_dir = PluginPaths.install_dir()
    assert dev_mode_uninstall_warning(install_dir, "foo") is None


def test_dev_mode_uninstall_warning_when_dev_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setattr(PluginPaths, "repo_root", staticmethod(lambda: repo_root))
    PluginPaths.set_dev_mode(True)
    dev_dir = PluginPaths.community_dev_dir()
    warning = dev_mode_uninstall_warning(dev_dir, "my_plugin")
    assert warning is not None
    assert "permanently delete" in warning.lower()
    assert "commit" in warning.lower()
    assert "my_plugin" in warning


def test_clone_dir_dev_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "plugins"))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)

    PluginPaths.set_dev_mode(False)
    assert PluginPaths.clone_dir(private=False) == (tmp_path / "plugins").resolve()
    assert PluginPaths.clone_dir(private=True) == (tmp_path / "plugins").resolve()

    PluginPaths.set_dev_mode(True)
    assert PluginPaths.clone_dir(private=False) == (tmp_path / "plugins").resolve()
    assert PluginPaths.clone_dir(private=True) == (tmp_path / "plugins").resolve()


def test_register_stored_path_uses_explicit_home_path(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "meta"))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "plugins"))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(PluginPaths, "repo_root", staticmethod(lambda: repo_root))

    PluginPaths.set_dev_mode(False)
    assert PluginPaths.register_stored_path("foo", private=False) == "~/plugins/foo"
    assert PluginPaths.register_stored_path("bar", private=True) == "~/plugins/bar"

    PluginPaths.set_dev_mode(True)
    assert PluginPaths.register_stored_path("foo", private=False) == "avlite-community-plugins/foo"
    assert PluginPaths.register_stored_path("bar", private=True) == "avlite-private-plugins/bar"


def test_load_path_community_dev_dir(monkeypatch, tmp_path):
    repo_root = tmp_path / "repo"
    plugin_dir = repo_root / "avlite-community-plugins" / "my_plugin"
    plugin_dir.mkdir(parents=True)
    monkeypatch.setattr(PluginPaths, "repo_root", staticmethod(lambda: repo_root))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "plugins"))

    loaded = PluginPaths.load_path("my_plugin", "avlite-community-plugins/my_plugin")
    assert loaded == plugin_dir.resolve()
    assert PluginPaths.load_path("missing", "avlite-community-plugins/missing") is None


def test_load_path_install_dir(monkeypatch, tmp_path):
    plugins_dir = tmp_path / "plugins"
    plugin_dir = plugins_dir / "my_plugin"
    plugin_dir.mkdir(parents=True)
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins_dir))

    loaded = PluginPaths.load_path("my_plugin", "my_plugin")
    assert loaded == plugin_dir.resolve()


def test_resolve_plugin_path_tilde_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "plugins"))
    plugin_dir = tmp_path / "plugins" / "foo"
    plugin_dir.mkdir(parents=True)
    stored = PluginPaths.format_display(plugin_dir)
    assert stored == "~/plugins/foo"
    assert PluginPaths.resolve("foo", stored) == plugin_dir.resolve()


def test_normalize_community_plugin_stored_under_plugins_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins_dir))
    install = plugins_dir / "my_plugin"
    install.mkdir()
    assert PluginPaths.normalize_stored("my_plugin", str(install)) == "~/plugins/my_plugin"


def test_normalize_community_plugin_stored_home_relative(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "plugins"))
    custom = tmp_path / "dev" / "my_plugin"
    custom.mkdir(parents=True)
    stored = PluginPaths.normalize_stored("my_plugin", str(custom))
    assert stored == "~/dev/my_plugin"
    assert PluginPaths.resolve("my_plugin", stored) == custom.resolve()


def test_format_user_path_and_settings_display(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "plugin_sample_avlite_plugin.yaml").write_text("default: {}\n")
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)
    display = PluginPaths.settings_display_path("sample_avlite_plugin")
    assert display == "~/config/plugin_sample_avlite_plugin.yaml"


def test_get_data_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_DATA_DIR", str(tmp_path))
    assert DataPaths.user_dir() == tmp_path.resolve()


def test_get_data_dir_default_under_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("AVLITE_DATA_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert DataPaths.user_dir() == ConfigPaths.user_dir() / "data"


def test_get_data_dir_follows_config_dir(monkeypatch, tmp_path):
    config_dir = tmp_path / "custom_config"
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("AVLITE_DATA_DIR", raising=False)
    assert DataPaths.user_dir() == (config_dir / "data").resolve()


def test_get_absolute_path_read_falls_back_to_legacy_share_data(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AVLITE_DATA_DIR", raising=False)
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    legacy_data = tmp_path / ".local" / "share" / "avlite" / "data"
    legacy_data.mkdir(parents=True)
    legacy_file = legacy_data / "san_campus.xodr"
    legacy_file.write_text("legacy copy")
    path = DataPaths.resolve(SAMPLE_MAP)
    assert Path(path) == legacy_file.resolve()


def test_get_absolute_path_read_falls_back_to_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_DATA_DIR", str(tmp_path / "user_data"))
    path = DataPaths.resolve(SAMPLE_MAP)
    assert Path(path) == REPO_DATA / "san_campus.xodr"


def test_get_absolute_path_read_prefers_user(monkeypatch, tmp_path):
    user_data = tmp_path / "user_data"
    monkeypatch.setenv("AVLITE_DATA_DIR", str(user_data))
    user_file = user_data / "san_campus.xodr"
    user_file.parent.mkdir(parents=True)
    user_file.write_text("user copy")
    path = DataPaths.resolve(SAMPLE_MAP)
    assert Path(path) == user_file


def test_get_absolute_path_write_targets_user_dir(monkeypatch, tmp_path):
    user_data = tmp_path / "user_data"
    monkeypatch.setenv("AVLITE_DATA_DIR", str(user_data))
    path = DataPaths.resolve("data/20260101_120000_global_plan.json", for_write=True)
    assert Path(path) == user_data / "20260101_120000_global_plan.json"
    assert user_data.is_dir()


def _home_test_user_data(name: str) -> Path:
    return Path.home() / ".avlite_test" / name


def test_list_map_candidates_show_user_and_repo(monkeypatch):
    user_data = _home_test_user_data("list_maps")
    user_data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AVLITE_DATA_DIR", str(user_data))
    try:
        user_file = user_data / "san_campus.xodr"
        user_file.write_text("user copy")
        candidates = DataPicker.list_map_candidates()
        user_picker = "~/" + user_file.resolve().relative_to(Path.home()).as_posix()
        assert "data/san_campus.xodr" in candidates
        assert user_picker in candidates
        assert candidates.index(user_picker) < candidates.index("data/san_campus.xodr")
    finally:
        import shutil
        shutil.rmtree(user_data.parent, ignore_errors=True)


def test_resolve_picker_data_path_user_explicit(monkeypatch):
    user_data = _home_test_user_data("resolve_user")
    user_data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AVLITE_DATA_DIR", str(user_data))
    try:
        user_file = user_data / "test_map.xodr"
        user_file.write_text("x")
        stored = "~/" + user_file.resolve().relative_to(Path.home()).as_posix()
        assert Path(DataPaths.resolve_stored(stored)) == user_file.resolve()
    finally:
        import shutil
        shutil.rmtree(user_data.parent, ignore_errors=True)


def test_resolve_picker_data_path_data_prefix_prefers_user(monkeypatch):
    user_data = _home_test_user_data("resolve_legacy")
    user_data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AVLITE_DATA_DIR", str(user_data))
    try:
        user_file = user_data / "san_campus.xodr"
        user_file.write_text("user copy")
        assert Path(DataPaths.resolve_stored("data/san_campus.xodr")) == user_file.resolve()
    finally:
        import shutil
        shutil.rmtree(user_data.parent, ignore_errors=True)


def test_data_picker_path_for_setting_shows_user_when_override_exists(monkeypatch):
    user_data = _home_test_user_data("display_user")
    user_data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("AVLITE_DATA_DIR", str(user_data))
    try:
        user_file = user_data / "san_campus.xodr"
        user_file.write_text("user copy")
        display = DataPicker.display_path("data/san_campus.xodr")
        assert display.startswith("~/")
        assert display.endswith("san_campus.xodr")
    finally:
        import shutil
        shutil.rmtree(user_data.parent, ignore_errors=True)


def test_apply_map_selection_xodr_clears_lidar_boundary():
    ExecutionSettings.c46_lidar_boundary_file = "data/yasmarina.track.json"
    StackSettingsSync.apply_map_selection("data/san_campus.xodr")
    assert ExecutionSettings.c46_lidar_boundary_file == ""
    assert ExecutionSettings.c40_hd_map == "data/san_campus.xodr"


def test_apply_map_selection_race_json_sets_lidar_boundary():
    path = "data/race_boundary_yas_marina.map.json"
    StackSettingsSync.apply_map_selection(path)
    assert ExecutionSettings.c43_race_boundary_map == path
    assert ExecutionSettings.c46_lidar_boundary_file == path


def test_order_profiles_for_dropdown_puts_default_first():
    assert order_profiles_for_dropdown(["Carla_Town10", "default", "ros"]) == [
        "default",
        "Carla_Town10",
        "ros",
    ]


def test_order_profiles_for_dropdown_without_default():
    assert order_profiles_for_dropdown(["ros", "SAN"]) == ["ros", "SAN"]


def test_order_profiles_for_dropdown_empty():
    assert order_profiles_for_dropdown([]) == []


def test_get_startup_profile_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    assert ConfigPaths.startup_profile() is None


def test_set_startup_profile_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    ConfigPaths.set_startup_profile("ros")
    assert ConfigPaths.startup_profile() == "ros"
    assert (ConfigPaths.user_dir() / "startup_profile").read_text() == "ros\n"


def test_clear_user_configs_removes_startup_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    ConfigPaths.set_startup_profile("ros")
    ConfigPaths.clear_user_profiles()
    assert ConfigPaths.startup_profile() is None


def test_repo_config_target_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    assert not ConfigPaths.is_repo_target()
    ConfigPaths.set_repo_target(True)
    assert ConfigPaths.is_repo_target()
    ConfigPaths.set_repo_target(False)
    assert not ConfigPaths.is_repo_target()


def test_effective_config_path_repo_target_write(monkeypatch, tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "c40_execution.yaml").write_text("x: 1\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(meta))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(
        "avlite.c50_apps.c58_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    ConfigPaths.set_repo_target(True)
    path = ConfigPaths.effective_path("configs/c40_execution.yaml", for_write=True)
    assert Path(path) == bundled / "c40_execution.yaml"


def test_is_community_plugin_settings_basename():
    assert not ConfigPaths.is_community_plugin_settings_basename("c40_execution.yaml")
    assert not ConfigPaths.is_community_plugin_settings_basename(
        "plugin_p50_headless_mode.yaml"
    )
    assert ConfigPaths.is_community_plugin_settings_basename(
        "plugin_avlite-bridge-carla.yaml"
    )


def test_effective_path_repo_target_skips_community_plugin_write(monkeypatch, tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "plugin_avlite-bridge-carla.yaml").write_text("default: {}\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(meta))
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(
        "avlite.c50_apps.c58_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    ConfigPaths.set_repo_target(True)
    path = ConfigPaths.effective_path(
        "configs/plugin_avlite-bridge-carla.yaml", for_write=True
    )
    assert Path(path) == config_dir / "plugin_avlite-bridge-carla.yaml"


def test_effective_path_repo_target_still_writes_builtin_plugin(monkeypatch, tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "plugin_p50_headless_mode.yaml").write_text("default: {}\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(meta))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(
        "avlite.c50_apps.c58_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    ConfigPaths.set_repo_target(True)
    path = ConfigPaths.effective_path(
        "configs/plugin_p50_headless_mode.yaml", for_write=True
    )
    assert Path(path) == bundled / "plugin_p50_headless_mode.yaml"


def test_effective_path_community_plugin_read_prefers_user(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    bundled_file = bundled / "plugin_avlite-bridge-carla.yaml"
    bundled_file.write_text("default: {repo: true}\n")
    user_file = config_dir / "plugin_avlite-bridge-carla.yaml"
    user_file.write_text("default: {user: true}\n")
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(
        "avlite.c50_apps.c58_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    ConfigPaths.set_repo_target(True)
    path = ConfigPaths.effective_path(
        "configs/plugin_avlite-bridge-carla.yaml", for_write=False
    )
    assert Path(path) == user_file


def test_profile_export_import_round_trip(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)

    (config_dir / "c40_execution.yaml").write_text(
        yaml.dump({"robot": {"c40_bridge": "BasicSim"}})
    )
    (config_dir / "c59_apps.yaml").write_text(
        yaml.dump({"robot": {"c50_community_plugins": {}}})
    )
    (config_dir / "c10_perception.yaml").write_text(
        yaml.dump({"robot": {"c15_detection_z_min": 0.5}})
    )

    zip_path = tmp_path / "robot.zip"
    count = export_profile("robot", zip_path, settings_classes=get_stack_settings_classes())
    assert count >= 2

    (config_dir / "c40_execution.yaml").unlink()
    (config_dir / "c59_apps.yaml").unlink()
    (config_dir / "c10_perception.yaml").unlink()

    assert import_profile(zip_path, settings_classes=get_stack_settings_classes()) == "robot"
    exec_data = yaml.safe_load((config_dir / "c40_execution.yaml").read_text())
    perc_data = yaml.safe_load((config_dir / "c10_perception.yaml").read_text())
    assert exec_data["robot"]["c40_bridge"] == "BasicSim"
    assert perc_data["robot"]["c15_detection_z_min"] == 0.5


def test_profile_export_finds_repo_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path / "empty"))
    ConfigPaths.set_repo_target(False)
    zip_path = tmp_path / "default.zip"
    count = export_profile("default", zip_path, settings_classes=get_stack_settings_classes())
    assert count >= 1
    with zipfile.ZipFile(zip_path) as zf:
        assert any(name.endswith(".yaml") for name in zf.namelist())


def test_profile_import_writes_to_user_config_dir(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)

    zip_path = tmp_path / "default.zip"
    export_profile("default", zip_path, settings_classes=get_stack_settings_classes())
    assert not (config_dir / "c40_execution.yaml").exists()

    import_profile(zip_path, settings_classes=get_stack_settings_classes(), overwrite=True)
    user_exec = config_dir / "c40_execution.yaml"
    assert user_exec.is_file()
    assert "default" in yaml.safe_load(user_exec.read_text())


def test_profile_import_conflict_without_overwrite(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)

    (config_dir / "c40_execution.yaml").write_text(
        yaml.dump({"robot": {"c40_bridge": "Existing"}})
    )
    zip_path = tmp_path / "robot.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "c40_execution.yaml",
            yaml.dump({"robot": {"c40_bridge": "Imported"}}),
        )

    with pytest.raises(ValueError, match="already exists"):
        import_profile(zip_path, settings_classes=get_stack_settings_classes(), overwrite=False)


def test_profile_import_rejects_invalid_types(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)

    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "c40_execution.yaml",
            yaml.dump({"robot": {"c40_control_dt": "bad"}}),
        )

    with pytest.raises(ValueError):
        import_profile(zip_path, settings_classes=get_stack_settings_classes(), overwrite=True)
    assert not (config_dir / "c40_execution.yaml").exists()


def test_profile_export_import_community_plugin(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    plugins_dir = tmp_path / "plugins"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins_dir))
    ConfigPaths.set_repo_target(False)

    plugin_path = plugins_dir / "foo"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    plugin_path.mkdir()
    (plugin_path / "settings.py").write_text(
        "class PluginSettings:\n    setting_a: int = 0\n"
    )
    (config_dir / "plugin_foo.yaml").write_text(
        yaml.dump({"robot": {"setting_a": 1}})
    )
    (config_dir / "c59_apps.yaml").write_text(
        yaml.dump(
            {
                "robot": {
                    "c50_community_plugins": {"foo": "foo"},
                }
            }
        )
    )
    (config_dir / "c40_execution.yaml").write_text(
        yaml.dump({"robot": {"c40_bridge": "BasicSim"}})
    )

    zip_path = tmp_path / "robot.zip"
    export_profile("robot", zip_path, settings_classes=get_stack_settings_classes(), community_plugins={"foo": "foo"})

    (config_dir / "plugin_foo.yaml").unlink()
    (config_dir / "c40_execution.yaml").unlink()
    (config_dir / "c59_apps.yaml").unlink()

    import_profile(zip_path, settings_classes=get_stack_settings_classes())
    cp_data = yaml.safe_load((config_dir / "plugin_foo.yaml").read_text())
    assert cp_data["robot"]["setting_a"] == 1


def test_profile_import_legacy_community_zip_entry(monkeypatch, tmp_path):
    import zipfile

    config_dir = tmp_path / "config"
    plugins_dir = tmp_path / "plugins"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins_dir))
    ConfigPaths.set_repo_target(False)

    plugin_path = plugins_dir / "foo"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    plugin_path.mkdir()
    (plugin_path / "settings.py").write_text(
        "class PluginSettings:\n    setting_a: int = 0\n"
    )

    zip_path = tmp_path / "robot.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "c59_apps.yaml",
            yaml.dump(
                {
                    "robot": {
                        "c50_community_plugins": {"foo": "foo"},
                    }
                }
            ),
        )
        zf.writestr(
            "community/foo.yaml",
            yaml.dump({"robot": {"setting_a": 2}}),
        )

    import_profile(zip_path, settings_classes=get_stack_settings_classes())
    cp_data = yaml.safe_load((config_dir / "plugin_foo.yaml").read_text())
    assert cp_data["robot"]["setting_a"] == 2


def test_cli_export_import_profile(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)

    (config_dir / "c40_execution.yaml").write_text(
        yaml.dump({"robot": {"c40_bridge": "BasicSim"}})
    )
    (config_dir / "c59_apps.yaml").write_text(
        yaml.dump({"robot": {"c50_community_plugins": {}}})
    )

    zip_path = tmp_path / "robot.zip"
    assert cmd_export_profile(argparse.Namespace(profile="robot", output=str(zip_path))) == 0

    (config_dir / "c40_execution.yaml").unlink()
    assert cmd_import_profile(argparse.Namespace(zip_path=str(zip_path), force=True)) == 0
    assert (config_dir / "c40_execution.yaml").is_file()


def test_resolve_ui_asset_path_independent_of_cwd(monkeypatch):
    monkeypatch.chdir("/tmp")
    path = UiAssets.resolve("logo.png")
    assert path.is_file()
    assert path.name == "logo.png"
