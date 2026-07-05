"""Tests for user config directory resolution."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import yaml

from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_apps.c62_factory import StackSettingsSync, load_stack_settings
from avlite.plugins.p60_visualizer_tk.p65_ui_lib import DataPicker, UiAssets
from avlite.c60_apps.c64_settings_schema import SettingsValidationError
from avlite.c60_apps.c68_paths import (
    COMMUNITY_DEV_SUBDIR,
    PRIVATE_DEV_SUBDIR,
    ConfigPaths,
    PluginPaths,
)
from avlite.c60_apps.c68_paths import DataPaths
from avlite.c60_apps.c65_setting_utils import (
    dev_mode_export_warning,
    dev_mode_uninstall_warning,
    export_profile,
    import_profile,
    list_profiles,
    order_profiles_for_dropdown,
    profile_file_path,
)
from avlite.plugins.p60_setting_cli.p61_setting_cli import cmd_export_profile, cmd_import_profile

REPO_EXEC = Path(__file__).resolve().parents[2] / "avlite" / "configs" / "c40_execution.yaml"
REPO_DATA = Path(__file__).resolve().parents[2] / "avlite" / "data"
SAMPLE_MAP = "data/san_campus.xodr"


def test_get_config_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    assert ConfigPaths.user_dir() == tmp_path.resolve()
    assert ConfigPaths.user_configs_dir() == tmp_path.resolve()


def test_effective_config_path_read_falls_back_to_repo(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    repo_file = bundled / "c40_execution.yaml"
    repo_file.write_text("c40_bridge: BasicSim\n")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.can_edit_bundled", lambda: False
    )
    path = ConfigPaths.effective_path("configs/c40_execution.yaml", for_write=False)
    assert Path(path) == repo_file


def test_effective_path_user_target_dev_no_repo_fallback(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "default.yaml").write_text("c40_execution: {}\n")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "meta"))
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.can_edit_bundled", lambda: True
    )
    ConfigPaths.set_repo_target(False)
    path = ConfigPaths.effective_path("configs/default.yaml", for_write=False)
    assert Path(path) == config_dir / "default.yaml"
    assert not (config_dir / "default.yaml").is_file()


def test_copy_bundled_profiles_to_user_when_empty(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "default.yaml").write_text("c40_execution: {repo: true}\n")
    (bundled / "Carla_Town10.yaml").write_text("c40_execution: {carla: true}\n")
    (bundled / "plugin_p60_headless_mode.yaml").write_text("ignored: true\n")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    copied = ConfigPaths.copy_bundled_profiles_to_user()
    assert sorted(copied) == ["Carla_Town10.yaml", "default.yaml"]
    assert (config_dir / "default.yaml").read_text() == "c40_execution: {repo: true}\n"
    assert (config_dir / "Carla_Town10.yaml").read_text() == "c40_execution: {carla: true}\n"
    assert not (config_dir / "plugin_p60_headless_mode.yaml").exists()
    assert ConfigPaths.copy_bundled_profiles_to_user() == []


def test_copy_bundled_profiles_to_user_copies_missing_when_user_has_other_profiles(
    monkeypatch, tmp_path,
):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "default.yaml").write_text("c40_execution: {repo: true}\n")
    (bundled / "Carla_Town10.yaml").write_text("c40_execution: {carla: true}\n")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "robot.yaml").write_text("c40_execution: {robot: true}\n")
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    copied = ConfigPaths.copy_bundled_profiles_to_user()
    assert sorted(copied) == ["Carla_Town10.yaml", "default.yaml"]
    assert (config_dir / "robot.yaml").read_text() == "c40_execution: {robot: true}\n"
    assert (config_dir / "default.yaml").read_text() == "c40_execution: {repo: true}\n"
    assert ConfigPaths.copy_bundled_profiles_to_user() == []


def test_copy_bundled_profiles_to_user_skips_existing_only(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "default.yaml").write_text("c40_execution: {repo: true}\n")
    (bundled / "Carla_Town10.yaml").write_text("c40_execution: {carla: true}\n")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text("c40_execution: {user: true}\n")
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    copied = ConfigPaths.copy_bundled_profiles_to_user()
    assert copied == ["Carla_Town10.yaml"]
    assert (config_dir / "default.yaml").read_text() == "c40_execution: {user: true}\n"
    assert (config_dir / "Carla_Town10.yaml").read_text() == "c40_execution: {carla: true}\n"


def test_list_profiles_respects_repo_target(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "default.yaml").write_text("c40_execution: {}\n")
    (bundled / "Carla_Town10.yaml").write_text("c40_execution: {}\n")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "robot.yaml").write_text("c40_execution: {}\n")
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "meta"))
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.can_edit_bundled", lambda: True
    )
    ConfigPaths.set_repo_target(False)
    assert list_profiles() == ["default", "Carla_Town10", "robot"]
    try:
        ConfigPaths.set_repo_target(True)
        assert list_profiles() == ["default", "Carla_Town10"]
        empty_user = tmp_path / "empty_user"
        empty_user.mkdir()
        monkeypatch.setenv("AVLITE_CONFIG_DIR", str(empty_user))
        ConfigPaths.set_repo_target(True)
        assert list_profiles() == ["default", "Carla_Town10"]
        assert not (empty_user / "default.yaml").exists()
    finally:
        ConfigPaths.set_repo_target(False)


def test_list_profiles_seeds_missing_bundled_profiles_in_user_mode(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "default.yaml").write_text("c40_execution: {repo: true}\n")
    (bundled / "Carla_Town10.yaml").write_text("c40_execution: {carla: true}\n")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "meta"))
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.can_edit_bundled", lambda: True
    )
    ConfigPaths.set_repo_target(False)
    assert list_profiles() == ["default", "Carla_Town10"]
    assert (config_dir / "default.yaml").read_text() == "c40_execution: {repo: true}\n"
    assert (config_dir / "Carla_Town10.yaml").read_text() == "c40_execution: {carla: true}\n"


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
    # User-local overrides of repo profile files are removed on reset.
    user_default = ConfigPaths.user_dir() / "default.yaml"
    user_default.write_text("c40_execution: {}\n")
    deleted = ConfigPaths.clear_user_profiles()
    assert not user_default.is_file()
    assert len(deleted) >= 1


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


def test_install_dir_dev_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "plugins"))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)

    PluginPaths.set_dev_mode(False)
    assert PluginPaths.install_dir() == (tmp_path / "plugins").resolve()

    PluginPaths.set_dev_mode(True)
    assert PluginPaths.install_dir() == (tmp_path / "plugins").resolve()


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


def test_plugin_settings_location_display(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "default.yaml").write_text("plugins: {}\n")
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)
    path = PluginPaths.format_display(profile_file_path("default", for_write=False))
    assert f"{path} (plugins.sample_avlite_plugin)" == (
        "~/config/default.yaml (plugins.sample_avlite_plugin)"
    )


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
        "avlite.c60_apps.c68_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    ConfigPaths.set_repo_target(True)
    path = ConfigPaths.effective_path("configs/c40_execution.yaml", for_write=True)
    assert Path(path) == bundled / "c40_execution.yaml"


def test_is_community_plugin_settings_basename():
    assert not ConfigPaths.is_community_plugin_settings_basename("c40_execution.yaml")
    assert not ConfigPaths.is_community_plugin_settings_basename(
        "plugin_p60_headless_mode.yaml"
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
        "avlite.c60_apps.c68_paths.ConfigPaths.bundled_dir", lambda: bundled
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
    (bundled / "plugin_p60_headless_mode.yaml").write_text("default: {}\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(meta))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    ConfigPaths.set_repo_target(True)
    path = ConfigPaths.effective_path(
        "configs/plugin_p60_headless_mode.yaml", for_write=True
    )
    assert Path(path) == bundled / "plugin_p60_headless_mode.yaml"


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
        "avlite.c60_apps.c68_paths.ConfigPaths.bundled_dir", lambda: bundled
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

    (config_dir / "robot.yaml").write_text(
        yaml.dump(
            {
                "c40_execution": {"c40_bridge": "BasicSim"},
                "c10_perception": {"c15_detection_z_min": 0.5},
                "c69_apps": {"c62_community_plugins": {}},
            }
        )
    )

    out_path = tmp_path / "robot.yaml"
    count = export_profile("robot", out_path)
    assert count >= 2

    (config_dir / "robot.yaml").unlink()

    assert import_profile(out_path) == "robot"
    data = yaml.safe_load((config_dir / "robot.yaml").read_text())
    assert data["c40_execution"]["c40_bridge"] == "BasicSim"
    assert data["c10_perception"]["c15_detection_z_min"] == 0.5


def test_profile_export_finds_repo_profile(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "default.yaml").write_text("c40_execution: {c40_bridge: BasicSim}\n")
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "meta"))
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.can_edit_bundled", lambda: False
    )
    ConfigPaths.set_repo_target(False)
    out_path = tmp_path / "default.yaml"
    count = export_profile("default", out_path)
    assert count >= 1
    data = yaml.safe_load(out_path.read_text())
    assert isinstance(data, dict) and data


def test_profile_import_writes_to_user_config_dir(monkeypatch, tmp_path):
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "default.yaml").write_text("c40_execution: {c40_bridge: BasicSim}\n")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "meta"))
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.bundled_dir", lambda: bundled
    )
    monkeypatch.setattr(
        "avlite.c60_apps.c68_paths.ConfigPaths.can_edit_bundled", lambda: False
    )
    ConfigPaths.set_repo_target(False)

    out_path = tmp_path / "default.yaml"
    export_profile("default", out_path)
    assert not (config_dir / "default.yaml").exists()

    import_profile(out_path, overwrite=True)
    user_file = config_dir / "default.yaml"
    assert user_file.is_file()
    assert "c40_execution" in yaml.safe_load(user_file.read_text())


def test_profile_import_conflict_without_overwrite(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)

    (config_dir / "robot.yaml").write_text(
        yaml.dump({"c40_execution": {"c40_bridge": "Existing"}})
    )
    incoming = tmp_path / "robot.yaml"
    incoming.write_text(yaml.dump({"c40_execution": {"c40_bridge": "Imported"}}))

    with pytest.raises(ValueError, match="already exists"):
        import_profile(incoming, overwrite=False)


def test_profile_import_rejects_invalid_types(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)

    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.dump({"c40_execution": {"c40_control_dt": "bad"}}))

    with pytest.raises(SettingsValidationError):
        import_profile(bad, overwrite=True)
    assert not (config_dir / "bad.yaml").exists()


def test_profile_export_import_community_plugin(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)

    (config_dir / "robot.yaml").write_text(
        yaml.dump(
            {
                "c40_execution": {"c40_bridge": "BasicSim"},
                "c69_apps": {"c62_community_plugins": {"foo": "foo"}},
                "plugins": {"foo": {"setting_a": 1}},
            }
        )
    )

    out_path = tmp_path / "robot.yaml"
    export_profile("robot", out_path, include_plugins=True)

    (config_dir / "robot.yaml").unlink()

    import_profile(out_path)
    data = yaml.safe_load((config_dir / "robot.yaml").read_text())
    assert data["plugins"]["foo"]["setting_a"] == 1


def test_profile_export_excludes_sections_when_flagged(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)

    (config_dir / "robot.yaml").write_text(
        yaml.dump(
            {
                "c40_execution": {"c40_bridge": "BasicSim"},
                "c69_apps": {"c62_community_plugins": {}},
                "plugins": {"foo": {"setting_a": 1}},
            }
        )
    )

    out_path = tmp_path / "robot.yaml"
    export_profile("robot", out_path, include_app=False, include_plugins=False)
    data = yaml.safe_load(out_path.read_text())
    assert "c40_execution" in data
    assert "c69_apps" not in data
    assert "plugins" not in data


def test_profile_export_excludes_stack_when_flagged(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)

    (config_dir / "robot.yaml").write_text(
        yaml.dump(
            {
                "c40_execution": {"c40_bridge": "BasicSim"},
                "c69_apps": {"c62_community_plugins": {}},
                "plugins": {"foo": {"setting_a": 1}},
            }
        )
    )

    out_path = tmp_path / "robot_stackless.yaml"
    export_profile("robot", out_path, include_stack=False)
    data = yaml.safe_load(out_path.read_text())
    assert "c40_execution" not in data
    assert "c69_apps" in data
    assert data["plugins"]["foo"]["setting_a"] == 1


def test_profile_export_rejects_empty_selection(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)

    (config_dir / "robot.yaml").write_text(
        yaml.dump({"c40_execution": {"c40_bridge": "BasicSim"}})
    )

    with pytest.raises(ValueError, match="Nothing selected to export"):
        export_profile(
            "robot",
            tmp_path / "empty.yaml",
            include_stack=False,
            include_app=False,
            include_plugins=False,
        )


def test_cli_export_import_profile(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    ConfigPaths.set_repo_target(False)

    (config_dir / "robot.yaml").write_text(
        yaml.dump(
            {
                "c40_execution": {"c40_bridge": "BasicSim"},
                "c69_apps": {"c62_community_plugins": {}},
            }
        )
    )

    out_path = tmp_path / "robot.yaml"
    assert (
        cmd_export_profile(
            argparse.Namespace(
                profile="robot", output=str(out_path), no_stack=False, no_app=False, no_plugins=False
            )
        )
        == 0
    )

    (config_dir / "robot.yaml").unlink()
    assert cmd_import_profile(argparse.Namespace(path=str(out_path), force=True)) == 0
    assert (config_dir / "robot.yaml").is_file()


def test_resolve_ui_asset_path_independent_of_cwd(monkeypatch):
    monkeypatch.chdir("/tmp")
    path = UiAssets.resolve("logo.png")
    assert path.is_file()
    assert path.name == "logo.png"
