"""Tests for user config directory resolution."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import pytest
import yaml

from avlite.c40_execution.c49_settings import ExecutionSettings
from avlite.c60_common.c67_paths import (
    apply_map_selection,
    clear_user_configs,
    community_plugin_settings_display_path,
    data_picker_path_for_setting,
    effective_config_path,
    format_user_path,
    get_absolute_path,
    get_config_dir,
    get_data_dir,
    get_startup_profile,
    get_user_configs_dir,
    installed_community_plugins_map,
    is_repo_config_target,
    list_map_file_candidates,
    normalize_community_plugin_stored,
    resolve_picker_data_path,
    resolve_plugin_path,
    set_repo_config_target,
    set_startup_profile,
)
from avlite.c60_common.c69_setting_utils import export_profile, import_profile
from avlite.plugins.p50_headless_mode.p52_config_cli import cmd_export_profile, cmd_import_profile

REPO_EXEC = Path(__file__).resolve().parents[2] / "configs" / "c40_execution.yaml"
REPO_DATA = Path(__file__).resolve().parents[2] / "data"
SAMPLE_MAP = "data/san_campus.xodr"


def test_get_config_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    assert get_config_dir() == tmp_path.resolve()
    assert get_user_configs_dir() == tmp_path.resolve()


def test_effective_config_path_read_falls_back_to_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    path = effective_config_path("configs/c40_execution.yaml", for_write=False)
    assert Path(path) == REPO_EXEC


def test_effective_config_path_read_prefers_user(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "meta"))
    set_repo_config_target(False)
    user_file = get_config_dir() / "c40_execution.yaml"
    user_file.write_text("custom: {}\n")
    path = effective_config_path("configs/c40_execution.yaml", for_write=False)
    assert Path(path) == user_file


def test_effective_config_path_write_targets_config_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "meta"))
    set_repo_config_target(False)
    path = effective_config_path("configs/c40_execution.yaml", for_write=True)
    assert Path(path) == get_config_dir() / "c40_execution.yaml"
    assert get_config_dir().is_dir()


def test_clear_user_configs_removes_flat_files(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    user_file = get_config_dir() / "c40_execution.yaml"
    user_file.write_text("user: {}\n")
    viz_file = get_config_dir() / "c50_visualization.yaml"
    viz_file.write_text("viz: {}\n")
    deleted = clear_user_configs()
    assert not user_file.is_file()
    assert not viz_file.is_file()
    assert len(deleted) >= 2


def test_resolve_plugin_path_name_only(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path))
    assert resolve_plugin_path("my_plugin", "my_plugin") == tmp_path / "my_plugin"
    assert resolve_plugin_path("my_plugin", "") == tmp_path / "my_plugin"


def test_installed_community_plugins_map(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path))
    (tmp_path / "foo").mkdir()
    (tmp_path / ".hidden").mkdir()
    assert installed_community_plugins_map() == {"foo": "foo"}


def test_resolve_plugin_path_legacy_absolute(tmp_path):
    legacy = tmp_path / "dev_plugin"
    legacy.mkdir()
    assert resolve_plugin_path("dev_plugin", str(legacy)) == legacy


def test_resolve_plugin_path_tilde_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "plugins"))
    plugin_dir = tmp_path / "plugins" / "foo"
    plugin_dir.mkdir(parents=True)
    stored = format_user_path(plugin_dir)
    assert stored == "~/plugins/foo"
    assert resolve_plugin_path("foo", stored) == plugin_dir.resolve()


def test_normalize_community_plugin_stored_under_plugins_dir(monkeypatch, tmp_path):
    plugins_dir = tmp_path / "plugins"
    plugins_dir.mkdir()
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins_dir))
    install = plugins_dir / "my_plugin"
    install.mkdir()
    assert normalize_community_plugin_stored("my_plugin", str(install)) == "my_plugin"


def test_normalize_community_plugin_stored_home_relative(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(tmp_path / "plugins"))
    custom = tmp_path / "dev" / "my_plugin"
    custom.mkdir(parents=True)
    stored = normalize_community_plugin_stored("my_plugin", str(custom))
    assert stored == "~/dev/my_plugin"
    assert resolve_plugin_path("my_plugin", stored) == custom.resolve()


def test_format_user_path_and_settings_display(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "plugin_sample_avlite_plugin.yaml").write_text("default: {}\n")
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    set_repo_config_target(False)
    display = community_plugin_settings_display_path("sample_avlite_plugin")
    assert display == "~/config/plugin_sample_avlite_plugin.yaml"


def test_get_data_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_DATA_DIR", str(tmp_path))
    assert get_data_dir() == tmp_path.resolve()


def test_get_absolute_path_read_falls_back_to_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_DATA_DIR", str(tmp_path / "user_data"))
    path = get_absolute_path(SAMPLE_MAP)
    assert Path(path) == REPO_DATA / "san_campus.xodr"


def test_get_absolute_path_read_prefers_user(monkeypatch, tmp_path):
    user_data = tmp_path / "user_data"
    monkeypatch.setenv("AVLITE_DATA_DIR", str(user_data))
    user_file = user_data / "san_campus.xodr"
    user_file.parent.mkdir(parents=True)
    user_file.write_text("user copy")
    path = get_absolute_path(SAMPLE_MAP)
    assert Path(path) == user_file


def test_get_absolute_path_write_targets_user_dir(monkeypatch, tmp_path):
    user_data = tmp_path / "user_data"
    monkeypatch.setenv("AVLITE_DATA_DIR", str(user_data))
    path = get_absolute_path("data/20260101_120000_global_plan.json", for_write=True)
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
        candidates = list_map_file_candidates()
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
        assert Path(resolve_picker_data_path(stored)) == user_file.resolve()
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
        assert Path(resolve_picker_data_path("data/san_campus.xodr")) == user_file.resolve()
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
        display = data_picker_path_for_setting("data/san_campus.xodr")
        assert display.startswith("~/")
        assert display.endswith("san_campus.xodr")
    finally:
        import shutil
        shutil.rmtree(user_data.parent, ignore_errors=True)


def test_apply_map_selection_xodr_clears_lidar_boundary():
    ExecutionSettings.c46_lidar_boundary_file = "data/yasmarina.track.json"
    apply_map_selection("data/san_campus.xodr")
    assert ExecutionSettings.c46_lidar_boundary_file == ""
    assert ExecutionSettings.c40_hd_map == "data/san_campus.xodr"


def test_apply_map_selection_race_json_sets_lidar_boundary():
    path = "data/race_boundary_yas_marina.map.json"
    apply_map_selection(path)
    assert ExecutionSettings.c43_race_boundary_map == path
    assert ExecutionSettings.c46_lidar_boundary_file == path


def test_get_startup_profile_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    assert get_startup_profile() is None


def test_set_startup_profile_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    set_startup_profile("ros")
    assert get_startup_profile() == "ros"
    assert (get_config_dir() / "startup_profile").read_text() == "ros\n"


def test_clear_user_configs_removes_startup_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path))
    set_startup_profile("ros")
    clear_user_configs()
    assert get_startup_profile() is None


def test_repo_config_target_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    assert not is_repo_config_target()
    set_repo_config_target(True)
    assert is_repo_config_target()
    set_repo_config_target(False)
    assert not is_repo_config_target()


def test_effective_config_path_repo_target_write(monkeypatch, tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    bundled = tmp_path / "bundled"
    bundled.mkdir()
    (bundled / "c40_execution.yaml").write_text("x: 1\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(meta))
    monkeypatch.delenv("AVLITE_CONFIG_DIR", raising=False)
    monkeypatch.setattr(
        "avlite.c60_common.c67_paths.bundled_config_dir", lambda: bundled
    )
    set_repo_config_target(True)
    path = effective_config_path("configs/c40_execution.yaml", for_write=True)
    assert Path(path) == bundled / "c40_execution.yaml"


def test_profile_export_import_round_trip(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    set_repo_config_target(False)

    (config_dir / "c40_execution.yaml").write_text(
        yaml.dump({"robot": {"c40_bridge": "BasicSim", "c40_community_plugins": {}}})
    )
    (config_dir / "c10_perception.yaml").write_text(
        yaml.dump({"robot": {"c15_detection_z_min": 0.5}})
    )

    zip_path = tmp_path / "robot.zip"
    count = export_profile("robot", zip_path)
    assert count >= 2

    (config_dir / "c40_execution.yaml").unlink()
    (config_dir / "c10_perception.yaml").unlink()

    assert import_profile(zip_path) == "robot"
    exec_data = yaml.safe_load((config_dir / "c40_execution.yaml").read_text())
    perc_data = yaml.safe_load((config_dir / "c10_perception.yaml").read_text())
    assert exec_data["robot"]["c40_bridge"] == "BasicSim"
    assert perc_data["robot"]["c15_detection_z_min"] == 0.5


def test_profile_export_finds_repo_profile(monkeypatch, tmp_path):
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(tmp_path / "empty"))
    set_repo_config_target(False)
    zip_path = tmp_path / "default.zip"
    count = export_profile("default", zip_path)
    assert count >= 1
    with zipfile.ZipFile(zip_path) as zf:
        assert any(name.endswith(".yaml") for name in zf.namelist())


def test_profile_import_writes_to_user_config_dir(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    set_repo_config_target(False)

    zip_path = tmp_path / "default.zip"
    export_profile("default", zip_path)
    assert not (config_dir / "c40_execution.yaml").exists()

    import_profile(zip_path, overwrite=True)
    user_exec = config_dir / "c40_execution.yaml"
    assert user_exec.is_file()
    assert "default" in yaml.safe_load(user_exec.read_text())


def test_profile_import_conflict_without_overwrite(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    set_repo_config_target(False)

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
        import_profile(zip_path, overwrite=False)


def test_profile_import_rejects_invalid_types(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    set_repo_config_target(False)

    zip_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "c40_execution.yaml",
            yaml.dump({"robot": {"c40_control_dt": "bad", "c40_community_plugins": {}}}),
        )

    with pytest.raises(ValueError):
        import_profile(zip_path, overwrite=True)
    assert not (config_dir / "c40_execution.yaml").exists()


def test_profile_export_import_community_plugin(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    plugins_dir = tmp_path / "plugins"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins_dir))
    set_repo_config_target(False)

    plugin_path = plugins_dir / "foo"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    plugin_path.mkdir()
    (plugin_path / "settings.py").write_text(
        "class PluginSettings:\n    setting_a: int = 0\n"
    )
    (config_dir / "plugin_foo.yaml").write_text(
        yaml.dump({"robot": {"setting_a": 1}})
    )
    (config_dir / "c40_execution.yaml").write_text(
        yaml.dump(
            {
                "robot": {
                    "c40_bridge": "BasicSim",
                    "c40_community_plugins": {"foo": "foo"},
                }
            }
        )
    )

    zip_path = tmp_path / "robot.zip"
    export_profile("robot", zip_path, community_plugins={"foo": "foo"})

    (config_dir / "plugin_foo.yaml").unlink()
    (config_dir / "c40_execution.yaml").unlink()

    import_profile(zip_path)
    cp_data = yaml.safe_load((config_dir / "plugin_foo.yaml").read_text())
    assert cp_data["robot"]["setting_a"] == 1


def test_profile_import_legacy_community_zip_entry(monkeypatch, tmp_path):
    import zipfile

    config_dir = tmp_path / "config"
    plugins_dir = tmp_path / "plugins"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("AVLITE_PLUGINS_DIR", str(plugins_dir))
    set_repo_config_target(False)

    plugin_path = plugins_dir / "foo"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    plugin_path.mkdir()
    (plugin_path / "settings.py").write_text(
        "class PluginSettings:\n    setting_a: int = 0\n"
    )

    zip_path = tmp_path / "robot.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "c40_execution.yaml",
            yaml.dump(
                {
                    "robot": {
                        "c40_bridge": "BasicSim",
                        "c40_community_plugins": {"foo": "foo"},
                    }
                }
            ),
        )
        zf.writestr(
            "community/foo.yaml",
            yaml.dump({"robot": {"setting_a": 2}}),
        )

    import_profile(zip_path)
    cp_data = yaml.safe_load((config_dir / "plugin_foo.yaml").read_text())
    assert cp_data["robot"]["setting_a"] == 2


def test_cli_export_import_profile(monkeypatch, tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(config_dir))
    set_repo_config_target(False)

    (config_dir / "c40_execution.yaml").write_text(
        yaml.dump({"robot": {"c40_bridge": "BasicSim", "c40_community_plugins": {}}})
    )

    zip_path = tmp_path / "robot.zip"
    assert cmd_export_profile(argparse.Namespace(profile="robot", output=str(zip_path))) == 0

    (config_dir / "c40_execution.yaml").unlink()
    assert cmd_import_profile(argparse.Namespace(zip_path=str(zip_path), force=True)) == 0
    assert (config_dir / "c40_execution.yaml").is_file()
