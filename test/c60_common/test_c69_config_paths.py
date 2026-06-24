"""Tests for user config directory resolution."""

from __future__ import annotations

from pathlib import Path

from avlite.c60_common.c67_paths import (
    clear_user_configs,
    copy_repo_configs_to_user,
    effective_config_path,
    get_absolute_path,
    get_config_dir,
    get_data_dir,
    get_startup_profile,
    get_user_configs_dir,
    installed_community_plugins_map,
    is_repo_config_target,
    resolve_plugin_path,
    set_repo_config_target,
    set_startup_profile,
)

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


def test_copy_repo_configs_to_user(monkeypatch, tmp_path):
    src = tmp_path / "bundled"
    src.mkdir()
    (src / "c40_execution.yaml").write_text("default: {}\n")
    dest = tmp_path / "user"
    monkeypatch.setenv("AVLITE_CONFIG_DIR", str(dest))
    monkeypatch.setattr(
        "avlite.c60_common.c67_paths.bundled_config_dir", lambda: src
    )
    copied = copy_repo_configs_to_user()
    assert len(copied) == 1
    assert (dest / "c40_execution.yaml").read_text() == "default: {}\n"


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
